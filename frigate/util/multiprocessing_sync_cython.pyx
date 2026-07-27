"""
Cython-accelerated multiprocessing synchronization primitives.

Provides Cython implementations of EventsPerSecond and InferenceSpeed
for reduced Python overhead in tight detection and processing loops.

Usage:
    from frigate.util.multiprocessing_sync_cython import (
        CythonEventsPerSecond,
        CythonInferenceSpeed,
    )

    # EventsPerSecond replacement
    eps = CythonEventsPerSecond(max_events=1000, last_n_seconds=10)
    eps.update()  # called per event
    rate = eps.eps()  # get current rate

    # InferenceSpeed replacement (works with multiprocessing.Value)
    speed = CythonInferenceSpeed(value_proxy)
    speed.update(inference_time)  # called after each detection
    current = speed.current()  # get EMA value
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import logging
import time

from libc.stdlib cimport malloc, free

logger = logging.getLogger(__name__)


cdef inline double _ema_update(double old_value, double new_value, double weight) noexcept:
    """Exponential moving average update: new = old*weight + new*(1-weight)

    Args:
        old_value: Previous EMA value
        new_value: New observation
        weight: EMA weight between 0.0 and 1.0

    Returns:
        Updated EMA value
    """
    return old_value * weight + new_value * (1.0 - weight)


cdef class CythonEventsPerSecond:
    """High-performance EventsPerSecond using a C circular buffer.

    Replaces the Python deque-based EventsPerSecond with a fixed-size
    C array for O(1) appends and reduced memory allocation overhead.

    The circular buffer uses a C-level pointer which Cython compiles
    to direct C array accesses, eliminating Python object overhead
    per timestamp.

    Args:
        max_events: Maximum number of events to track (also buffer capacity)
        last_n_seconds: Sliding window size in seconds (default: 10)

    Example:
        >>> eps = CythonEventsPerSecond()
        >>> for _ in range(100):
        ...     eps.update()
        ...     time.sleep(0.01)
        >>> print(f"Rate: {eps.eps():.1f} eps")
    """

    cdef double *_timestamps
    cdef long _head
    cdef long _count
    cdef int _max_events
    cdef double _last_n_seconds
    cdef object _start

    def __init__(self, max_events=1000, last_n_seconds=10) -> None:
        self._timestamps = <double*>malloc(max_events * sizeof(double))
        if not self._timestamps:
            raise MemoryError("Failed to allocate timestamp buffer")
        self._head = 0
        self._count = 0
        self._max_events = max_events
        self._last_n_seconds = last_n_seconds
        self._start = None

    def __dealloc__(self):
        if self._timestamps:
            free(self._timestamps)
            self._timestamps = NULL

    def start(self) -> None:
        """Start the timer by recording the initial monotonic time."""
        self._start = time.monotonic()

    def update(self) -> None:
        """Record an event timestamp in the circular buffer.

        O(1) amortized operation — appends to the circular buffer
        and advances the head pointer. When the buffer is full,
        the oldest entries are naturally overwritten.
        """
        cdef double now = time.monotonic()

        if self._start is None:
            self._start = now

        if self._count < self._max_events:
            self._timestamps[self._head] = now
            self._head = (self._head + 1) % self._max_events
            self._count += 1
        else:
            self._timestamps[self._head] = now
            self._head = (self._head + 1) % self._max_events

    def expire_timestamps(self, now: float) -> None:
        """Remove timestamps outside the sliding window.

        Scans from the tail and counts how many entries are within
        the window, updating _count to reflect only valid entries.

        Args:
            now: Current monotonic time
        """
        cdef:
            double threshold = now - self._last_n_seconds
            long tail
            long new_tail

        if self._count == 0:
            return

        tail = (self._head - self._count) % self._max_events
        if tail < 0:
            tail += self._max_events

        while tail != self._head and self._timestamps[tail] < threshold:
            new_tail = (tail + 1) % self._max_events
            self._count -= 1
            tail = new_tail

    def eps(self) -> float:
        """Calculate current events per second in the sliding window.

        Expires old timestamps, then computes rate as count / elapsed
        time (capped at last_n_seconds to avoid division by zero).

        Returns:
            Events per second (float), 0.0 if no events in window
        """
        cdef:
            double now = time.monotonic()
            double seconds

        if self._start is None:
            self._start = now

        self.expire_timestamps(now)

        if self._count == 0:
            return 0.0

        seconds = now - self._start
        if seconds > self._last_n_seconds:
            seconds = self._last_n_seconds

        if seconds == 0:
            seconds = 1.0

        return self._count / seconds


cdef class CythonInferenceSpeed:
    """High-performance InferenceSpeed with Cython-typed EMA computation.

    Reduces Python overhead in the EMA calculation loop by keeping
    the weight and initialized flag in C-level variables. The metric
    value is accessed via Python's ValueProxy.value but the arithmetic
    is performed in C.

    This is most beneficial when called thousands of times per second
    (e.g., after every object detection).

    Args:
        metric: A multiprocessing.Value or ValueProxy containing a float
        weight: EMA weight (0.0-1.0, default: 0.9 for 10% new data per update)

    Example:
        >>> from multiprocessing import Value
        >>> val = Value('d', 0.0)  # double
        >>> speed = CythonInferenceSpeed(val)
        >>> speed.update(0.05)  # 50ms inference
        >>> print(f"EMA: {speed.current():.4f}s")
    """

    cdef object _metric
    cdef double _weight
    cdef bint _initialized

    def __init__(self, metric, weight=0.9) -> None:
        self._metric = metric
        self._weight = weight
        self._initialized = False

    def update(self, inference_time: float) -> None:
        """Update the EMA with a new inference time.

        Uses Cython-typed arithmetic for the EMA calculation while
        accessing the metric's .value via Python's ValueProxy.

        Args:
            inference_time: Inference time in seconds (float)
        """
        cdef double old_value, new_value

        if not self._initialized:
            self._metric.value = inference_time
            self._initialized = True
            return

        old_value = self._metric.value
        new_value = _ema_update(old_value, inference_time, self._weight)
        self._metric.value = new_value

    def current(self) -> float:
        """Get the current EMA inference speed.

        Returns:
            Current EMA value in seconds
        """
        return self._metric.value
