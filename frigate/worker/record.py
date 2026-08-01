"""Standalone recording worker for Rust subprocess orchestration.

Run via: python3 -m frigate.worker.record --config /path/to/config.yml
"""

import argparse
import logging
import signal
import sys
import threading

from frigate.config import FrigateConfig
from frigate.record.maintainer import RecordingMaintainer


logger = logging.getLogger("frigate.recording_manager")


def setup_logging() -> None:
    """Set up standalone logging for this subprocess."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(prog="frigate.worker.record")
    parser.add_argument("--config", required=True, help="Path to config.yml")
    args = parser.parse_args()

    setup_logging()

    # Load config independently
    config = FrigateConfig.load(install=True)

    # Signal-based shutdown
    stop_event = threading.Event()

    def handle_sigterm(signum, frame):
        stop_event.set()

    def handle_sigint(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)

    # Start recording maintainer (threading.Event works at runtime despite type mismatch)
    maintainer = RecordingMaintainer(config, stop_event)  # type: ignore[arg-type]
    maintainer.start()

    # Wait for shutdown
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass

    maintainer.join(timeout=10)
    logger.info("Recording worker stopped")


if __name__ == "__main__":
    main()
