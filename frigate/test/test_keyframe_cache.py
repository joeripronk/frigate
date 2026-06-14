"""Tests for KeyframeCache and keyframe parsing in frigate.util.media."""

import time
import unittest

from frigate.util.media import KeyframeCache, _parse_keyframe_packets


class TestKeyframeCache(unittest.TestCase):
    """Test KeyframeCache singleton behavior."""

    def setUp(self):
        # Reset the singleton for each test
        KeyframeCache._instance = None

    def test_singleton(self):
        """Multiple calls return the same instance."""
        cache1 = KeyframeCache()
        cache2 = KeyframeCache()
        self.assertIs(cache1, cache2)

    def test_set_and_get(self):
        """Basic set/get roundtrip."""
        cache = KeyframeCache()
        cache.set("/path/to/segment.mp4", [0, 1000, 2000, 3000])
        result = cache.get("/path/to/segment.mp4")
        self.assertEqual(result, [0, 1000, 2000, 3000])

    def test_get_missing_path(self):
        """Returns None for a path that was never cached."""
        cache = KeyframeCache()
        self.assertIsNone(cache.get("/nonexistent.mp4"))

    def test_lru_eviction(self):
        """Old entries are evicted when max_size is exceeded."""
        cache = KeyframeCache()
        cache._max_size = 3
        for i in range(5):
            cache.set(f"/path{i}.mp4", [i * 1000])
        # Only the last 3 should remain
        self.assertIsNone(cache.get("/path0.mp4"))
        self.assertIsNone(cache.get("/path1.mp4"))
        self.assertEqual(cache.get("/path2.mp4"), [2000])
        self.assertEqual(cache.get("/path3.mp4"), [3000])
        self.assertEqual(cache.get("/path4.mp4"), [4000])

    def test_lru_touch(self):
        """Accessing an entry moves it to the end (LRU behavior)."""
        cache = KeyframeCache()
        cache._max_size = 3
        cache.set("/a.mp4", [1000])
        cache.set("/b.mp4", [2000])
        cache.set("/c.mp4", [3000])
        # Touch /a to make it recently used
        cache.get("/a.mp4")
        cache.set("/d.mp4", [4000])  # Should evict /b (least recently used)
        self.assertEqual(cache.get("/a.mp4"), [1000])
        self.assertIsNone(cache.get("/b.mp4"))
        self.assertEqual(cache.get("/c.mp4"), [3000])
        self.assertEqual(cache.get("/d.mp4"), [4000])

    def test_ttl_expiry(self):
        """Entries older than TTL are expired."""
        cache = KeyframeCache()
        cache.set("/path.mp4", [1000])
        # Manually age the entry beyond TTL
        cache._cache["/path.mp4"] = (time.monotonic() - cache._ttl - 1, [1000])
        self.assertIsNone(cache.get("/path.mp4"))


class TestParseKeyframePackets(unittest.TestCase):
    """Test _parse_keyframe_packets function."""

    def test_parses_keyframe_timestamps(self):
        output = (
            b"0.000000,K__\n0.033333,___\n1.000000,K__\n2.000000,___\n3.000000,K__\n"
        )
        result = _parse_keyframe_packets(output)
        self.assertEqual(result, [0, 1000, 3000])

    def test_skips_non_keyframe_frames(self):
        output = b"0.000000,___\n0.033333,___\n1.000000,K__\n"
        result = _parse_keyframe_packets(output)
        self.assertEqual(result, [1000])

    def test_skips_unparseable_lines(self):
        output = b"N/A,K__\n\n1.0,K__\nbad line\n2.0,K__\n"
        result = _parse_keyframe_packets(output)
        self.assertEqual(result, [1000, 2000])

    def test_empty_output(self):
        result = _parse_keyframe_packets(b"")
        self.assertEqual(result, [])

    def test_no_keyframes(self):
        output = b"0.033333,___\n0.066667,___\n"
        result = _parse_keyframe_packets(output)
        self.assertEqual(result, [])

    def test_handles_malformed_flags(self):
        # Only flags containing "K" are treated as keyframes
        output = b"1.0,___\n2.0,K__\n3.0,___\n"
        result = _parse_keyframe_packets(output)
        self.assertEqual(result, [2000])


if __name__ == "__main__":
    unittest.main()
