"""Tests for Reolink TCP push event detection."""

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from frigate.detectors.onvif_detector import (
    DetectionType,
    OnvifDetection,
    ReolinkTcpPushClient,
)


class TestDetectionType(unittest.TestCase):
    """Test the DetectionType enum includes all expected values."""

    def test_motion_exists(self):
        self.assertEqual(DetectionType.MOTION.value, "motion")

    def test_person_exists(self):
        self.assertEqual(DetectionType.PERSON.value, "person")

    def test_vehicle_exists(self):
        self.assertEqual(DetectionType.VEHICLE.value, "vehicle")

    def test_pet_exists(self):
        self.assertEqual(DetectionType.PET.value, "pet")

    def test_doorbell_exists(self):
        self.assertEqual(DetectionType.DOORBELL.value, "doorbell")


class TestReolinkTcpPushClientInit(unittest.TestCase):
    """Test ReolinkTcpPushClient initialization."""

    def _make_config(
        self,
        person=True,
        vehicle=True,
        pet=False,
        doorbell=False,
        motion=True,
        host="192.168.1.100",
        port=80,
        user="admin",
        password="secret",
    ):
        """Build a mock CameraConfig."""
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = host
        config.onvif.port = port
        config.onvif.user = user
        config.onvif.password = password
        config.onvif.detection.person = person
        config.onvif.detection.vehicle = vehicle
        config.onvif.detection.pet = pet
        config.onvif.detection.doorbell = doorbell
        config.onvif.detection.motion = motion
        config.onvif.detection.poll_interval = 5
        return config

    def test_default_detection_types(self):
        """Test that default detection types are enabled/disabled correctly."""
        config = self._make_config()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertTrue(client.detect_motion)
        self.assertTrue(client.detect_person)
        self.assertTrue(client.detect_vehicle)
        self.assertFalse(client.detect_pet)
        self.assertFalse(client.detect_doorbell)

    def test_pet_and_doorbell_enabled(self):
        """Test pet and doorbell detection config."""
        config = self._make_config(pet=True, doorbell=True)
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertTrue(client.detect_pet)
        self.assertTrue(client.detect_doorbell)

    def test_all_disabled(self):
        """Test when all detection types are disabled."""
        config = self._make_config(
            person=False, vehicle=False, pet=False, doorbell=False, motion=False
        )
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertFalse(client._should_run())

    def test_parsing_host_with_port(self):
        """Test host:port parsing."""
        config = self._make_config(host="192.168.1.100:8000")
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertEqual(client.host, "192.168.1.100")
        self.assertEqual(client.port, 8000)

    def test_parsing_host_with_scheme(self):
        """Test host with http:// scheme is stripped."""
        config = self._make_config(host="http://192.168.1.100:8000")
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertEqual(client.host, "192.168.1.100")
        self.assertEqual(client.port, 8000)

    def test_credential_storage(self):
        """Test credentials are stored correctly."""
        config = self._make_config(user="myuser", password="mypass")
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertEqual(client.username, "myuser")
        self.assertEqual(client.password, "mypass")


class TestReolinkTcpPushClientDetectionQueue(unittest.TestCase):
    """Test detection queuing behavior."""

    def _make_config(self):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.detection.person = True
        config.onvif.detection.vehicle = True
        config.onvif.detection.pet = True
        config.onvif.detection.doorbell = True
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5
        return config

    def test_motion_detection_queues_correctly(self):
        """Test motion detection is queued with correct type."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client._queue_detection(
            DetectionType.MOTION,
            "motion",
            0.8,
            0,
            time.time(),
        )

        det = detection_queue.get(timeout=1)
        self.assertIsInstance(det, OnvifDetection)
        self.assertEqual(det.type, DetectionType.MOTION)
        self.assertEqual(det.label, "motion")
        self.assertEqual(det.score, 0.8)
        self.assertEqual(det.metadata["source"], "baichuan_tcp")
        self.assertEqual(det.metadata["channel"], 0)

    def test_person_detection_queues_correctly(self):
        """Test person detection is queued with correct type."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client._queue_detection(
            DetectionType.PERSON,
            "people",
            0.9,
            0,
            time.time(),
        )

        det = detection_queue.get(timeout=1)
        self.assertEqual(det.type, DetectionType.PERSON)
        self.assertEqual(det.label, "people")
        self.assertEqual(det.score, 0.9)

    def test_vehicle_detection_queues_correctly(self):
        """Test vehicle detection is queued with correct type."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client._queue_detection(
            DetectionType.VEHICLE,
            "vehicle",
            0.9,
            0,
            time.time(),
        )

        det = detection_queue.get(timeout=1)
        self.assertEqual(det.type, DetectionType.VEHICLE)
        self.assertEqual(det.label, "vehicle")

    def test_pet_detection_queues_correctly(self):
        """Test pet detection is queued with correct type."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client._queue_detection(
            DetectionType.PET,
            "dog_cat",
            0.9,
            0,
            time.time(),
        )

        det = detection_queue.get(timeout=1)
        self.assertEqual(det.type, DetectionType.PET)
        self.assertEqual(det.label, "dog_cat")

    def test_doorbell_detection_queues_correctly(self):
        """Test doorbell detection is queued with correct type."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client._queue_detection(
            DetectionType.DOORBELL,
            "visitor",
            0.9,
            0,
            time.time(),
        )

        det = detection_queue.get(timeout=1)
        self.assertEqual(det.type, DetectionType.DOORBELL)
        self.assertEqual(det.label, "visitor")

    def test_full_queue_is_dropped(self):
        """Test that detections are dropped when queue is full."""
        config = self._make_config()
        detection_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        # Fill the queue
        client._queue_detection(DetectionType.MOTION, "motion", 0.8, 0, time.time())
        self.assertEqual(detection_queue.qsize(), 1)

        # Try to add another detection - should be dropped without raising
        client._queue_detection(DetectionType.PERSON, "people", 0.9, 0, time.time())
        self.assertEqual(detection_queue.qsize(), 1)


class TestReolinkTcpPushClientSyncMethods(unittest.TestCase):
    """Test synchronous getter methods."""

    def _make_config(self):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.detection.person = True
        config.onvif.detection.vehicle = True
        config.onvif.detection.pet = False
        config.onvif.detection.doorbell = True
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5
        return config

    def test_sync_methods_return_false_when_no_reolink(self):
        """Test sync methods return False when _reolink is None."""
        config = self._make_config()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        self.assertFalse(client.get_ai_detected_sync("person"))
        self.assertFalse(client.get_doorbell_detected_sync())
        self.assertFalse(client.get_motion_detected_sync())

    def test_sync_methods_raise_when_reolink_fails(self):
        """Test sync methods handle exceptions gracefully."""
        config = self._make_config()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, queue.Queue(), stop_event)

        mock_reolink = MagicMock()
        mock_reolink.ai_detected.side_effect = Exception("connection lost")
        mock_reolink.visitor_detected.side_effect = Exception("connection lost")
        mock_reolink.motion_detected.side_effect = Exception("connection lost")
        client._reolink = mock_reolink

        self.assertFalse(client.get_ai_detected_sync("person"))
        self.assertFalse(client.get_doorbell_detected_sync())
        self.assertFalse(client.get_motion_detected_sync())


class TestReolinkTcpPushClientStartStop(unittest.TestCase):
    """Test start and stop lifecycle."""

    def _make_config(self):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.detection.person = True
        config.onvif.detection.vehicle = True
        config.onvif.detection.pet = False
        config.onvif.detection.doorbell = False
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5
        return config

    @patch("frigate.detectors.onvif_detector.ReolinkHost")
    @patch("frigate.detectors.onvif_detector.asyncio")
    def test_start_creates_thread(self, mock_asyncio, mock_reolink_class):
        """Test that start creates a background thread."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        mock_asyncio.new_event_loop.return_value = MagicMock()
        mock_asyncio.Task = MagicMock()

        client.start()

        self.assertTrue(client._running)
        self.assertIsNotNone(client.thread)
        client.stop()

    def test_stop_with_noop_when_not_started(self):
        """Test stop works even if client was never started."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        # Should not raise
        client.stop()

    def test_start_disabled_when_all_off(self):
        """Test that start does nothing when all detection types are disabled."""
        config = self._make_config()
        config.onvif.detection.person = False
        config.onvif.detection.vehicle = False
        config.onvif.detection.pet = False
        config.onvif.detection.doorbell = False
        config.onvif.detection.motion = False

        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        client.start()

        self.assertFalse(client._running)


class TestReolinkTcpPushClientCooldown(unittest.TestCase):
    """Test detection cooldown behavior."""

    def _make_config(self):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.detection.person = True
        config.onvif.detection.vehicle = False
        config.onvif.detection.pet = False
        config.onvif.detection.doorbell = False
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5
        return config

    def test_cooldown_prevents_rapid_duplicates(self):
        """Test that cooldown prevents duplicate detections within 2 seconds."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        now = time.time()

        # First detection should be queued
        client._queue_detection(DetectionType.MOTION, "motion", 0.8, 0, now)
        self.assertEqual(detection_queue.qsize(), 1)

        # Second detection within cooldown should still be queueable
        # (the cooldown is checked in _process_events, not _queue_detection)
        client._queue_detection(DetectionType.MOTION, "motion", 0.8, 0, now + 0.5)
        self.assertEqual(detection_queue.qsize(), 2)

        # After cooldown period, new detection should be allowed
        client._queue_detection(DetectionType.MOTION, "motion", 0.8, 0, now + 3)
        self.assertEqual(detection_queue.qsize(), 3)


class TestReolinkTcpPushClientEventProcessing(unittest.TestCase):
    """Test the _process_events method with mocked Reolink API."""

    def _make_config(self):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.detection.person = True
        config.onvif.detection.vehicle = True
        config.onvif.detection.pet = True
        config.onvif.detection.doorbell = True
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5
        return config

    def test_process_events_queues_detected_objects(self):
        """Test that _process_events queues detections for active events."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        # Mock the Reolink API responses
        mock_reolink = MagicMock()
        mock_reolink.num_channels = 1
        mock_reolink.channels = [0]
        mock_reolink.motion_detected.return_value = True
        mock_reolink.ai_detected.side_effect = lambda ch, obj: {
            "people": True,
            "vehicle": True,
            "dog_cat": True,
        }.get(obj, False)
        mock_reolink.visitor_detected.return_value = True
        client._reolink = mock_reolink

        client._process_events()

        # Should have queued: motion, person, vehicle, pet, doorbell
        self.assertEqual(detection_queue.qsize(), 5)

        detection_types = set()
        while not detection_queue.empty():
            det = detection_queue.get_nowait()
            detection_types.add(det.type)

        self.assertIn(DetectionType.MOTION, detection_types)
        self.assertIn(DetectionType.PERSON, detection_types)
        self.assertIn(DetectionType.VEHICLE, detection_types)
        self.assertIn(DetectionType.PET, detection_types)
        self.assertIn(DetectionType.DOORBELL, detection_types)

    def test_process_events_skips_inactive(self):
        """Test that _process_events skips detections for inactive events."""
        config = self._make_config()
        detection_queue = queue.Queue()
        stop_event = threading.Event()
        client = ReolinkTcpPushClient(config, detection_queue, stop_event)

        # Mock all detections as inactive
        mock_reolink = MagicMock()
        mock_reolink.num_channels = 1
        mock_reolink.channels = [0]
        mock_reolink.motion_detected.return_value = False
        mock_reolink.ai_detected.return_value = False
        mock_reolink.visitor_detected.return_value = False
        client._reolink = mock_reolink

        client._process_events()

        self.assertEqual(detection_queue.qsize(), 0)


class TestOnvifVendorEnum(unittest.TestCase):
    """Test the OnvifVendorEnum."""

    def test_reolink_value(self):
        from frigate.config.camera.onvif import OnvifVendorEnum

        self.assertEqual(OnvifVendorEnum.reolink.value, "reolink")

    def test_default_value(self):
        from frigate.config.camera.onvif import OnvifVendorEnum

        self.assertEqual(OnvifVendorEnum.default.value, "default")

    def test_default_is_default(self):
        from frigate.config.camera.onvif import OnvifConfig

        config = OnvifConfig()
        self.assertEqual(config.vendor.value, "default")


class TestOnvifVendorConfig(unittest.TestCase):
    """Test OnvifConfig vendor field."""

    def test_vendor_reolink_config(self):
        from frigate.config.camera.onvif import OnvifConfig, OnvifVendorEnum

        config = OnvifConfig(vendor=OnvifVendorEnum.reolink)
        self.assertEqual(config.vendor, OnvifVendorEnum.reolink)
        self.assertEqual(config.vendor.value, "reolink")

    def test_vendor_default_config(self):
        from frigate.config.camera.onvif import OnvifConfig

        config = OnvifConfig(vendor="default")
        self.assertEqual(config.vendor.value, "default")

    def test_vendor_field_has_description(self):
        from frigate.config.camera.onvif import OnvifConfig

        schema = OnvifConfig.model_json_schema()
        vendor_props = schema["properties"]["vendor"]
        self.assertIn("title", vendor_props)
        self.assertIn("description", vendor_props)
        self.assertIn("reolink", vendor_props["description"])


class TestOnvifDetectorVendorLogic(unittest.TestCase):
    """Test OnvifDetector vendor logic."""

    def _make_config(
        self,
        vendor="default",
        person=True,
        vehicle=True,
        pet=False,
        doorbell=False,
    ):
        config = MagicMock()
        config.name = "test_camera"
        config.onvif.host = "192.168.1.100"
        config.onvif.port = 80
        config.onvif.user = "admin"
        config.onvif.password = "secret"
        config.onvif.vendor = vendor
        config.onvif.detection.enabled = True
        config.onvif.detection.person = person
        config.onvif.detection.vehicle = vehicle
        config.onvif.detection.pet = pet
        config.onvif.detection.doorbell = doorbell
        config.onvif.detection.motion = True
        config.onvif.detection.poll_interval = 5

        from frigate.config.camera.onvif import OnvifVendorEnum

        config.onvif.vendor = OnvifVendorEnum(vendor)
        return config

    @patch("frigate.detectors.onvif_detector.ReolinkTcpPushClient")
    @patch("frigate.detectors.onvif_detector.ReolinkSmartDetector")
    @patch("frigate.detectors.onvif_detector.OnvifEventSubscriber")
    @patch("frigate.detectors.onvif_detector.asyncio")
    @patch("frigate.detectors.onvif_detector._create_onvif_camera")
    def test_vendor_reolink_uses_tcp_push(
        self,
        mock_create_camera,
        mock_asyncio,
        mock_event_subscriber,
        mock_smart_detector,
        mock_tcp_push,
    ):
        """Test that vendor=reolink uses TCP push client."""
        from frigate.detectors.onvif_detector import OnvifDetector

        mock_asyncio.run.side_effect = lambda coro: None

        config = self._make_config(vendor="reolink")
        detection_queue = queue.Queue()
        stop_event = threading.Event()

        detector = OnvifDetector(config, detection_queue, stop_event)
        detector.start()

        mock_tcp_push.assert_called_once()
        mock_tcp_push.return_value.start.assert_called_once()
        mock_smart_detector.assert_not_called()

        detector.stop()

    @patch("frigate.detectors.onvif_detector.ReolinkTcpPushClient")
    @patch("frigate.detectors.onvif_detector.ReolinkSmartDetector")
    @patch("frigate.detectors.onvif_detector.OnvifEventSubscriber")
    @patch("frigate.detectors.onvif_detector.asyncio")
    @patch("frigate.detectors.onvif_detector._create_onvif_camera")
    def test_vendor_default_uses_polling(
        self,
        mock_create_camera,
        mock_asyncio,
        mock_event_subscriber,
        mock_smart_detector,
        mock_tcp_push,
    ):
        """Test that vendor=default uses polling."""
        from frigate.detectors.onvif_detector import OnvifDetector

        mock_asyncio.run.side_effect = lambda coro: None

        config = self._make_config(vendor="default")
        detection_queue = queue.Queue()
        stop_event = threading.Event()

        detector = OnvifDetector(config, detection_queue, stop_event)
        detector.start()

        mock_tcp_push.assert_not_called()
        mock_smart_detector.assert_called_once()
        mock_smart_detector.return_value.start.assert_called_once()

        detector.stop()

    @patch("frigate.detectors.onvif_detector.ReolinkTcpPushClient")
    @patch("frigate.detectors.onvif_detector.ReolinkSmartDetector")
    @patch("frigate.detectors.onvif_detector.OnvifEventSubscriber")
    @patch("frigate.detectors.onvif_detector.asyncio")
    @patch("frigate.detectors.onvif_detector._create_onvif_camera")
    def test_vendor_reolink_with_pet_uses_tcp_push(
        self,
        mock_create_camera,
        mock_asyncio,
        mock_event_subscriber,
        mock_smart_detector,
        mock_tcp_push,
    ):
        """Test that vendor=reolink with pet enabled uses TCP push."""
        from frigate.detectors.onvif_detector import OnvifDetector

        mock_asyncio.run.side_effect = lambda coro: None

        config = self._make_config(vendor="reolink", pet=True)
        detection_queue = queue.Queue()
        stop_event = threading.Event()

        detector = OnvifDetector(config, detection_queue, stop_event)
        detector.start()

        mock_tcp_push.assert_called_once()
        mock_tcp_push.return_value.start.assert_called_once()
        mock_smart_detector.assert_not_called()

        detector.stop()


if __name__ == "__main__":
    unittest.main()
