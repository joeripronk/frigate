"""Tests for CameraMaintainer SHM cleanup on camera remove.

Regression coverage for the case where a camera is removed and then a
new camera is added with the same name. Without unlinking the per-frame
YUV SHM slots, the maintainer's frame_manager.create call hits
FileExistsError and falls back to reopening the existing segment at the
*old* size, which the new ffmpeg process then writes mismatched-size
frames into.
"""

import unittest
from unittest.mock import MagicMock, patch

from frigate.camera.maintainer import CameraMaintainer


class TestMaintainerUnlinkFrameSlotsOnRemove(unittest.TestCase):
    def _make_maintainer(self) -> CameraMaintainer:
        """Build a maintainer without invoking __init__ (avoids needing real
        FrigateConfig, queues, multiprocessing manager, etc.). We're only
        exercising the SHM-cleanup helper, so the surrounding init is
        irrelevant."""
        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.frame_manager = MagicMock()
        return maintainer

    def test_unlinks_only_segments_with_matching_prefix(self) -> None:
        maintainer = self._make_maintainer()
        maintainer.frame_manager.shm_store = {
            "front_frame0": object(),
            "front_frame1": object(),
            "front_frame2": object(),
            # Different camera; must not be touched.
            "side_frame0": object(),
            # Detector input/output buffers are sized by the model and
            # cached by the long-lived DetectorRunner — must not be
            # touched even when their owning camera is removed.
            "front": object(),
            "out-front": object(),
        }
        # Per-camera tracking set — the optimization path.
        maintainer.camera_shm_slots = {
            "front": {"front_frame0", "front_frame1", "front_frame2"},
        }

        # __name-mangled access from outside the class.
        maintainer._CameraMaintainer__unlink_camera_frame_slots("front")

        deleted = [c.args[0] for c in maintainer.frame_manager.delete.call_args_list]
        self.assertEqual(
            sorted(deleted),
            ["front_frame0", "front_frame1", "front_frame2"],
        )

    def test_handles_camera_with_no_slots(self) -> None:
        """Cameras that were removed before any frame slot was ever
        created (e.g. cancelled during preparing_clip) should be a no-op."""
        maintainer = self._make_maintainer()
        maintainer.frame_manager.shm_store = {"other_frame0": object()}
        # No tracking entries for "front" — triggers fallback path.
        maintainer.camera_shm_slots = {}

        maintainer._CameraMaintainer__unlink_camera_frame_slots("front")

        maintainer.frame_manager.delete.assert_not_called()

    def test_swallows_delete_errors(self) -> None:
        """Unlink failures shouldn't abort the remove loop — best-effort."""
        maintainer = self._make_maintainer()
        maintainer.frame_manager.shm_store = {
            "front_frame0": object(),
            "front_frame1": object(),
        }
        maintainer.camera_shm_slots = {
            "front": {"front_frame0", "front_frame1"},
        }
        maintainer.frame_manager.delete.side_effect = OSError("simulated")

        # Both slots are attempted; the OSError on the first doesn't
        # prevent the second from being tried.
        with patch("frigate.camera.maintainer.logger"):
            maintainer._CameraMaintainer__unlink_camera_frame_slots("front")

        self.assertEqual(maintainer.frame_manager.delete.call_count, 2)

    def test_fallback_prefix_scan_when_no_tracking(self) -> None:
        """When camera_shm_slots lacks this camera (e.g. pre-optimization
        deployments), the method falls back to a prefix scan of shm_store."""
        maintainer = self._make_maintainer()
        maintainer.frame_manager.shm_store = {
            "front_frame0": object(),
            "front_frame1": object(),
            "other_frame0": object(),
            "front": object(),
            "out-front": object(),
        }
        # No tracking entry for "front" — triggers the fallback path.
        maintainer.camera_shm_slots = {}

        # __name-mangled access from outside the class.
        maintainer._CameraMaintainer__unlink_camera_frame_slots("front")

        deleted = [c.args[0] for c in maintainer.frame_manager.delete.call_args_list]
        self.assertEqual(
            sorted(deleted),
            ["front_frame0", "front_frame1"],
        )


class TestMaintainerParallelRecycling(unittest.TestCase):
    """Tests for parallel camera recycling (issue #2 fix)."""

    def _make_maintainer(self) -> CameraMaintainer:
        """Build a maintainer without invoking __init__."""
        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.frame_manager = MagicMock()
        maintainer.camera_shm_slots = {}
        maintainer.camera_processes = {}
        maintainer.capture_processes = {}
        maintainer.camera_stop_events = {}
        maintainer.camera_metrics = {}
        maintainer.ptz_metrics = {}
        maintainer.config = MagicMock()
        return maintainer

    def test_recycle_camera_calls_correct_sequence(self) -> None:
        """Recycling should stop capture, stop processor, unlink SHM,
        and then start processor and capture."""
        maintainer = self._make_maintainer()
        new_config = MagicMock()
        maintainer.camera_processes["replay_test"] = MagicMock()
        maintainer.capture_processes["replay_test"] = MagicMock()
        maintainer.camera_shm_slots["replay_test"] = {"replay_test_frame0"}
        maintainer.camera_metrics["replay_test"] = MagicMock()
        maintainer.camera_metrics["replay_test"].frame_queue = MagicMock()
        maintainer.ptz_metrics["replay_test"] = MagicMock()

        maintainer._CameraMaintainer__recycle_camera("replay_test", new_config)

        # Verify ffmpeg cmds rebuilt
        new_config.recreate_ffmpeg_cmds.assert_called_once()
        # Verify SHM slots were cleared
        self.assertNotIn("replay_test", maintainer.camera_shm_slots)

    def test_recycle_camera_handles_missing_processes(self) -> None:
        """Recycling a camera with no running processes should not crash."""
        maintainer = self._make_maintainer()
        new_config = MagicMock()
        maintainer.camera_processes = {}
        maintainer.capture_processes = {}

        with patch("frigate.camera.maintainer.logger"):
            maintainer._CameraMaintainer__recycle_camera("replay_test", new_config)

        new_config.recreate_ffmpeg_cmds.assert_called_once()

    def test_parallel_recycling_completes(self) -> None:
        """Multiple cameras should be recycled in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        cameras = [f"replay_{i}" for i in range(4)]
        configs = {cam: MagicMock() for cam in cameras}

        maintainer = self._make_maintainer()
        for cam in cameras:
            maintainer.camera_processes[cam] = MagicMock()
            maintainer.capture_processes[cam] = MagicMock()
            maintainer.camera_shm_slots[cam] = {f"{cam}_frame0"}
            maintainer.camera_metrics[cam] = MagicMock()
            maintainer.camera_metrics[cam].frame_queue = MagicMock()
            maintainer.ptz_metrics[cam] = MagicMock()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    maintainer._CameraMaintainer__recycle_camera, cam, config
                ): cam
                for cam, config in configs.items()
            }
            for future in futures:
                future.result()

        # All cameras should be recycled (shm_slots cleared)
        for cam in cameras:
            self.assertNotIn(cam, maintainer.camera_shm_slots)

        # All configs should have ffmpeg cmds rebuilt
        for cam, config in configs.items():
            config.recreate_ffmpeg_cmds.assert_called_once()

    def test_parallel_recycling_error_handling(self) -> None:
        """If one camera recycling fails, others should still complete."""
        from concurrent.futures import ThreadPoolExecutor

        cameras = ["replay_good", "replay_bad", "replay_also_good"]
        configs = {cam: MagicMock() for cam in cameras}

        maintainer = self._make_maintainer()
        for cam in cameras:
            maintainer.camera_processes[cam] = MagicMock()
            maintainer.capture_processes[cam] = MagicMock()
            maintainer.camera_shm_slots[cam] = {f"{cam}_frame0"}
            maintainer.camera_metrics[cam] = MagicMock()
            maintainer.camera_metrics[cam].frame_queue = MagicMock()
            maintainer.ptz_metrics[cam] = MagicMock()

        # Make one camera fail
        original_recycle = maintainer._CameraMaintainer__recycle_camera

        def failing_recycle(camera: str, config: MagicMock) -> None:
            if camera == "replay_bad":
                raise RuntimeError("simulated failure")
            return original_recycle(camera, config)

        with patch.object(
            maintainer, "_CameraMaintainer__recycle_camera", side_effect=failing_recycle
        ):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        maintainer._CameraMaintainer__recycle_camera, cam, config
                    ): cam
                    for cam, config in configs.items()
                }
                # Track which futures raised
                failed_cameras = []
                for future in futures:
                    camera = futures[future]
                    try:
                        future.result()
                    except RuntimeError:
                        failed_cameras.append(camera)

        # replay_bad should have failed
        self.assertEqual(failed_cameras, ["replay_bad"])

        # Good cameras should still be recycled
        self.assertNotIn("replay_good", maintainer.camera_shm_slots)
        self.assertNotIn("replay_also_good", maintainer.camera_shm_slots)


class TestMaintainerParallelRemove(unittest.TestCase):
    """Tests for parallel camera removal."""

    def _make_maintainer(self) -> CameraMaintainer:
        """Build a maintainer without invoking __init__."""
        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.frame_manager = MagicMock()
        maintainer.camera_shm_slots = {}
        maintainer.camera_processes = {}
        maintainer.capture_processes = {}
        maintainer.camera_stop_events = {}
        maintainer.camera_metrics = {}
        maintainer.ptz_metrics = {}
        maintainer.region_grids = {}
        return maintainer

    def test_remove_camera_calls_correct_sequence(self) -> None:
        """Removing a camera should stop processes, unlink SHM, and clean
        all tracking dicts."""
        maintainer = self._make_maintainer()
        maintainer.camera_processes["remove_test"] = MagicMock()
        maintainer.capture_processes["remove_test"] = MagicMock()
        maintainer.camera_shm_slots["remove_test"] = {"remove_test_frame0"}
        maintainer.camera_stop_events["remove_test"] = MagicMock()
        maintainer.region_grids["remove_test"] = MagicMock()
        maintainer.camera_metrics["remove_test"] = MagicMock()
        maintainer.camera_metrics["remove_test"].frame_queue = MagicMock()
        maintainer.ptz_metrics["remove_test"] = MagicMock()

        maintainer._CameraMaintainer__remove_camera("remove_test")

        # All tracking dicts should be cleared
        self.assertNotIn("remove_test", maintainer.camera_shm_slots)
        self.assertNotIn("remove_test", maintainer.camera_processes)
        self.assertNotIn("remove_test", maintainer.capture_processes)
        self.assertNotIn("remove_test", maintainer.camera_stop_events)
        self.assertNotIn("remove_test", maintainer.region_grids)
        self.assertNotIn("remove_test", maintainer.camera_metrics)
        self.assertNotIn("remove_test", maintainer.ptz_metrics)

    def test_remove_camera_handles_missing_processes(self) -> None:
        """Removing a camera with no running processes should not crash."""
        maintainer = self._make_maintainer()
        maintainer.camera_processes = {}
        maintainer.capture_processes = {}

        with patch("frigate.camera.maintainer.logger"):
            maintainer._CameraMaintainer__remove_camera("remove_test")

    def test_parallel_remove_completes(self) -> None:
        """Multiple cameras should be removed in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        cameras = [f"remove_{i}" for i in range(4)]

        maintainer = self._make_maintainer()
        for cam in cameras:
            maintainer.camera_processes[cam] = MagicMock()
            maintainer.capture_processes[cam] = MagicMock()
            maintainer.camera_shm_slots[cam] = {f"{cam}_frame0"}
            maintainer.camera_stop_events[cam] = MagicMock()
            maintainer.region_grids[cam] = MagicMock()
            maintainer.camera_metrics[cam] = MagicMock()
            maintainer.camera_metrics[cam].frame_queue = MagicMock()
            maintainer.ptz_metrics[cam] = MagicMock()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(maintainer._CameraMaintainer__remove_camera, cam): cam
                for cam in cameras
            }
            for future in futures:
                future.result()

        # All cameras should be removed from all tracking dicts
        for cam in cameras:
            self.assertNotIn(cam, maintainer.camera_shm_slots)
            self.assertNotIn(cam, maintainer.camera_processes)
            self.assertNotIn(cam, maintainer.capture_processes)


class TestMaintainerBoundedJoin(unittest.TestCase):
    """Tests that process join calls have bounded timeouts."""

    def _make_maintainer(self) -> CameraMaintainer:
        """Build a maintainer without invoking __init__."""
        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.camera_shm_slots = {}
        maintainer.camera_processes = {}
        maintainer.capture_processes = {}
        maintainer.camera_stop_events = {}
        maintainer.camera_metrics = {}
        maintainer.ptz_metrics = {}
        return maintainer

    def test_stop_capture_has_bounded_join(self) -> None:
        """__stop_camera_capture_process should have bounded join timeouts.

        The second join after terminate() should also have a timeout,
        and kill() should be called as a final fallback if the process
        doesn't exit after termination.
        """
        from unittest.mock import PropertyMock

        maintainer = self._make_maintainer()

        # Process that refuses to exit after SIGTERM
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True

        maintainer.capture_processes["test_cam"] = mock_process
        maintainer.camera_stop_events["test_cam"] = MagicMock()

        # Time out immediately on first join, then always alive after terminate
        type(mock_process).is_alive = PropertyMock(side_effect=[True, True, True, True])
        mock_process.join.side_effect = [None, None, None, None]

        maintainer._CameraMaintainer__stop_camera_capture_process("test_cam")

        # Verify join was called with a timeout on the second call (after terminate)
        # join calls: first join(10), then after terminate: join(5), then after kill: join(2)
        self.assertEqual(mock_process.join.call_count, 3)
        # Second join should have timeout=5
        calls = mock_process.join.call_args_list
        self.assertEqual(calls[1].kwargs.get("timeout"), 5)
        # kill() should have been called as final fallback
        mock_process.kill.assert_called_once()

    def test_stop_process_has_bounded_join(self) -> None:
        """__stop_camera_process should have bounded join timeouts."""
        from unittest.mock import PropertyMock

        maintainer = self._make_maintainer()
        maintainer.camera_metrics["test_cam"] = MagicMock()
        maintainer.camera_metrics["test_cam"].frame_queue = MagicMock()

        mock_process = MagicMock()
        type(mock_process).is_alive = PropertyMock(side_effect=[True, True, True, True])
        mock_process.join.side_effect = [None, None, None, None]

        maintainer.camera_processes["test_cam"] = mock_process
        maintainer.camera_stop_events["test_cam"] = MagicMock()

        maintainer._CameraMaintainer__stop_camera_process("test_cam")

        # join was called 3 times: (10), (5), (2)
        self.assertEqual(mock_process.join.call_count, 3)
        calls = mock_process.join.call_args_list
        self.assertEqual(calls[1].kwargs.get("timeout"), 5)
        mock_process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
