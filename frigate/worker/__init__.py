"""Standalone worker entry points for Rust subprocess orchestration.

Each module can be run via `python3 -m frigate.worker.<module>` and will:
1. Load config independently
2. Set up its own logging (stderr-based, not shared with main process)
3. Use signal-based shutdown (SIGTERM/SIGINT)
4. Run the worker's main loop

This is the Phase 1 approach per rust.md section 10b: Python workers run
as fresh subprocesses (no forkserver preload), each importing everything
from scratch. The performance-critical video pipeline (Phase 6) will be
in Rust.
"""
