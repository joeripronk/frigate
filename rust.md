# Rewrite plan: orchestrator (`app.py`) → Rust

## Overview

Replace `frigate/app.py` + `frigate/__main__.py` (orchestrator: ~900 lines, process manager, DB lifecycle, API host) with a Rust binary `frgated` while keeping the Python detector/video workers alive until the video pipeline is migrated. The Rust process uses the same IPC contracts (shared memory, ZMQ, MQTT, SQLite) so workers are interchangeable — migration is additive.

## 1. Project layout

```
frigate-rs/
├── Cargo.toml              # workspace: frgated binary + frgated-core
├── crates/
│   ├── frgated/            # binary: entrypoint, signal handling, supervisor
│   └── frgated-core/       # library: config, services, workers, IPC
└── docker/main/Dockerfile  # updated (section 6)
```

Workspace deps (selected):

| crate | role |
|---|---|
| `clap` | CLI (`--validate-config`, same args as Python) |
| `serde`/`serde_yaml` (serde_derive + ruamel equivalent) | YAML config — reuse `pydantic` validation via a one-shot Python subprocess call in Phase 1, full Rust Pydantic-style validation in Phase 2 |
| `tokio` + `tokio-util` | async runtime, worker process mgmt |
| `tokio::process` + `std::os::unix::process::CommandExt` | start/monitor subprocesses, reap on death |
| `sqlx` (sqlite, `features = ["offline"]`) | SQLite — replaces peewee; WAL, `auto_vacuum FULL`, sqlite-vec extension |
| `zmq` or `zeromq` | ZMQ proxy — drop-in replacement |
| `rumqttc` | MQTT (same broker) |
| `axum` + `tower` | API — same endpoints, same auth; share OpenAPI via `utoipa` |
| `tracing` + `tracing-subscriber` | logging — same stdout+file layout, timestamp prefix (s6 expects it) |
| `nix` | `prctl(PR_SET_PDEATHSIG)`, shared memory cleanup (`/dev/shm`) |
| `portable-atomic` | cross-compile (the Dockerfile is multiarch) |

## 2. Config — first shared module

`config/config.rs` parses `config.yml` → a single `FrigateConfig` struct. Same YAML keys as `FrigateConfig` (pydantic model). Two validation paths:

- Phase 1 (parallel): pass config body as JSON to a tiny Python helper (`python3 -m frigate.config.validate`) and re-parse errors — gives instant parity, no Rust validation rewrite.
- Phase 2: port every Pydantic model to Rust structs; drop the Python helper.

Config paths: `/config/config.yml`, DB at `/config/frigate.db`, `/dev/shm` artifacts — all same mount points, so config and volumes need zero changes.

## 3. Service registration (drop-in to `app.py`'s `start()`)

`app.py` starts ~15 services in order. Rust supervisor holds a table of `WorkerSpec { name, pid_ref, restart: OnDeath, ready_check }`:

| name | Python class | Rust task |
|---|---|---|
| `recording` | `RecordProcess` | `tokio::process::Child`, reap on exit |
| `review_segment` | `ReviewProcess` | same |
| `embeddings` | `EmbeddingProcess` | same |
| `output` | `OutputProcess` | same |
| `camera` | `CameraMaintainer` (×N cameras) | one task per enabled camera |
| `audio` | `AudioProcessor` | one |
| `detectors` | `ObjectDetectProcess` (×N detectors) | one per detector |
| `timeline` / `event_processor` / `event_cleanup` / `record_cleanup` / `storage_maintainer` | same | one each |
| `stats_emitter` / `watchdog` | same | background tasks |

Watchdog's restart semantics (`FrigateWatchdog.register(key, current, factory, on_restart)`) → supervisor maps each `WorkerSpec` to a `tokio::task::JoinHandle`; on `handle.await` Err, re-spawn from the factory closure. PID tracking in `self.processes: HashMap<String, u32>` → `workers[name].pid` — same shape, API consumers unchanged.

## 4. IPC — keep contracts, change plumbing

- **Multiprocessing Queue → `tokio` channel / `mimalloc`-backed ring buffer**: each Python `mp.Queue()` (detection, detected-frames, timeline) becomes a `tokio::sync::mpsc` or a shared `crossbeam` channel — same message types (`TrackedObject`, frame refs), same backpressure (`Queue(maxsize=...)`).
- **Shared memory**: `UntrackedSharedMemory(name)` → `memmap2` or POSIX shared memory (`/dev/shm/frigate-{name}`); cleanup loop on `SIGTERM` mirrors `stop()`'s `shm.unlink()`.
- **ZMQ proxy** (`ZmqProxy`) → `zmq` crate, same PUB/SUB/REQ topology.
- **Dispatcher**: `Dispatcher(comms: list[Communicator])` port directly — MQTT, WebSocket, WebPush, inter-process are independent tasks in Rust.
- **MQTT / WebSocket / WebPush**: same broker, same topics, same JSON schema — no wire change.

## 5. SQLite

- `SqliteVecQueueDatabase` → `sqlx::SqlitePool` with identical pragmas (`auto_vacuum FULL`, `cache_size` 512 MB, `synchronous NORMAL`).
- Models (`Event`, `Export`, `Recordings`, `ReviewSegment`, `Timeline`, `Trigger`, `User`) → sqlx query structs (port `models.py` one-for-one).
- Schema migrations: run `peewee_migrate` once at transition, then `sqlx migrate` thereafter — dual-run period is short (one deploy window).
- Vacuum logic (`check_db_data_migrations`, `.vacuum` file) → `sqlx migrate` replaces it.
- `sqlite-vec`: load the `.so` exactly as the Python side does (`load_vec_extension=True` → `ATTACH ... vec` in connection init).

## 6. API

- Axum tower stack: same route tree, same auth dependency (`api/auth`), same WS topic classifier (`comms/ws.py`'s per-recipient filter).
- Share the OpenAPI generation: `utoipa` → `generate_api_auth_spec.py` reads the Rust `utoipa` annotations. After the API migration, regenerate the spec (`python3 generate_api_auth_spec.py --check` passes).

## 7. Startup ordering (identical sequence)

`s6-rc.d/frigate/run` → `__main__.py` → `FrigateApp.start()`. Rust replica preserves the exact order — camera processes must be registered before `bind_database()`, ZMQ subscribers before `restore_active_profile()` (PUB/SUB drops before subscribers):

```
ensure_dirs() → set_file_limit() → init_database() → bind_database()
→ check_db_data_migrations() → init_ipc → start_detectors → init_dispatcher
→ init_profile_manager → restore_active_profile() → start workers → init_auth
→ axum (uvicorn equivalent)
```

Shutdown (`stop()`): same sequence — cancel motion search, set `SIGTERM` on workers, join order (audio→detector→frames→timeline→output→recording→review), DB `update(end_time)` for open events, shared-memory cleanup, dispatcher stop. Write `/dev/shm/.frigate-is-stopping` for the healthcheck.

## 8. Dockerfile changes

**`docker/main/Dockerfile`** — multi-stage Rust build before `COPY frigate`:

```dockerfile
# ---- Rust toolchain (only in build stage) ----
FROM --platform=$BUILDPLATFORM debian:12-slim AS rust-base
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /workspace

# ---- Build frgated ----
FROM rust-base AS frigate-build
COPY frigate-rs/ Cargo.toml Cargo.lock ./
RUN cargo build --release --target $(rustc -Vv | grep host | awk '{print $2}')

# ---- Final frigate stage (under `FROM deps AS frigate`) ----
# Add BEFORE `COPY frigate frigate/`:
COPY --from=frigate-build /workspace/target/release/frgated /usr/local/bin/frgated
# Keep the Python side for now (parallel run):
# COPY frigate frigate/          ← still here, workers still Python
# COPY migrations migrations/   ← still here
```

**`docker/main/rootfs/etc/s6-overlay/s6-rc.d/frigate/run`**:

```bash
# Old: exec python3 -u -m frigate
# New (Phase 1 — dual mode):
if [ "${FRIGATE_ENGINE:-rust}" = "rust" ]; then
    exec /usr/local/bin/frgated
else
    exec python3 -u -m frigate
fi
```

**Other Dockerfiles** (`rockchip`, `rocm`, `tensorrt`, `rpi`, `synaptics`): inherit `frgated` via the `FROM deps` base — no per-platform changes needed since the Rust binary is built cross-compile (target triple set at build time).

**`docker/main/rootfs/etc/s6-overlay/s6-rc.d/frigate-log/run`** and **`log-prepare`**: unchanged — `frgated` writes to stdout, same s6 log pipeline (timestamps, rotate 10 MB).

**Healthcheck**: unchanged — `curl ... /api/version` still hits the API (axum binds port 5001, same as uvicorn).

## 9. Migration phases (parallel run, flip switch)

Each phase is a self-contained, running system. Commit and push **after every phase** (the phases are designed to be safe alone — nothing here requires the next phase to work).

### Phase 1 — Config + supervisor

- Add `frigate-rs/` workspace: `Cargo.toml`, `crates/frgated/` (binary), `crates/frgated-core/` (library)
- `config/config.rs`: parse `config.yml` → `FrigateConfig` struct (Phase 1 validation delegates to Python helper `python3 -m frigate.config.validate`; drop it in Phase 2)
- `supervisor.rs`: `WorkerSpec` table, `WorkerManager::spawn(name, Command)`, PID tracking (`HashMap<String, u32>`)
- `signal.rs`: SIGINT/SIGTERM → set stop flag (same semantics as `mp.Event()` in `__main__.py`)
- `frgated --validate-config` mirrors `python3 -m frigate --validate-config`
- Dockerfile: add `rust-base` + `frigate-build` stages, copy binary into `FROM deps AS frigate`
- s6 run script: dual mode `FRIGATE_ENGINE=rust`

**Commit & push:** `git commit -m "rust: orchestrator skeleton — config parse, supervisor, Dockerfile build stages"` && `git push`

**Verify:** `cargo test`; `FRIGATE_ENGINE=rust ./frgated --validate-config` succeeds; s6 dual-mode start works.

### Phase 2 — Worker lifecycle + watchdog

- `WorkerManager` manages `tokio::process::Child`: start, reap on exit, restart-on-death from factory closure
- Watchdog: per-worker `JoinHandle` + `on_death` callback (exact same `self.processes` dict shape as `app.py`'s `FrigateWatchdog.register()`)
- Join-order shutdown: per-worker graceful-then-kill (terminate → kill timeout 10s, same as `service_manager/multiprocessing.py`'s `on_stop`)
- s6 run script switches to `FRIGATE_ENGINE=rust` (default) — Python side still functional

**Commit & push:** `git commit -m "rust: worker lifecycle — spawn, reap, watchdog restart, join-order shutdown"` && `git push`

**Verify:** same as Phase 1; kill a worker, confirm watchdog re-spawns and PID in `self.processes` updates.

### Phase 3 — IPC: queues/channels, shared memory, ZMQ

- `mp.Queue` → `tokio::sync::mpsc` or `crossbeam` channel (same message types, same backpressure)
- `UntrackedSharedMemory(name)` → `memmap2` / POSIX shared memory at `/dev/shm/frigate-{name}`; SIGQUIT/SIGABRT trap to unlink on crash (replaces Python's `stop()` loop)
- `SyncManager.dict()`/`Value`/`MpEvent` → `Arc<RwLock<...>>` + `tokio::sync::Notify` (replaces `camera_metrics` DictProxy, `stop_event` Event)
- ZMQ proxy via `zmq` crate; Dispatcher tasks (MQTT/WebSocket/WebPush) as independent tokio tasks
- Forkserver/preload note: `Command::spawn` has no preload list (irrelevant); `at_fork` handler in `log.py` not needed — documented in section 10b

**Commit & push:** `git commit -m "rust: IPC — channels, shared memory, ZMQ proxy, dispatcher tasks"` && `git push`

**Verify:** workers (still Python) and `frgated` exchange frames/metrics; kill the parent, confirm children cleaned up at `/dev/shm`.

### Phase 4 — API

- Axum tower stack: routes, auth dependency (`api/auth`), WS topic classifier (`comms/ws.py`)
- `utoipa` annotations → regenerate OpenAPI spec: `python3 generate_api_auth_spec.py` (then CI's `--check` variant passes)
- Same endpoints, same WS topic routing, same auth as uvicorn

**Commit & push:** `git commit -m "rust: API layer — axum routes, auth, WebSocket topic classifier"` && `git push`

**Verify:** `python3 generate_api_auth_spec.py --check` passes; API responses identical to uvicorn.

### Phase 5 — SQLite

- `sqlx::SqlitePool` with identical pragmas (`auto_vacuum FULL`, `cache_size` 512MB, `synchronous NORMAL`)
- Models ported from `models.py` → sqlx query structs (one-for-one)
- Run `peewee_migrate` once, then `sqlx migrate` — dual-run DB migration in one deploy window
- `sqlite-vec` ATTACH in connection init (same as `load_vec_extension=True`)

**Commit & push:** `git commit -m "rust: SQLite — sqlx pool, model port, migration, sqlite-vec"` && `git push`

**Verify:** `frgated` reads/writes the live `/config/frigate.db` without corruption; vacuum runs.

### Phase 6 — Video pipeline (separate project)

- `video/detect.py`, `video/ffmpeg.py`, `object_detection/`, `track/`, `embeddings/` ported
- Per-camera tokio tasks (one ffmpeg demuxer → frame buffer pipeline per camera)
- ONNX Runtime via `ort` crate; shared_memory→zero-copy
- Norfair tracker port

**Commit & push:** `git commit -m "rust: video pipeline — camera processes, detection, tracking"` && `git push`

### Phase 7 — Full switch + retirement

- `FRIGATE_ENGINE=rust` is the only mode (dual-mode block removed from s6 run)
- Python `frigate/` still mounted for workers until Phase 6 done; retire once verified
- Retire peewee/Python-worker deps from Dockerfile (keep ffmpeg/opencv/ONNX runtimes — they're still used by the Rust `ort` FFI)

**Commit & push:** `git commit -m "rust: production — default to Rust, retire Python orchestrator and peewee"` && `git push`

## 10. Risks & mitigations

- **ONNX Runtime** (used by `ObjectDetectProcess` workers) — not in scope for the orchestrator; `ort` crate exists if/when detectors migrate.
- **Shared-memory cleanup on crash** — `frgated` installs `SIGQUIT`/`SIGABRT` trap to unlink `/dev/shm/frigate-*`; forkserver preload list stays in the Python side (Rust's `std::sync::Mutex` doesn't need it).
- **Profile switching** (`ProfileManager.activate_profile`, `restore_runtime_state`) — port the runtime override mechanism first (it's config + broadcast, cheap).
- **Home Assistant Add-on config migration** (`migrate_addon_config_dir`, `migrate_db_from_media_to_config`) — stays in `prepare/run`; `frgated` reads `/config/config.yml` only.
- **Forkserver preload** (`mp.set_start_method("forkserver")`, preload list) — Rust uses `tokio::process::Command` spawn, no fork; preload list is irrelevant once Rust owns the parent.

## 10b. Forkserver and preload list (critical detail)

`frigate/__main__.py` sets `mp.set_start_method("forkserver", force=True)` and a preload list (`sqlite3, numpy, cv2, peewee, zmq, ruamel.yaml, frigate.camera.maintainer`). The parent process starts big (loads numpy/cv2/peewee/zmq), then every `mp.Process` fork inherits that state via copy-on-write — preload modules arrive free in the child, avoiding re-import cost and fork-time import deadlocks. Workers are created via `service_manager/multiprocessing.py:BaseServiceProcess._process = mp.Process(target=self._run)`.

The logging system depends on this: `setup_logging()` creates a `SyncManager` + `QueueListener` thread; `before_start()` passes the log queue to each child, and `ServiceProcess.before_run()` installs `QueueHandler(log_queue)` on top of `basicConfig`.

**Rust equivalent:**

- `forkserver` → `tokio::process::Command::spawn()`. Spawn (Rust) is the closest analogue to forkserver (not fork — forkserver avoids fork's lock-hazards). Rust `Command::spawn` inherits file descriptors and `LD_LIBRARY_PATH` automatically, no preload list needed (forkserver's preload list is a pure-Python startup optimization; Rust has no equivalent since modules aren't imported at spawn).
- `os.register_at_fork(after_in_child=reopen_std_streams)` in `log.py` — workaround for a Python bug where forked children deadlock flushing stdout when a thread holds the lock. Rust `Command::spawn` doesn't fork, so no at_fork handler needed. The child inherits FD 1/2 and writes directly to them (same as forkserver children).
- `SyncManager.dict()` / `Value` / `mp.Queue` → `tokio::sync` channels + shared memory. The `SyncManager`-backed `camera_metrics: DictProxy`, `ValueProxy` counters, and `MpEvent` stop-events become `Arc<RwLock<...>>` or `crossbeam` channels — same semantics, no manager process.
- Worker log queue → `tracing` subscriber on FD 1, or `tracing`'s `os_pipe` → `tokio` channel for per-worker log streams.
- Preload list → irrelevant. Rust workers start from scratch; if any module needs to be pre-loaded (e.g., ONNX runtime shared library), it's done explicitly in the worker's init function, not inherited.

**Migration implication:** the preload list is lost in the rewrite, so Rust workers that import numpy/cv2 (while still Python) via `Command::spawn` inherit nothing. Two options:
1. **Phase 1:** keep workers as a Python subprocess (same as now) — `Command::spawn("python3", ["-m", "frigate.camera.maintainer"])` — forkserver preload is irrelevant since we use spawn, and the Python subprocess gets a fresh GIL + imports.
2. **Phase 2:** once workers are Rust, no preload needed.

## 11. Verification

- `python3 -u -m frigate` and `frgated` run side-by-side against the same config + DB → same API responses, same events.
- `cargo test` for config parse + worker lifecycle; `python3 -u -m unittest` still passes (workers untouched).
- Stress: kill random workers, confirm watchdog re-spawns and API reports new PIDs (same as `self.processes` dict semantics).
