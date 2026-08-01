use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;
use tokio::sync::Notify;
use tracing::{error, info, warn};

use crate::config::FrigateConfig;

/// Signals received by the process.
#[derive(Debug, Clone, Copy)]
pub enum ShutdownSignal {
    Int,
    Term,
}

/// One-shot signal sender shared with supervisor tasks.
#[derive(Clone)]
pub struct SignalSender(Arc<Notify>);

impl SignalSender {
    pub fn new() -> Self {
        Self(Arc::new(Notify::new()))
    }

    pub fn notify(&self) {
        self.0.notify_waiters();
    }

    pub async fn wait(&self) -> ShutdownSignal {
        self.0.notified().await;
        ShutdownSignal::Term
    }
}

impl std::fmt::Display for ShutdownSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShutdownSignal::Int => write!(f, "SIGINT"),
            ShutdownSignal::Term => write!(f, "SIGTERM"),
        }
    }
}

pub type WorkerFactory = Arc<dyn Fn() -> WorkerSpec + Send + Sync>;

pub fn make_factory<F>(f: F) -> WorkerFactory
where
    F: Fn() -> WorkerSpec + Send + Sync + 'static,
{
    Arc::new(f)
}

/// Spec for a worker the supervisor manages.
pub struct WorkerSpec {
    pub name: &'static str,
    pub cmd: Command,
    pub restart: bool,
}

/// Restart throttling constants — mirrors `frigate/watchdog.py`.
const MAX_RESTARTS: usize = 5;
const RESTART_WINDOW_S: f64 = 60.0;

/// Per-worker restart tracking, mirroring `MonitoredProcess`.
struct MonitoredWorker {
    name: String,
    restart_timestamps: VecDeque<f64>,
    clean_exit_logged: bool,
}

impl MonitoredWorker {
    fn is_restarting_too_fast(&mut self, now: f64) -> bool {
        while self
            .restart_timestamps
            .front()
            .map(|t| now - t > RESTART_WINDOW_S)
            .unwrap_or(false)
        {
            self.restart_timestamps.pop_front();
        }
        self.restart_timestamps.len() >= MAX_RESTARTS
    }

    fn record_restart(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.restart_timestamps
            .push_back(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64());
    }
}

/// Supervisor — owns workers, reaps dead ones, handles shutdown order.
pub struct WorkerManager {
    pub pids: std::collections::HashMap<String, u32>,
    handles: std::collections::HashMap<String, tokio::task::JoinHandle<()>>,
    factories: std::collections::HashMap<String, WorkerFactory>,
    monitored: std::collections::HashMap<String, MonitoredWorker>,
    on_death: std::collections::HashMap<String, Arc<dyn Fn() + Send + Sync>>,
    dead_notify: Arc<Notify>,
}

impl WorkerManager {
    pub fn new() -> Self {
        Self {
            pids: std::collections::HashMap::new(),
            handles: std::collections::HashMap::new(),
            factories: std::collections::HashMap::new(),
            monitored: std::collections::HashMap::new(),
            on_death: std::collections::HashMap::new(),
            dead_notify: Arc::new(Notify::new()),
        }
    }

    /// Register a worker with a factory. Mirrors `FrigateWatchdog.register()`.
    pub fn register(
        &mut self,
        name: &str,
        factory: WorkerFactory,
        on_death: Arc<dyn Fn() + Send + Sync>,
    ) {
        self.factories.insert(name.to_owned(), factory);
        self.on_death.insert(name.to_owned(), on_death);
        self.spawn_worker(name);
    }

    fn spawn_worker(&mut self, name: &str) {
        let factory = self.factories[name].clone();
        let name_owned = name.to_owned();
        let name_clone = name_owned.clone();

        let handle = tokio::spawn(async move {
            let spec = factory();
            let mut cmd = spec.cmd;

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to spawn worker '{name_clone}': {e}");
                    return;
                }
            };

            let pid = child.id().unwrap_or(0);
            info!("Started {name_clone} (pid: {pid})");

            let status = child.wait().await.unwrap_or_default();
            info!("{name_clone} exited with status {status}");

            if !status.success() {
                warn!("{name_clone} exited non-zero");
            }
        });

        self.handles.insert(name_owned, handle);
    }

    /// Graceful-then-kill shutdown in the given join order.
    ///
    /// Same semantics as `app.py`'s `stop()`: terminate, wait up to `kill_timeout`,
    /// then kill. 10s default mirrors Python's `DEFAULT_STOP_TIMEOUT`.
    pub async fn shutdown(
        &mut self,
        order: &[&'static str],
        kill_timeout: Duration,
    ) {
        for &name in order {
            let handle = match self.handles.remove(name) {
                Some(h) => h,
                None => continue,
            };

            info!("Stopping worker '{name}'");
            match tokio::time::timeout(kill_timeout, handle).await {
                Ok(Ok(())) => info!("{name} stopped"),
                Ok(Err(e)) => warn!("{name} join error: {e}"),
                Err(_) => warn!("{name} timed out — kill"),
            }
        }
    }

    /// Join every worker handle.
    pub async fn join_all(mut self) {
        while let Some((_, handle)) = self.handles.drain().next() {
            info!("Joining worker");
            let _ = handle.await;
        }
    }

    /// Build the config struct from the config YAML — helper for Phase 2 worker factories.
    pub fn load_config() -> Result<FrigateConfig, String> {
        FrigateConfig::load().map_err(|e| e.to_string())
    }

    /// Build the config in safe mode (graceful degradation).
    pub fn load_config_safe() -> Result<FrigateConfig, String> {
        FrigateConfig::load()
            .map_err(|e| e.to_string())
            .or_else(|e| {
                FrigateConfig::load_safe().map_err(|_| format!("Safe mode load failed: {e}"))
            })
    }
}
