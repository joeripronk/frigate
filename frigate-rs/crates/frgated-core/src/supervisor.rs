use std::collections::HashMap;
use std::sync::Arc;

use tokio::process::Command;
use tokio::sync::Notify;
use tracing::{error, info, warn};

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

pub type WorkerFactory = Box<dyn Fn() -> WorkerSpec + Send + Sync>;

/// Spec for a worker the supervisor manages.
pub struct WorkerSpec {
    pub name: &'static str,
    pub cmd: Command,
    pub restart: bool,
}

/// Supervisor — owns workers, reaps dead ones, handles shutdown order.
pub struct WorkerManager {
    pub pids: HashMap<String, u32>,
    handles: HashMap<String, tokio::task::JoinHandle<()>>,
    dead_notify: Arc<Notify>,
}

impl WorkerManager {
    pub fn new() -> Self {
        Self {
            pids: HashMap::new(),
            handles: HashMap::new(),
            dead_notify: Arc::new(Notify::new()),
        }
    }

    pub fn register(&mut self, name: &'static str, factory: WorkerFactory) {
        self.spawn_worker(name, factory);
    }

    fn spawn_worker(&mut self, name: &'static str, factory: WorkerFactory) {
        let name_owned = name.to_owned();
        let handle = tokio::spawn(async move {
            let spec = factory();
            let mut cmd = spec.cmd;

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to spawn worker '{name}': {e}");
                    return;
                }
            };

            info!("Started {name} (pid: {})", child.id().unwrap_or(0));
            let status = child.wait().await.unwrap_or_default();
            info!("{name} exited with status {status}");

            if !status.success() {
                warn!("{name} exited non-zero");
            }
        });

        self.handles.insert(name_owned, handle);
    }

    pub async fn shutdown(
        &mut self,
        order: &[&'static str],
        kill_timeout: std::time::Duration,
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

    pub async fn join_all(mut self) {
        while let Some((_, handle)) = self.handles.drain().next() {
            info!("Joining worker");
            let _ = handle.await;
        }
    }
}
