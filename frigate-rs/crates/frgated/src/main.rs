use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use tokio::sync::Notify;
use tracing_subscriber::{
    fmt::Layer,
    prelude::*,
    EnvFilter,
};

use frgated_core::config::FrigateConfig;
use frgated_core::supervisor::{SignalSender, WorkerManager};

/// Frigate Rust orchestrator — drop-in for `python3 -m frigate`.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[arg(long)]
    validate_config: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::new("info,frgated=debug,frgated_core=debug")
            }),
        )
        .with(Layer::default().with_writer(std::io::stderr))
        .init();

    if cli.validate_config {
        return validate_config_only();
    }

    let cfg = load_config()?;
    let sig = SignalSender::new();
    let mut mgr = WorkerManager::new();

    // Real worker factories — wired in Phase 2.
    register_workers(&mut mgr, &sig);

    let _ = cfg;

    let stop = Arc::new(Notify::new());
    let stop_clone = stop.clone();
    tokio::spawn(async move {
        sig.wait().await;
        stop_clone.notify_one();
    });

    // Watchdog-style restart loop — mirrors Python watchdog's 10s poll.
    let stop_watchdog = stop.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = stop_watchdog.notified() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {
                    // Periodic watchdog check — in Phase 2 this reaps dead workers
                    // and restarts them (throttled to MAX_RESTARTS/5 in RESTART_WINDOW_S/60s).
                }
            }
        }
    });

    tracing::info!("All workers registered — ready");

    // Block on shutdown signal, then join-order shutdown.
    stop.notified().await;
    tracing::info!("Shutting down workers");

    // Join order mirrors app.py stop(): audio→detector→frames→timeline→output→recording→review
    mgr.shutdown(
        &[
            "audio",
            "detector",
            "frames",
            "timeline",
            "output",
            "recording",
            "review_segment",
        ],
        Duration::from_secs(10),
    )
    .await;

    mgr.join_all().await;
    tracing::info!("Shutdown complete");

    Ok(())
}

fn register_workers(mgr: &mut WorkerManager, _sig: &SignalSender) {
    // One factory per worker type; each closure builds a WorkerSpec with Command.
    // Example skeleton (Phase 2 wiring):
    // mgr.register(
    //     "recording",
    //     make_factory(|| {
    //         WorkerSpec {
    //             name: "recording",
    //             cmd: Command::new("python3")
    //                 .args(["-m", "frigate.record.record"])
    //                 .kill_on_drop(true),
    //             restart: true,
    //         }
    //     }),
    //     Arc::new(|| { /* on_death: update self.processes["recording"] = 0 */ }),
    // );

    let _ = mgr;
    let _ = _sig;
}

fn validate_config_only() -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(frgated_core::config::config_path())?;
    match FrigateConfig::validate(&raw) {
        Ok(()) => {
            println!("Your config file is valid.");
            Ok(())
        }
        Err(msgs) => {
            for m in msgs {
                eprintln!("{m}");
            }
            anyhow::bail!("Config validation failed");
        }
    }
}

fn load_config() -> anyhow::Result<FrigateConfig> {
    FrigateConfig::load().or_else(|e| {
        eprintln!("Failed to load config: {e}");
        FrigateConfig::load_safe().map(|cfg| {
            println!("Starting Frigate in safe mode.");
            cfg
        })
    })
}
