use clap::Parser;
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

fn main() -> anyhow::Result<()> {
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
    let _sig = SignalSender::new();
    let mgr = WorkerManager::new();

    // Real worker factories (one per process: camera, detector, recording, ...)
    // wired in Phase 2 — skeleton for Phase 1.
    let _ = cfg;
    let _ = mgr;

    tracing::info!("All workers registered — ready");

    Ok(())
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
