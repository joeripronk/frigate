use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post, put, delete};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use clap::Parser;
use tokio::sync::Notify;
use tracing_subscriber::{
    fmt::Layer,
    prelude::*,
    EnvFilter,
};

use frgated_core::config::FrigateConfig;
use frgated_core::supervisor::{SignalSender, WorkerManager};
use frgated_core::api::app;
use frgated_core::api::auth_routes;

/// Frigate Rust orchestrator — drop-in for `python3 -m frigate`.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[arg(long)]
    validate_config: bool,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        app::is_healthy,
        app::config_schema,
        app::version,
        app::stats,
        app::stats_history,
        app::metrics,
        app::genai_models,
        app::genai_probe,
        app::config,
        app::get_profiles,
        app::get_active_profile,
        app::ffmpeg_presets,
        app::config_raw_paths,
        app::config_raw,
        app::config_save,
        app::config_set,
        app::vainfo,
        app::nvinfo,
        app::logs,
        app::restart,
        app::sync_media,
        app::get_media_sync_current,
        app::get_media_sync_status,
        app::get_labels,
        app::get_sub_labels,
        app::get_audio_labels,
        app::plus_models,
        app::timeline,
        auth_routes::first_time_login,
        auth_routes::auth,
        auth_routes::profile,
        auth_routes::logout,
        auth_routes::login,
        auth_routes::get_users,
        auth_routes::create_user,
        auth_routes::delete_user,
        auth_routes::update_password,
        auth_routes::update_role,
    ),
    tags(
        (name = "App", description = "App endpoints"),
        (name = "Auth", description = "Authentication endpoints"),
        (name = "Camera", description = "Camera endpoints"),
        (name = "Chat", description = "Chat endpoints"),
        (name = "Events", description = "Event endpoints"),
        (name = "Export", description = "Export endpoints"),
        (name = "Classification", description = "Classification endpoints"),
        (name = "Logs", description = "Log endpoints"),
        (name = "Media", description = "Media endpoints"),
        (name = "Motion Search", description = "Motion search endpoints"),
        (name = "Notifications", description = "Notification endpoints"),
        (name = "Preview", description = "Preview endpoints"),
        (name = "Recordings", description = "Recording endpoints"),
        (name = "Review", description = "Review endpoints"),
    ),
)]
struct ApiDoc;

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

    register_workers(&mut mgr, &sig);

    let stop = Arc::new(Notify::new());
    let stop_clone = stop.clone();
    tokio::spawn(async move {
        sig.wait().await;
        stop_clone.notify_one();
    });

    let stop_watchdog = stop.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = stop_watchdog.notified() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
        }
    });

    tracing::info!("All workers registered — ready");

    // Build Axum API router — mirrors `create_fastapi_app()` wiring.
    let api = build_api(cfg.clone());
    let api_clone = api.clone();
    let stop_api = stop.clone();
    let api_handle = tokio::spawn(async move {
        axum::serve(
            tokio::net::TcpListener::bind("127.0.0.1:5001").await.unwrap(),
            api_clone,
        )
        .with_graceful_shutdown(async move {
            stop_api.notified().await;
        })
        .await
    });

    // Block on shutdown signal, then join-order shutdown.
    stop.notified().await;
    tracing::info!("Shutting down workers");

    mgr.shutdown(
        &[
            "audio", "detector", "frames", "timeline",
            "output", "recording", "review_segment",
        ],
        Duration::from_secs(10),
    ).await;

    mgr.join_all().await;
    let _ = api_handle.await;
    tracing::info!("Shutdown complete");

    Ok(())
}

fn build_api(cfg: FrigateConfig) -> Router {
    let state = app::AppState { config: cfg };

    Router::new()
        // App endpoints
        .route("/", get(app::is_healthy))
        .route("/version", get(app::version))
        .route("/stats", get(app::stats))
        .route("/stats/history", get(app::stats_history))
        .route("/metrics", get(app::metrics))
        .route("/genai/models", get(app::genai_models))
        .route("/genai/probe", post(app::genai_probe))
        .route("/config", get(app::config))
        .route("/config/schema.json", get(app::config_schema))
        .route("/profiles", get(app::get_profiles))
        .route("/profile/active", get(app::get_active_profile))
        .route("/ffmpeg/presets", get(app::ffmpeg_presets))
        .route("/config/raw_paths", get(app::config_raw_paths))
        .route("/config/raw", get(app::config_raw))
        .route("/config/save", post(app::config_save))
        .route("/config/set", put(app::config_set))
        .route("/vainfo", get(app::vainfo))
        .route("/nvinfo", get(app::nvinfo))
        .route("/logs/{service}", get(app::logs))
        .route("/restart", post(app::restart))
        .route("/media/sync", post(app::sync_media))
        .route("/media/sync/current", get(app::get_media_sync_current))
        .route("/media/sync/status/{job_id}", get(app::get_media_sync_status))
        .route("/labels", get(app::get_labels))
        .route("/sub_labels", get(app::get_sub_labels))
        .route("/audio_labels", get(app::get_audio_labels))
        .route("/plus/models", get(app::plus_models))
        .route("/timeline", get(app::timeline))
        // Auth endpoints
        .route("/auth/first_time_login", get(auth_routes::first_time_login))
        .route("/auth", get(auth_routes::auth))
        .route("/profile", get(auth_routes::profile))
        .route("/logout", get(auth_routes::logout))
        .route("/login", post(auth_routes::login))
        .route("/users", get(auth_routes::get_users).post(auth_routes::create_user))
        .route("/users/{username}", delete(auth_routes::delete_user))
        .route("/users/{username}/password", put(auth_routes::update_password))
        .route("/users/{username}/role", put(auth_routes::update_role))
        // Swagger UI at /swagger-ui
        .merge(
            SwaggerUi::new("/swagger-ui")
                .url("/api/openapi.json", ApiDoc::openapi()),
        )
        .with_state(state.config)
        .layer(TraceLayer::new_for_http())
}

fn register_workers(mgr: &mut WorkerManager, _sig: &SignalSender) {
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
