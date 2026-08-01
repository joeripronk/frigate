use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{any, get, post, put, delete};
use axum::extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State};
use axum::response::Response;
use futures::{SinkExt, StreamExt};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use clap::Parser;
use tokio::process::Command;
use tokio::sync::{broadcast, Notify};
use tracing_subscriber::{
    fmt::Layer,
    prelude::*,
    EnvFilter,
};

use frgated_core::config::FrigateConfig;
use frgated_core::supervisor::{SignalSender, WorkerManager, make_factory, WorkerSpec};
use frgated_core::api::app;
use frgated_core::api::auth_routes;
use frgated_core::api::ws as ws_auth;
use frgated_ipc::Subscriber;

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

    // WebSocket — also serve /ws on port 5002 (mirrors Python comms/ws.py).
    let ws_cfg = cfg.clone();
    let stop_ws = stop.clone();
    let ws_handle = tokio::spawn(async move {
        let ws_router = Router::new()
            .route("/", any(ws_handler))
            .with_state(ws_cfg)
            .layer(TraceLayer::new_for_http());
        axum::serve(
            tokio::net::TcpListener::bind("127.0.0.1:5002").await.unwrap(),
            ws_router,
        )
        .with_graceful_shutdown(async move {
            stop_ws.notified().await;
        })
        .await
    });

    // Block on shutdown signal, then join-order shutdown.
    stop.notified().await;
    tracing::info!("Shutting down workers");

    // Phase 1: only recording worker is registered; others run in main process.
    mgr.shutdown(
        &["recording"],
        Duration::from_secs(10),
    ).await;

    mgr.join_all().await;
    let _ = api_handle.await;
    let _ = ws_handle.await;
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

fn register_workers(mgr: &mut WorkerManager, sig: &SignalSender) {
    let config_path = frgated_core::config::config_path();

    // Worker factories — each spawns a Python subprocess.
    // Per rust.md section 10b: Command::spawn starts a fresh Python process
    // (no forkserver preload), so each worker imports everything from scratch.
    // This is acceptable for Phase 1-5; the performance-critical video
    // pipeline (Phase 6) will be in Rust.
    //
    // NOTE: Only workers with standalone modules (frigate.worker.*) are
    // registered here. Other workers (embeddings, audio, output, review)
    // still run in the main Python process and will be migrated later.

    let make_python_worker = |module: &'static str, name: &'static str| {
        let config_path = config_path.clone();
        make_factory(move || {
            WorkerSpec {
                name,
                cmd: {
                    let mut cmd = Command::new("python3");
                    cmd.arg("-m")
                       .arg(format!("frigate.worker.{}", module))
                       .arg("--config")
                       .arg(&config_path)
                       .env("FRIGATE_WORKER", module);
                    cmd
                },
                restart: true,
            }
        })
    };

    // Core workers that the supervisor manages (watchdog restart on death).
    // Phase 1: only recording has a standalone worker module.
    mgr.register(
        "recording",
        make_python_worker("record", "recording"),
        Arc::new(move || {
            tracing::info!("Recording worker died — restarting");
        }),
    );

    // Camera maintainers — one per enabled camera.
    // These are registered dynamically based on config.
    // For now, register a placeholder that will be filled in once config
    // is fully loaded and cameras are enumerated.
    let _sig = sig;
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

// ── WebSocket handler ────────────────────────────────────────────────

/// WebSocket upgrade handler — mirrors `comms/ws.py` (port 5002, proxied by nginx /ws).
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(config): State<FrigateConfig>,
    headers: axum::http::HeaderMap,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws(socket, config, headers))
}

async fn handle_ws(mut socket: WebSocket, config: FrigateConfig, headers: axum::http::HeaderMap) {
    // Separator — mirrors config.proxy.separator (not yet in Rust config, hardcoded default).
    let separator = ",";
    let role_header = headers
        .get("Remote-Role")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned());
    let has_role = role_header.is_some();
    let camera_names: std::collections::HashSet<String> = config.cameras.keys().cloned().collect();
    // Zones — not yet in Rust CameraConfig; use empty set (zone filtering will drop these topics).
    let all_zones: std::collections::HashSet<String> = std::collections::HashSet::new();

    let hub = Hub::new("frigate").await;
    let hub = Arc::new(tokio::sync::Mutex::new(hub));

    // Subscriber task: ZMQ pub-sub → broadcast channel.
    let hub_b = hub.clone();
    let broadcast_tx = hub.lock().await.sender.clone();
    let broadcast_tx_sub = broadcast_tx.clone();
    let camera_names_sub = camera_names.clone();
    let all_zones_sub = all_zones.clone();
    tokio::spawn(async move {
        loop {
            let mut hub = hub_b.lock().await;
            let (sub_topic, payload) = hub.subscriber.check_for_update();
            drop(hub);
            if let Some(payload) = payload {
                let scope = ws_auth::classify_outbound(&sub_topic, &camera_names_sub, &all_zones_sub);
                if scope != ws_auth::OutboundScope::Drop {
                    let msg = ws_auth::wrap_envelope(&sub_topic, &payload);
                    let _ = broadcast_tx_sub.send(msg);
                }
            }
            tokio::time::sleep(std::time::Duration::from_micros(10)).await;
        }
    });

    // Fan-out + bidirectional loop via select!: broadcast → socket (with per-recipient filtering), socket → auth.
    let broadcast_tx_fanout = broadcast_tx.clone();
    let mut rx = broadcast_tx_fanout.subscribe();
    let config_for_filter = config.clone();
    loop {
        tokio::select! {
            biased;

            msg = rx.recv() => {
                match msg {
                    Ok(text) => {
                        // Per-recipient filtering — mirrors _materialize_for_ws() in comms/ws.py.
                        let scope = {
                            let topic = text.as_str();
                            // Extract topic from envelope to classify
                            if let Ok(envelope) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(topic_val) = envelope.get("topic").and_then(|t| t.as_str()) {
                                    ws_auth::classify_outbound(topic_val, &camera_names, &all_zones)
                                } else {
                                    ws_auth::OutboundScope::Drop
                                }
                            } else {
                                ws_auth::OutboundScope::Drop
                            }
                        };
                        if let Some(filtered) = ws_auth::materialize_for_ws(
                            "",
                            &text,
                            &scope,
                            &config_for_filter,
                            has_role,
                        ) {
                            if socket.send(Message::Text(filtered.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::debug!("WebSocket client lagged by {n} messages");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }

            msg = socket.next() => {
                let frame = match msg {
                    Some(Ok(m)) => m,
                    _ => break,
                };
                if let Message::Text(text) = frame.clone() {
                    if let Ok(envelope) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(topic) = envelope.get("topic").and_then(|t| t.as_str()) {
                            if let Err(e) = ws_auth::check_ws_authorization(
                                topic,
                                role_header.as_deref(),
                                separator,
                                &config.auth.roles,
                                &camera_names,
                            ) {
                                tracing::warn!(
                                    "Blocked unauthorized WebSocket message: topic={topic}, role={role_header:?}, reason={e}"
                                );
                                continue;
                            }
                        }
                    }
                    let _ = socket.send(frame);
                }
            }
        }
    }
}

struct Hub {
    subscriber: Subscriber,
    sender: broadcast::Sender<String>,
}

impl Hub {
    async fn new(topic: &str) -> Self {
        let subscriber = Subscriber::new(topic);
        let (sender, _) = broadcast::channel::<String>(4096);
        Self { subscriber, sender }
    }
}
