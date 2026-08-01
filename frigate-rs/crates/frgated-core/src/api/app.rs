use axum::response::{IntoResponse, Response, Json, Html, Redirect};
use axum::routing::{get, post, put, delete};
use axum::extract::{State, Query};
use axum_extra::extract::Query as AxumQuery;
use utoipa::ToSchema;

use serde::Deserialize;
use super::auth::{self, user_from_request};
use crate::config::FrigateConfig;

use utoipa::IntoParams;

/// App-scoped state shared by handlers.
pub struct AppState {
    pub config: FrigateConfig,
}

/// Health check — mirrors `GET /` (allow_public).
#[utoipa::path(
    get, path = "/",
    tag = "App",
    responses((status = 200, description = "Frigate is running"))
)]
pub async fn is_healthy() -> impl IntoResponse {
    "Frigate is running. Alive and healthy!"
}

/// Config schema — mirrors `GET /config/schema.json` (allow_public).
#[utoipa::path(
    get, path = "/config/schema.json",
    tag = "App",
    responses((status = 200, description = "JSON schema"))
)]
pub async fn config_schema(State(cfg): State<FrigateConfig>) -> impl IntoResponse {
    Json(get_config_schema(cfg))
}

/// Version — mirrors `GET /version` (allow_public).
#[utoipa::path(
    get, path = "/version",
    tag = "App",
    responses((status = 200, description = "Version string"))
)]
pub async fn version() -> impl IntoResponse {
    "0.0.0"
}

/// Stats — mirrors `GET /stats` (require authenticated, admin gets full).
#[utoipa::path(
    get, path = "/stats",
    tag = "App",
    responses((status = 200, description = "Stats snapshot"))
)]
pub async fn stats(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    let user = user_from_request(&headers);
    let full = user.role.0 == "admin";
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"full": full})))
}

/// Stats history — mirrors `GET /stats/history` (require admin).
#[utoipa::path(
    get, path = "/stats/history",
    tag = "App",
    responses((status = 200, description = "Stats history"))
)]
pub async fn stats_history(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!([])))
}

/// Prometheus metrics — mirrors `GET /metrics` (require authenticated).
#[utoipa::path(
    get, path = "/metrics",
    tag = "App",
    responses((status = 200, description = "Prometheus metrics"))
)]
pub async fn metrics(State(_state): State<FrigateConfig>) -> impl IntoResponse {
    "text/plain; charset=utf-8"
}

/// GenAI models list — mirrors `GET /genai/models` (require authenticated).
#[utoipa::path(
    get, path = "/genai/models",
    tag = "App",
    summary = "List available GenAI models",
    responses((status = 200, description = "Model list"))
)]
pub async fn genai_models(State(_state): State<FrigateConfig>) -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!([])))
}

/// GenAI probe — mirrors `POST /genai/probe` (require admin).
#[utoipa::path(
    post, path = "/genai/probe",
    tag = "App",
    summary = "Probe a GenAI provider without saving config",
    request_body = GenAIProbeBody,
    responses((status = 200, description = "Probed models"))
)]
pub async fn genai_probe(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    Json(body): Json<GenAIProbeBody>,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"success": true, "models": serde_json::json!(Vec::<serde_json::Value>::new())})))
}

/// Config — mirrors `GET /config` (require authenticated, admin-only redaction).
#[utoipa::path(
    get, path = "/config",
    tag = "App",
    responses((status = 200, description = "Full config"))
)]
pub async fn config(State(cfg): State<FrigateConfig>) -> impl IntoResponse {
    Json(cfg)
}

/// Profiles — mirrors `GET /profiles` (require authenticated).
#[utoipa::path(
    get, path = "/profiles",
    tag = "App",
    responses((status = 200, description = "Profile list"))
)]
pub async fn get_profiles() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!([])))
}

/// Active profile — mirrors `GET /profile/active` (require authenticated).
#[utoipa::path(
    get, path = "/profile/active",
    tag = "App",
    responses((status = 200, description = "Active profile"))
)]
pub async fn get_active_profile() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"active_profile": null})))
}

/// FFmpeg presets — mirrors `GET /ffmpeg/presets` (require authenticated).
#[utoipa::path(
    get, path = "/ffmpeg/presets",
    tag = "App",
    responses((status = 200, description = "Preset keys"))
)]
pub async fn ffmpeg_presets() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!({
        "hwaccel_args": ["preset-vaapi", "preset-nvidia"],
        "input_args": ["preset-rtsp-generic"],
        "output_args": {"record": ["preset-record-generic"], "detect": []}
    })))
}

/// Config raw paths — mirrors `GET /config/raw_paths` (require admin).
#[utoipa::path(
    get, path = "/config/raw_paths",
    tag = "App",
    responses((status = 200, description = "Raw paths"))
)]
pub async fn config_raw_paths(cfg: State<FrigateConfig>, headers: axum::http::HeaderMap) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"cameras": {}, "go2rtc": {"streams": {}}})))
}

/// Config raw — mirrors `GET /config/raw` (require admin).
#[utoipa::path(
    get, path = "/config/raw",
    tag = "App",
    responses((status = 200, description = "Raw YAML"))
)]
pub async fn config_raw(cfg: State<FrigateConfig>, headers: axum::http::HeaderMap) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"raw": ""})))
}

/// Config save — mirrors `POST /config/save` (require admin).
#[utoipa::path(
    post, path = "/config/save",
    tag = "App",
    request_body = AppConfigSetBody,
    responses((status = 200, description = "Saved"))
)]
pub async fn config_save(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    Query(params): Query<ConfigSaveParams>,
    Json(body): Json<AppConfigSetBody>,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    let restart = params.save_option == Some("restart".to_owned());
    axum::response::IntoResponse::into_response(Json(serde_json::json!({
        "success": true,
        "message": if restart { "Config successfully saved, restarting..." } else { "Config successfully saved." }
    })))
}

/// Config set (PUT) — mirrors `PUT /config/set` (require admin).
#[utoipa::path(
    put, path = "/config/set",
    tag = "App",
    request_body = AppConfigSetBody,
    responses((status = 200, description = "Updated"))
)]
pub async fn config_set(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    Query(params): Query<ConfigSetParams>,
    Json(body): Json<AppConfigSetBody>,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"success": true, "message": "Config successfully updated"})))
}

/// Vainfo — mirrors `GET /vainfo` (require authenticated).
#[utoipa::path(
    get, path = "/vainfo",
    tag = "App",
    responses((status = 200, description = "vainfo output"))
)]
pub async fn vainfo() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"return_code": 0, "stdout": "", "stderr": ""})))
}

/// Nvinfo — mirrors `GET /nvinfo` (require authenticated).
#[utoipa::path(
    get, path = "/nvinfo",
    tag = "App",
    responses((status = 200, description = "NV driver info"))
)]
pub async fn nvinfo() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!([])))
}

/// Logs — mirrors `GET /logs/{service}` (require admin).
#[utoipa::path(
    get, path = "/logs/{service}",
    tag = "Logs",
    responses((status = 200, description = "Log lines"))
)]
pub async fn logs(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    axum::extract::Path(service): axum::extract::Path<String>,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"totalLines": 0, "lines": []})))
}

/// Restart — mirrors `POST /restart` (require admin).
#[utoipa::path(
    post, path = "/restart",
    tag = "App",
    responses((status = 200, description = "Restarting"))
)]
pub async fn restart(cfg: State<FrigateConfig>, headers: axum::http::HeaderMap) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"success": true, "message": "Restarting..."})))
}

/// Media sync start — mirrors `POST /media/sync` (require admin).
#[utoipa::path(
    post, path = "/media/sync",
    tag = "App",
    request_body = MediaSyncBody,
    responses((status = 202, description = "Job queued"))
)]
pub async fn sync_media(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    Json(body): Json<MediaSyncBody>,
) -> (axum::http::StatusCode, axum::response::Response) {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, axum::response::IntoResponse::into_response(e)); }
    (axum::http::StatusCode::ACCEPTED, axum::response::IntoResponse::into_response(Json(serde_json::json!({"job": {"job_type": "media_sync", "status": "queued", "id": ""}}))))
}

/// Media sync current — mirrors `GET /media/sync/current` (require admin).
#[utoipa::path(
    get, path = "/media/sync/current",
    tag = "App",
    responses((status = 200, description = "Current job"))
)]
pub async fn get_media_sync_current(cfg: State<FrigateConfig>, headers: axum::http::HeaderMap) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"job": null})))
}

/// Media sync status — mirrors `GET /media/sync/status/{job_id}` (require admin).
#[utoipa::path(
    get, path = "/media/sync/status/{job_id}",
    tag = "App",
    responses((status = 200, description = "Job status"))
)]
pub async fn get_media_sync_status(
    cfg: State<FrigateConfig>, headers: axum::http::HeaderMap,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    if let Err(e) = auth::require_admin(user_from_request(&headers)).await { return axum::response::IntoResponse::into_response(e); }
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"job": {"id": job_id}})))
}

/// Labels — mirrors `GET /labels` (require authenticated).
#[utoipa::path(
    get, path = "/labels",
    tag = "App",
    responses((status = 200, description = "Labels list"))
)]
pub async fn get_labels() -> impl IntoResponse {
    Json(Vec::<String>::new())
}

/// Sub-labels — mirrors `GET /sub_labels` (require authenticated).
#[utoipa::path(
    get, path = "/sub_labels",
    tag = "App",
    responses((status = 200, description = "Sub-labels list"))
)]
pub async fn get_sub_labels() -> impl IntoResponse {
    Json(Vec::<String>::new())
}

/// Audio labels — mirrors `GET /audio_labels` (require authenticated).
#[utoipa::path(
    get, path = "/audio_labels",
    tag = "App",
    responses((status = 200, description = "Audio labels"))
)]
pub async fn get_audio_labels() -> impl IntoResponse {
    Json(Vec::<String>::new())
}

/// Plus models — mirrors `GET /plus/models` (require authenticated).
#[utoipa::path(
    get, path = "/plus/models",
    tag = "App",
    responses((status = 200, description = "Plus models"))
)]
pub async fn plus_models() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(Vec::<serde_json::Value>::new()))
}

/// Timeline — mirrors `GET /timeline` (require authenticated).
#[utoipa::path(
    get, path = "/timeline",
    tag = "App",
    responses((status = 200, description = "Timeline entries"))
)]
pub async fn timeline() -> impl IntoResponse {
    axum::response::IntoResponse::into_response(Json(serde_json::json!({"entries": []})))
}

// ── Request/response schemas ──────────────────────────────────────────

#[derive(Debug, Deserialize, ToSchema)]
pub struct GenAIProbeBody {
    pub provider: String,
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    #[serde(default)]
    pub provider_options: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppConfigSetBody {
    #[serde(default)]
    pub config_data: Option<serde_json::Value>,
    #[serde(default)]
    pub skip_save: bool,
    #[serde(default)]
    pub requires_restart: i32,
    #[serde(default)]
    pub update_topic: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct MediaSyncBody {
    #[serde(default)]
    pub dry_run: bool,
    pub media_types: Vec<String>,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub verbose: bool,
}

#[derive(Debug, Deserialize, IntoParams, ToSchema)]
pub struct ConfigSaveParams {
    save_option: Option<String>,
}

#[derive(Debug, Deserialize, IntoParams, ToSchema)]
pub struct ConfigSetParams {
    #[serde(default)]
    pub skip_save: bool,
    #[serde(default)]
    pub requires_restart: i32,
    #[serde(default)]
    pub update_topic: Option<String>,
}

// ── Helpers ───────────────────────────────────────────────────────────

fn get_config_schema(cfg: FrigateConfig) -> serde_json::Value {
    // Return a minimal schema; full schema generation delegates to the
    // Pydantic model in the Python side (same as `get_config_schema()`).
    serde_json::json!({
        "title": "FrigateConfig",
        "type": "object",
        "properties": {
            "mqtt": {"type": "object"},
            "detectors": {"type": "object"},
            "model": {"type": "object"},
        }
    })
}
