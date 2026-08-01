use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::auth;

/// Inbound topic authorization for WebSocket clients.
///
/// Mirrors `_check_ws_authorization()` in `comms/ws.py`. IPC topics are
/// blocked regardless of role; admin can send anything; restricted roles
/// are limited to read-only topics and camera-scoped commands.

const WS_BLOCKED_TOPICS: &[&str] = &[
    "insert_many_recordings",
    "insert_preview",
    "request_region_grid",
    "upsert_review_segment",
    "clear_ongoing_review_segments",
    "update_camera_activity",
    "update_audio_activity",
    "expire_audio_activity",
    "update_event_description",
    "update_review_description",
    "update_model_state",
    "embeddings_reindex_progress",
    "update_birdseye_layout",
    "update_audio_transcription_state",
];

const WS_VIEWER_TOPICS: &[&str] = &[
    "onConnect",
    "modelState",
    "audioTranscriptionState",
    "birdseyeLayout",
    "embeddingsReindexProgress",
    "jobState",
];

const WS_CAMERA_COMMAND_TOPICS: &[&str] = &["ptz"];

/// Check if an inbound WebSocket message topic is authorized.
///
/// Returns `Ok(())` when the topic is allowed, `Err` otherwise.
pub fn check_ws_authorization(
    topic: &str,
    role_header: Option<&str>,
    separator: &str,
    roles_config: &std::collections::HashMap<String, Vec<String>>,
    camera_names: &HashSet<String>,
) -> Result<(), String> {
    // Block IPC-only topics unconditionally
    if WS_BLOCKED_TOPICS.contains(&topic) {
        return Err(format!("Blocked IPC topic: {topic}"));
    }

    // No role header: default to viewer (fail-closed)
    let roles: Vec<String> = role_header
        .map(|h| h.split(separator).map(|r| r.trim().to_owned()).filter(|r| !r.is_empty()).collect())
        .unwrap_or_default();

    // Admin can send anything
    if roles.iter().any(|r| r == "admin") {
        return Ok(());
    }

    // Read-only topics any authenticated user can send
    if WS_VIEWER_TOPICS.contains(&topic) {
        return Ok(());
    }

    // Camera-scoped command like "<camera>/ptz"
    if let Some((cam, cmd)) = topic.split_once('/') {
        if WS_CAMERA_COMMAND_TOPICS.iter().any(|c| *c == cmd) {
            let allowed = resolve_allowed_cameras(&roles, roles_config, camera_names);
            if allowed.contains(cam) {
                return Ok(());
            }
        }
    }

    Err(format!("Denied: topic={topic} roles={roles:?}"))
}

/// Resolve which cameras a set of roles can access.
fn resolve_allowed_cameras(
    roles: &[String],
    roles_config: &std::collections::HashMap<String, Vec<String>>,
    camera_names: &HashSet<String>,
) -> HashSet<String> {
    let mut allowed = HashSet::new();
    for role in roles.iter().chain(Some("viewer".into()).iter().filter(|_| roles.is_empty())) {
        if role == "admin" {
            return camera_names.clone();
        }
        if let Some(cam_list) = roles_config.get(role) {
            if cam_list.is_empty() {
                return camera_names.clone(); // full-access role
            }
            allowed.extend(cam_list.clone());
        }
    }
    allowed
}

/// Outbound classification — mirrors `_classify_outbound()` in ws.py.
///
/// Every broadcast is classified into a scope, then materialized per recipient.
/// Unknown topics are dropped (fail-closed).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutboundScope {
    Global,
    Drop,
    UnrestrictedOnly,
    Camera(String),
    PayloadCamera(Vec<String>),
    ReshapeByCameraKey,
    ReshapeJobState,
    ReshapeStats,
}

/// Classify an outbound topic into (scope kind, extra).
pub fn classify_outbound(topic: &str, all_cameras: &HashSet<String>, all_zones: &HashSet<String>) -> OutboundScope {
    const GLOBAL: &[&str] = &[
        "model_state",
        "embeddings_reindex_progress",
        "audio_transcription_state",
        "profile/state",
        "notifications/state",
        "notification_test",
    ];
    const UNRESTRICTED: &[&str] = &["birdseye_layout"];
    const RESHAPE_BY_KEY: &[&str] = &["camera_activity", "audio_detections"];
    const RESHAPE_JOB: &[&str] = &["job_state"];
    const RESHAPE_STATS: &[&str] = &["stats"];
    const PAYLOAD_CAMERA: &[(&str, &[&str])] = &[
        ("events", &["after", "camera"]),
        ("reviews", &["after", "camera"]),
        ("tracked_object_update", &["camera"]),
        ("triggers", &["camera"]),
        ("camera_monitoring", &["camera"]),
    ];

    if GLOBAL.contains(&topic) {
        return OutboundScope::Global;
    }
    if UNRESTRICTED.contains(&topic) {
        return OutboundScope::UnrestrictedOnly;
    }
    if RESHAPE_BY_KEY.contains(&topic) {
        return OutboundScope::ReshapeByCameraKey;
    }
    if RESHAPE_JOB.contains(&topic) {
        return OutboundScope::ReshapeJobState;
    }
    if RESHAPE_STATS.contains(&topic) {
        return OutboundScope::ReshapeStats;
    }
    if let Some((_, path)) = PAYLOAD_CAMERA.iter().find(|(t, _)| *t == topic) {
        return OutboundScope::PayloadCamera(path.iter().map(|s| (*s).to_owned()).collect());
    }

    // Prefix-based: first segment names owning camera or zone
    let first = topic.split('/').next().unwrap_or(topic);
    if all_cameras.contains(first) {
        return OutboundScope::Camera(first.to_owned());
    }
    if all_zones.contains(first) {
        return OutboundScope::UnrestrictedOnly;
    }

    OutboundScope::Drop
}

/// Whether a WebSocket connection has unrestricted camera access.
///
/// Mirrors `_ws_is_unrestricted()` — admin or any role with an empty
/// allow-list grants full access.
pub fn ws_is_unrestricted(
    role: &str,
    roles_config: &std::collections::HashMap<String, Vec<String>>,
) -> bool {
    if role == "admin" {
        return true;
    }
    roles_config.get(role).map(|v| v.is_empty()).unwrap_or(false)
}

/// Check camera access for a single camera — mirrors `ws_has_camera_access()`.
pub fn ws_has_camera_access(
    role: &str,
    camera: &str,
    roles_config: &std::collections::HashMap<String, Vec<String>>,
    camera_names: &HashSet<String>,
) -> bool {
    if role == "admin" || !roles_config.get(role).map(|v| v.is_empty()).unwrap_or(false) {
        return true;
    }
    roles_config
        .get(role)
        .map(|allowed| allowed.contains(&camera.to_owned()))
        .unwrap_or(false)
}

/// Envelope a payload with topic — mirrors `_wrap_envelope()`.
pub fn wrap_envelope(topic: &str, payload: &serde_json::Value) -> String {
    serde_json::json!({"topic": topic, "payload": payload.to_string()}).to_string()
}
