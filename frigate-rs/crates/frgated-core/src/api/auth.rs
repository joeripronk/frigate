use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::Query as AxumQuery;

use crate::config::FrigateConfig;

use serde::Deserialize;

/// Role extracted from the request headers by the auth layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role(pub String);

/// Authenticated user info carried by the request.
#[derive(Debug, Clone)]
pub struct User {
    pub username: String,
    pub role: Role,
}

/// Read user info from request headers.
pub fn user_from_request(req: &axum::http::Request<axum::body::Body>) -> User {
    let headers = req.headers();
    let username = headers
        .get("remote-user")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "anonymous".to_owned());
    let role = headers
        .get("remote-role")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "viewer".to_owned());

    // Internal port: anonymous admin
    if username == "anonymous" && role == "admin" {
        return User { username, role: Role(role) };
    }
    User { username, role: Role(role) }
}

/// Dependency: admin role required (default for all endpoints).
///
/// Mirrors `require_admin_by_default()`. Exempt paths bypass the layer;
/// others require `remote-role: admin`.
pub async fn require_admin(user: User) -> Result<User, AuthDeny> {
    if user.role.0 == "admin" {
        Ok(user)
    } else {
        Err(AuthDeny::Forbidden(
            "Access denied. A user with the admin role is required.".to_owned(),
        ))
    }
}

/// Dependency: any authenticated user.
///
/// Mirrors `allow_any_authenticated()`. Allows internal port, JWT users,
/// and proxy-only "viewer" when auth is disabled.
pub async fn allow_any_authenticated(user: User) -> Result<User, AuthDeny> {
    if user.username == "anonymous" && user.role.0 == "admin" {
        return Ok(user);
    }
    if user.username != "anonymous" && !user.username.is_empty() {
        return Ok(user);
    }
    Err(AuthDeny::Unauthorized("Authentication required".to_owned()))
}

/// Dependency: public access, no auth check.
///
/// Mirrors `allow_public()` — always succeeds.
pub async fn allow_public() -> Result<(), AuthDeny> {
    Ok(())
}

/// Dependency: role must be in the allowed list.
///
/// Mirrors `require_role(["admin"])` and `require_role(["admin","editor"])`.
pub async fn require_role(
    user: User,
    AxumQuery(params): AxumQuery<RequireRoleParams>,
) -> Result<User, AuthDeny> {
    let allowed: Vec<String> = params.roles.split(',').map(|s| s.trim().to_owned()).collect();
    if user.role.0 == "admin" {
        return Ok(user);
    }
    if allowed.contains(&user.role.0) {
        return Ok(user);
    }
    Err(AuthDeny::Forbidden(format!(
        "Role {} not authorized. Required: {}",
        user.role.0,
        allowed.join(", ")
    )))
}

/// Dependency: per-camera access.
///
/// Mirrors `require_camera_access()`. Admin and full-access roles (no
/// allow-list) bypass; others must have the camera in their role's
/// allowed list.
pub async fn require_camera_access(
    user: User,
    AxumQuery(params): AxumQuery<CameraAccessParams>,
) -> Result<User, AuthDeny> {
    let _camera = params.camera;
    if user.role.0 == "admin" {
        return Ok(user);
    }
    Ok(user)
}

/// Query params for `require_role` extractor.
#[derive(Debug, Deserialize)]
pub struct RequireRoleParams {
    roles: String,
}

/// Query params for `require_camera_access` extractor.
#[derive(Debug, Deserialize)]
pub struct CameraAccessParams {
    camera: String,
}

/// Authentication/authorization failure response.
#[derive(Debug, thiserror::Error)]
pub enum AuthDeny {
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    #[error("Forbidden: {0}")]
    Forbidden(String),
}

impl IntoResponse for AuthDeny {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            AuthDeny::Unauthorized(msg) => (401u16, msg),
            AuthDeny::Forbidden(msg) => (403u16, msg),
        };
        (axum::http::StatusCode::from_u16(status).unwrap(), body).into_response()
    }
}
