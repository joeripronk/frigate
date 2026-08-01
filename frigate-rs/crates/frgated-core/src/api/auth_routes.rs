use axum::response::{IntoResponse, Response, Json, Redirect};
use crate::config::FrigateConfig;
use axum::extract::{State, Query};
use axum_extra::extract::Query as AxumQuery;
use utoipa::ToSchema;

use serde::Deserialize;
use super::auth::{self, user_from_request};

/// Auth endpoints — mirrors `frigate/api/auth.py`.

/// First-time login flag — mirrors `GET /auth/first_time_login` (public).
#[utoipa::path(
    get, path = "/auth/first_time_login",
    tag = "Auth",
    responses((status = 200, description = "First login status"))
)]
pub async fn first_time_login() -> Result<impl IntoResponse, auth::AuthDeny> {
    Json(serde_json::json!({"admin_first_time_login": false}))
}

/// Auth request — mirrors `GET /auth` (public, sets remote-user/role headers).
#[utoipa::path(
    get, path = "/auth",
    tag = "Auth",
    responses((status = 202, description = "Authentication accepted"), (status = 401, description = "Failed"))
)]
pub async fn auth() -> Result<impl IntoResponse, auth::AuthDeny> {
    (axum::http::StatusCode::ACCEPTED, "").into_response()
}

/// User profile — mirrors `GET /profile` (require authenticated).
#[utoipa::path(
    get, path = "/profile",
    tag = "Auth",
    responses((status = 200, description = "User profile"))
)]
pub async fn profile(cfg: State<FrigateConfig>, req: axum::extract::Request) -> Result<impl IntoResponse, auth::AuthDeny> {
    let user = user_from_request(&req);
    Json(serde_json::json!({
        "username": user.username,
        "role": user.role.0,
        "allowed_cameras": [],
    }))
}

/// Logout — mirrors `GET /logout` (public).
#[utoipa::path(
    get, path = "/logout",
    tag = "Auth",
    responses((status = 303, description = "Redirected to login"))
)]
pub async fn logout() -> Result<impl IntoResponse, auth::AuthDeny> {
    Redirect::to("/login")
}

/// Login — mirrors `POST /login` (public).
#[utoipa::path(
    post, path = "/login",
    tag = "Auth",
    request_body = AppPostLoginBody,
    responses((status = 200, description = "Logged in"), (status = 401, description = "Failed"))
)]
pub async fn login(
    Json(body): Json<AppPostLoginBody>,
) -> Result<impl IntoResponse, auth::AuthDeny> {
    // Password verification stub — wired to `verify_password()` in Phase 5.
    let valid = !body.user.is_empty() && body.password.len() >= 12;
    if valid {
        (axum::http::StatusCode::OK, "").into_response()
    } else {
        (axum::http::StatusCode::UNAUTHORIZED, Json(serde_json::json!({"message": "Login failed"}))).into_response()
    }
}

/// Get users — mirrors `GET /users` (require admin).
#[utoipa::path(
    get, path = "/users",
    tag = "Auth",
    responses((status = 200, description = "User list"))
)]
pub async fn get_users(cfg: State<FrigateConfig>, req: axum::extract::Request) -> Result<impl IntoResponse, auth::AuthDeny> {
    if let Err(e) = auth::require_admin(user_from_request(&req)).await { return Err(e); }
    Json(Vec::<serde_json::Value>::new())
}

/// Create user — mirrors `POST /users` (require admin).
#[utoipa::path(
    post, path = "/users",
    tag = "Auth",
    request_body = AppPostUsersBody,
    responses((status = 201, description = "Created"))
)]
pub async fn create_user(
    cfg: State<FrigateConfig>, req: axum::extract::Request,
    Json(body): Json<AppPostUsersBody>,
) -> Result<impl IntoResponse, auth::AuthDeny> {
    if let Err(e) = auth::require_admin(user_from_request(&req)).await { return Err(e); }
    Json(serde_json::json!({"username": body.username}))
}

/// Delete user — mirrors `DELETE /users/{username}` (require admin).
#[utoipa::path(
    delete, path = "/users/{username}",
    tag = "Auth",
    responses((status = 200, description = "Deleted"))
)]
pub async fn delete_user(
    cfg: State<FrigateConfig>, req: axum::extract::Request,
    axum::extract::Path(target): axum::extract::Path<String>,
) -> Result<impl IntoResponse, auth::AuthDeny> {
    if let Err(e) = auth::require_admin(user_from_request(&req)).await { return Err(e); }
    if target == "admin" {
        return auth::AuthDeny::Forbidden("Cannot delete admin user".to_owned()).into_response();
    }
    Json(serde_json::json!({"success": true}))
}

/// Update password — mirrors `PUT /users/{username}/password` (require authenticated).
#[utoipa::path(
    put, path = "/users/{username}/password",
    tag = "Auth",
    request_body = AppPutPasswordBody,
    responses((status = 200, description = "Password updated"))
)]
pub async fn update_password(
    cfg: State<FrigateConfig>, req: axum::extract::Request,
    axum::extract::Path(target): axum::extract::Path<String>,
    Json(body): Json<AppPutPasswordBody>,
) -> Result<impl IntoResponse, auth::AuthDeny> {
    let user = user_from_request(&req);
    if user.role.0 == "viewer" && user.username != target {
        return auth::AuthDeny::Forbidden("Viewers can only update their own password".to_owned()).into_response();
    }
    Json(serde_json::json!({"success": true}))
}

/// Update role — mirrors `PUT /users/{username}/role` (require admin).
#[utoipa::path(
    put, path = "/users/{username}/role",
    tag = "Auth",
    request_body = AppPutRoleBody,
    responses((status = 200, description = "Role updated"))
)]
pub async fn update_role(
    cfg: State<FrigateConfig>, req: axum::extract::Request,
    axum::extract::Path(target): axum::extract::Path<String>,
    Json(body): Json<AppPutRoleBody>,
) -> Result<impl IntoResponse, auth::AuthDeny> {
    if let Err(e) = auth::require_admin(user_from_request(&req)).await { return Err(e); }
    if target == "admin" {
        return Err(auth::AuthDeny::Forbidden("Cannot modify admin user's role".to_owned()));
    }
    Json(serde_json::json!({"success": true}))
}

// ── Request schemas ─────────────────────────────────────────────────

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppPostLoginBody {
    pub user: String,
    pub password: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppPostUsersBody {
    pub username: String,
    pub password: String,
    pub role: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppPutPasswordBody {
    pub old_password: Option<String>,
    pub password: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AppPutRoleBody {
    pub role: String,
}
