use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

// Event model — mirrors frigate/models.py Event
// TODO: region, box, area columns marked for removal
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Event {
    pub id: String,
    pub label: String,
    pub sub_label: Option<String>,
    pub camera: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub top_score: f64,
    pub score: f64,
    pub false_positive: bool,
    pub zones: String,
    pub thumbnail: String,
    pub has_clip: bool,
    pub has_snapshot: bool,
    pub region: String,
    pub r#box: String,
    pub area: i32,
    pub retain_indefinitely: bool,
    pub ratio: f64,
    pub plus_id: String,
    pub model_hash: String,
    pub detector_type: String,
    pub model_type: String,
    pub data: String,
}

// Timeline model — mirrors frigate/models.py Timeline
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Timeline {
    pub timestamp: DateTime<Utc>,
    pub camera: String,
    pub source: String,
    pub source_id: String,
    pub class_type: String,
    pub data: String,
}

// Regions model — mirrors frigate/models.py Regions
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Regions {
    pub camera: String,
    pub grid: String,
    pub last_update: DateTime<Utc>,
}

// Recordings model — mirrors frigate/models.py Recordings
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Recordings {
    pub id: String,
    pub camera: String,
    pub path: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration: f64,
    pub motion: Option<i32>,
    pub objects: Option<i32>,
    pub dbfs: Option<i32>,
    pub segment_size: f64,
    pub regions: Option<i32>,
    pub motion_heatmap: Option<String>,
}

// ExportCase model — mirrors frigate/models.py ExportCase
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct ExportCase {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Export model — mirrors frigate/models.py Export
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Export {
    pub id: String,
    pub camera: String,
    pub name: String,
    pub date: DateTime<Utc>,
    pub video_path: String,
    pub thumb_path: String,
    pub in_progress: bool,
    pub export_case_id: Option<String>,
}

// ReviewSegment model — mirrors frigate/models.py ReviewSegment
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct ReviewSegment {
    pub id: String,
    pub camera: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub severity: String,
    pub thumb_path: String,
    pub data: String,
}

// UserReviewStatus model — mirrors frigate/models.py UserReviewStatus
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct UserReviewStatus {
    pub user_id: String,
    pub review_segment_id: String,
    pub has_been_reviewed: bool,
}

// Previews model — mirrors frigate/models.py Previews
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Previews {
    pub id: String,
    pub camera: String,
    pub path: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration: f64,
}

// User model — mirrors frigate/models.py User
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct User {
    pub username: String,
    pub role: String,
    pub password_hash: String,
    pub password_changed_at: Option<DateTime<Utc>>,
    pub notification_tokens: String,
}

// Trigger model — mirrors frigate/models.py Trigger
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Trigger {
    pub camera: String,
    pub name: String,
    pub type_: String,
    pub data: String,
    pub threshold: f64,
    pub model: String,
    pub embedding: Vec<u8>,
    pub triggering_event_id: String,
    pub last_triggered: DateTime<Utc>,
}
