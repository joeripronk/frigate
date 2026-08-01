-- Create events table (mirrors final state after peewee migrations 001-020)
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    label TEXT NOT NULL,
    sub_label TEXT,
    camera TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    top_score REAL NOT NULL,
    score REAL NOT NULL,
    false_positive INTEGER NOT NULL,
    zones TEXT NOT NULL,
    thumbnail TEXT NOT NULL,
    has_clip INTEGER NOT NULL DEFAULT 1,
    has_snapshot INTEGER NOT NULL DEFAULT 1,
    region TEXT NOT NULL,
    box TEXT NOT NULL,
    area INTEGER NOT NULL,
    retain_indefinitely INTEGER NOT NULL DEFAULT 0,
    ratio REAL NOT NULL DEFAULT 1.0,
    plus_id TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    detector_type TEXT NOT NULL,
    model_type TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_label ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);

-- Create timeline table (mirrors peewee migration 013)
CREATE TABLE IF NOT EXISTS timeline (
    timestamp DATETIME NOT NULL,
    camera TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    class_type TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timeline_camera ON timeline(camera);
CREATE INDEX IF NOT EXISTS idx_timeline_source ON timeline(source);
CREATE INDEX IF NOT EXISTS idx_timeline_source_id ON timeline(source_id);

-- Create regions table (mirrors peewee migration 019)
CREATE TABLE IF NOT EXISTS regions (
    camera TEXT PRIMARY KEY NOT NULL,
    grid TEXT NOT NULL,
    last_update DATETIME NOT NULL
);

-- Create recordings table (mirrors peewee migration 003 + subsequent migrations)
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY NOT NULL,
    camera TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    duration REAL NOT NULL,
    motion INTEGER,
    objects INTEGER,
    dBFS INTEGER,
    segment_size REAL NOT NULL DEFAULT 0,
    regions INTEGER,
    motion_heatmap TEXT
);

CREATE INDEX IF NOT EXISTS idx_recordings_camera ON recordings(camera);

-- Create previews table (mirrors peewee migration 021)
CREATE TABLE IF NOT EXISTS previews (
    id TEXT PRIMARY KEY NOT NULL,
    camera TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    duration REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_previews_camera ON previews(camera);

-- Create review_segment table (mirrors peewee migration 022)
CREATE TABLE IF NOT EXISTS review_segment (
    id TEXT PRIMARY KEY NOT NULL,
    camera TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    severity TEXT NOT NULL,
    thumb_path TEXT NOT NULL UNIQUE,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_segment_camera ON review_segment(camera);

-- Create user_review_status table (mirrors peewee migration 030)
CREATE TABLE IF NOT EXISTS user_review_status (
    user_id TEXT NOT NULL,
    review_segment_id TEXT NOT NULL,
    has_been_reviewed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, review_segment_id)
);

-- Create user table (mirrors peewee migration 025 + 029 + 032)
CREATE TABLE IF NOT EXISTS user (
    username TEXT PRIMARY KEY NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    password_hash TEXT NOT NULL,
    password_changed_at DATETIME,
    notification_tokens TEXT NOT NULL
);

-- Create trigger table (mirrors peewee migration 031)
CREATE TABLE IF NOT EXISTS trigger (
    camera TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    threshold REAL NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    triggering_event_id TEXT NOT NULL,
    last_triggered DATETIME NOT NULL,
    PRIMARY KEY (camera, name)
);

-- Create export_case table (mirrors peewee migration 033)
CREATE TABLE IF NOT EXISTS export_case (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_export_case_name ON export_case(name);

-- Create export table (mirrors peewee migration 024 + 034)
CREATE TABLE IF NOT EXISTS export (
    id TEXT PRIMARY KEY NOT NULL,
    camera TEXT NOT NULL,
    name TEXT NOT NULL,
    date DATETIME NOT NULL,
    video_path TEXT NOT NULL UNIQUE,
    thumb_path TEXT NOT NULL UNIQUE,
    in_progress INTEGER NOT NULL,
    export_case_id TEXT,
    FOREIGN KEY (export_case_id) REFERENCES export_case(id)
);

CREATE INDEX IF NOT EXISTS idx_export_camera ON export(camera);
CREATE INDEX IF NOT EXISTS idx_export_name ON export(name);
