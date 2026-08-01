use sqlx::sqlite::SqlitePool;

use crate::config::FrigateConfig;

const SQLITE_VEC_PATH: &str = "/usr/local/lib/vec0";

/// Configure SQLite pragmas on a newly opened connection.
///
/// Mirrors the Python `SqliteVecQueueDatabase` settings:
/// - `auto_vacuum FULL`
/// - `cache_size` −512000 (512 MB, negative = KB)
/// - `synchronous NORMAL` (WAL-safe)
async fn configure_pragmas(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query("PRAGMA auto_vacuum = FULL")
        .execute(pool)
        .await?;
    sqlx::query("PRAGMA cache_size = -512000")
        .execute(pool)
        .await?;
    sqlx::query("PRAGMA synchronous = NORMAL")
        .execute(pool)
        .await?;
    sqlx::query("PRAGMA journal_mode = WAL")
        .execute(pool)
        .await?;
    Ok(())
}

/// Attach the sqlite-vec extension so that vec0 virtual tables are available.
///
/// Mirrors the Python `_load_vec_extension()`:
/// - `conn.enable_load_extension(True)`
/// - `conn.load_extension("/usr/local/lib/vec0")`
/// - Graceful degradation on failure
async fn attach_vec_extension(pool: &SqlitePool) -> bool {
    let attach_query = format!(
        "ATTACH '{}' AS vec; CREATE VIRTUAL TABLE IF NOT EXISTS vec.thumbnails USING vec0(id TEXT PRIMARY KEY, thumbnail_embedding FLOAT[768] distance_metric=cosine); CREATE VIRTUAL TABLE IF NOT EXISTS vec.descriptions USING vec0(id TEXT PRIMARY KEY, description_embedding FLOAT[768] distance_metric=cosine);",
        SQLITE_VEC_PATH
    );

    match sqlx::query(&attach_query).execute(pool).await {
        Ok(_) => {
            tracing::info!("sqlite-vec extension attached successfully");
            true
        }
        Err(e) => {
            tracing::error!("Unable to attach sqlite-vec extension: {}", e);
            false
        }
    }
}

/// Create the REGEXP function for pattern matching.
///
/// Mirrors the Python `_register_regexp()` which delegates to the `regex` module.
/// sqlx's `regexp` feature provides this automatically via the `regexp` crate.
async fn register_regexp(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    // sqlx's `regexp` feature registers REGEXP using the `regex` crate internally.
    // We verify it works by running a test query.
    sqlx::query("SELECT 1 WHERE 'hello' REGEXP 'h.llo'")
        .fetch_one(pool)
        .await?;
    Ok(())
}

/// Initialize the database pool with all required configuration.
///
/// This mirrors `FrigateApp.bind_database()` in `app.py`:
/// 1. Create SqlitePool
/// 2. Configure pragmas (auto_vacuum, cache_size, synchronous, journal_mode)
/// 3. Attach sqlite-vec extension
/// 4. Register REGEXP function
pub async fn init_database(config: &FrigateConfig) -> Result<SqlitePool, anyhow::Error> {
    let db_path = config.database.path.as_str();
    tracing::info!("Initializing database at {}", db_path);

    let pool = SqlitePool::connect(&format!("sqlite:{}", db_path)).await?;

    configure_pragmas(&pool).await?;
    let vec_loaded = attach_vec_extension(&pool).await;

    if vec_loaded {
        register_regexp(&pool).await?;
    }

    tracing::info!("Database initialized successfully");
    Ok(pool)
}

/// Run a raw SQL query for migrations or one-off operations.
pub async fn execute_raw(pool: &SqlitePool, sql: &str) -> Result<(), sqlx::Error> {
    sqlx::query(sql).execute(pool).await?;
    Ok(())
}
