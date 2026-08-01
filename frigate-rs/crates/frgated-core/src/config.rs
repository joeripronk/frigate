use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Path where Frigate stores config and DB — mirrors `frigate.const`.
pub const CONFIG_DIR: &str = "/config";
pub const DEFAULT_DB_PATH: &str = "/config/frigate.db";

/// Where `config.yml` lives, with the same defaults as the Python side.
pub fn config_path() -> PathBuf {
    PathBuf::from(CONFIG_DIR).join("config.yml")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrigateConfig {
    pub version: Option<String>,
    pub safe_mode: bool,
    pub mqtt: MqttConfig,
    pub cameras: HashMap<String, CameraConfig>,
    pub detectors: HashMap<String, DetectorConfig>,
    pub model: ModelConfig,
    pub database: DatabaseConfig,
    pub auth: AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    #[serde(default)]
    pub roles: HashMap<String, Vec<String>>,
}

impl FrigateConfig {
    /// Load config.yml and validate it by shelling out to the Python parser.
    pub fn load() -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(config_path())?;
        Self::validate_and_parse(&raw)
    }

    /// Validate a config string via the Python parser (same path as FrigateConfig.parse).
    pub fn validate_and_parse(raw: &str) -> anyhow::Result<Self> {
        let mut child = Command::new("python3")
            .args(["-c", VALIDATE_SCRIPT])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin.write_all(raw.as_bytes())?;
        }

        let out = child.wait_with_output()?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            anyhow::bail!("Python validation failed: {stderr}");
        }
        let json = String::from_utf8(out.stdout)?;
        let cfg: FrigateConfig = serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(cfg)
    }

    /// Validate config text without parsing into Rust types. Returns Ok/Err with messages.
    pub fn validate(raw: &str) -> Result<(), Vec<String>> {
        let mut child = Command::new("python3")
            .args(["-c", VALIDATE_SCRIPT])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to start python3");

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin.write_all(raw.as_bytes()).expect("write stdin");
        }

        let out = child.wait_with_output().expect("wait python");
        if out.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        let messages: Vec<String> = stderr.lines().map(|l| l.to_owned()).collect();
        Err(messages)
    }

    /// Load a safe (minimal) config for recovery mode — mirrors `safe_load=True`.
    pub fn load_safe() -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(config_path())?;
        let safe = format!(
            "safe_mode: true\ncameras: {{}}\nmqtt:\n  enabled: false\n{raw}"
        );
        Self::validate_and_parse(&safe)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    #[serde(rename = "type")]
    pub type_field: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub width: u32,
    pub height: u32,
    pub input_tensor: String,
    pub input_pixel_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub ffmpeg: CameraFfmpegConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFfmpegConfig {
    pub inputs: Vec<CameraInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInput {
    pub path: String,
    #[serde(rename = "roles")]
    pub roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
}

/// Minimal validation script — delegates to Pydantic and emits JSON on success.
const VALIDATE_SCRIPT: &str = r#"
import sys, json
sys.path.insert(0, "/opt/frigate")
from frigate.config.config import FrigateConfig
from ruamel.yaml import YAML
yaml = YAML()
data = yaml.load(sys.stdin.read())
out = FrigateConfig.model_validate(data)
# strip the PlusApi context field (unserialisable)
d = out.model_dump(exclude={"_plus_api"})
json.dump(d, sys.stdout)
"#;
