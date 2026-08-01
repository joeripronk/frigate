use std::path::PathBuf;

use tracing::{info, warn};

/// Shared-memory region backed by `/dev/shm/frigate-{name}`.
///
/// Replaces `UntrackedSharedMemory` (Python `multiprocessing.shared_memory`) —
/// same create/open semantics, no tracker registration.
///
/// On crash trap (`SIGQUIT`/`SIGABRT`), the cleanup loop unlinks stale
/// `/dev/shm/frigate-*` entries, mirroring `stop()`'s `shm.unlink()`.
pub struct SharedRegion {
    name: String,
    path: PathBuf,
    size: usize,
    mapped: Option<memmap2::Mmap>,
}

impl SharedRegion {
    pub fn create(name: &str, size: usize) -> Result<Self, SharedMemoryError> {
        let base = PathBuf::from("/dev/shm");
        let path = base.join(format!("frigate-{name}"));

        // Truncate file to size (creates if absent, overwrites if present).
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| SharedMemoryError::Create(name.to_string(), e))?;

        file.set_len(size as u64)
            .map_err(|e| SharedMemoryError::Resize(name.to_string(), e))?;

        let mapped = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| SharedMemoryError::Map(name.to_string(), e))?
        };

        info!("Created shared region '{name}' ({size} bytes) at {path:?}");
        Ok(Self {
            name: name.to_owned(),
            path,
            size,
            mapped: Some(mapped),
        })
    }

    pub fn open(name: &str) -> Result<Self, SharedMemoryError> {
        let base = PathBuf::from("/dev/shm");
        let path = base.join(format!("frigate-{name}"));

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SharedMemoryError::Open(name.to_string(), e))?;

        let len = file
            .metadata()
            .map_err(|e| SharedMemoryError::Meta(name.to_string(), e))?.len() as usize;

        let mapped = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| SharedMemoryError::Map(name.to_string(), e))?
        };

        info!("Opened shared region '{name}' ({len} bytes)");
        Ok(Self {
            name: name.to_owned(),
            path,
            size: len,
            mapped: Some(mapped),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn buf(&self) -> &[u8] {
        self.mapped.as_ref().unwrap()
    }

    pub fn buf_mut(&mut self) -> &mut [u8] {
        let ptr = self.mapped.as_ref().unwrap().as_ptr();
        // SAFETY: mmap is MAP_SHARED, writes are visible to other processes.
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, self.size) }
    }

    /// Unlink the shared-memory file. Called on clean exit and by the
    /// crash-cleanup handler.
    pub fn unlink(&self) {
        if self.path.exists() {
            if std::fs::remove_file(&self.path).is_ok() {
                info!("Unlinked shared region '{}'", self.name);
            } else {
                warn!("Failed to unlink shared region '{}'", self.name);
            }
        }
    }
}

impl Drop for SharedRegion {
    fn drop(&mut self) {
        self.unlink();
    }
}

/// Collects stale `/dev/shm/frigate-*` regions left by a crashed parent.
///
/// Installed as a `SIGQUIT`/`SIGABRT` handler: on signal, iterate
/// `/dev/shm` and unlink anything matching `frigate-*`.
pub fn install_crash_cleanup() {
    use nix::sys::signal;

    let handler = signal::SigAction::new(
        signal::SigHandler::Handler(cleanup_shm_handler),
        signal::SaFlags::SA_RESETHAND,
        signal::SigSet::empty(),
    );
    // Ignore errors — signal may already be registered.
    unsafe {
        let _ = signal::sigaction(signal::Signal::SIGQUIT, &handler);
        let _ = signal::sigaction(signal::Signal::SIGABRT, &handler);
    }

    info!("Crash-cleanup handler installed (SIGQUIT/SIGABRT → unlink /dev/shm/frigate-*)");
}

extern "C" fn cleanup_shm_handler(_: i32) {
    if let Ok(entries) = std::fs::read_dir("/dev/shm") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if let Some(ns) = name.to_str() {
                if ns.starts_with("frigate-") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
    // Re-raise to get a core dump / signal-default behavior.
    nix::sys::signal::raise(nix::sys::signal::Signal::SIGQUIT).ok();
}

#[derive(Debug)]
pub enum SharedMemoryError {
    Create(String, std::io::Error),
    Open(String, std::io::Error),
    Map(String, std::io::Error),
    Resize(String, std::io::Error),
    Meta(String, std::io::Error),
}

impl std::fmt::Display for SharedMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(n, e) => write!(f, "create shared region '{n}': {e}"),
            Self::Open(n, e) => write!(f, "open shared region '{n}': {e}"),
            Self::Map(n, e) => write!(f, "map shared region '{n}': {e}"),
            Self::Resize(n, e) => write!(f, "resize shared region '{n}': {e}"),
            Self::Meta(n, e) => write!(f, "metadata shared region '{n}': {e}"),
        }
    }
}

impl std::error::Error for SharedMemoryError {}
