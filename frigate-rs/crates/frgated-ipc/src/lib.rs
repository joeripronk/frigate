mod channels;
mod shared_memory;
mod zmq;

pub use channels::*;
pub use shared_memory::*;
pub use zmq::*;

/// Fast poll timeout — mirrors Python `FAST_QUEUE_TIMEOUT` (10µs).
pub const FAST_QUEUE_TIMEOUT: std::time::Duration = std::time::Duration::from_micros(10);
