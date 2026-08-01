use tracing::warn;
use zmq::Context;

use crate::FAST_QUEUE_TIMEOUT;

/// PUB socket publisher — mirrors `frigate/comms/zmq_proxy.py::Publisher`.
pub struct Publisher {
    socket: zmq::Socket,
    topic: String,
}

impl Publisher {
    pub fn new(topic: &str) -> Self {
        let ctx = Context::new();
        let socket = ctx.socket(zmq::PUB).expect("create PUB");
        socket.connect("ipc:///tmp/cache/proxy_pub").expect("connect PUB");
        Self {
            socket,
            topic: topic.to_owned(),
        }
    }

    pub fn new_with_context(ctx: &Context, topic: &str) -> Self {
        let socket = ctx.socket(zmq::PUB).expect("create PUB");
        socket.connect("ipc:///tmp/cache/proxy_pub").expect("connect PUB");
        Self {
            socket,
            topic: topic.to_owned(),
        }
    }

    /// Publish `payload` under `topic` (same wire format as Python: `"topic subtopic {json}"`).
    pub fn publish(&self, sub_topic: &str, payload: &serde_json::Value) {
        let msg = format!("{}{} {}", self.topic, sub_topic, payload);
        if let Err(e) = self.socket.send_str(&msg, zmq::DONTWAIT) {
            warn!("PUB send failed: {e}");
        }
    }
}

/// SUB socket subscriber — mirrors `frigate/comms/zmq_proxy.py::Subscriber`.
pub struct Subscriber {
    socket: zmq::Socket,
    topic: String,
}

impl Subscriber {
    pub fn new(topic: &str) -> Self {
        let ctx = Context::new();
        let socket = ctx.socket(zmq::SUB).expect("create SUB");
        socket
            .set_subscribe(topic.as_bytes())
            .expect("subscribe");
        socket.connect("ipc:///tmp/cache/proxy_sub").expect("connect SUB");
        Self {
            socket,
            topic: topic.to_owned(),
        }
    }

    pub fn new_with_context(ctx: &Context, topic: &str) -> Self {
        let socket = ctx.socket(zmq::SUB).expect("create SUB");
        socket
            .set_subscribe(topic.as_bytes())
            .expect("subscribe");
        socket.connect("ipc:///tmp/cache/proxy_sub").expect("connect SUB");
        Self {
            socket,
            topic: topic.to_owned(),
        }
    }

    /// Poll for a message; returns `(topic, payload)` or `("", None)` on timeout.
    pub fn check_for_update(&self) -> (String, Option<serde_json::Value>) {
        let pollitem = self.socket.as_poll_item(zmq::POLLIN);
        if zmq::poll(&mut [pollitem], FAST_QUEUE_TIMEOUT.as_millis() as i64)
            .unwrap_or(0)
            == 0
        {
            return (String::new(), None);
        }

        match self.socket.recv_string(zmq::DONTWAIT) {
            Ok(Ok(raw)) => {
                let parts: Vec<&str> = raw.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let payload = serde_json::from_str(parts[1]).ok();
                    (parts[0].to_owned(), payload)
                } else {
                    (raw, None)
                }
            }
            _ => (String::new(), None),
        }
    }
}


