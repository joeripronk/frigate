use std::sync::Arc;

use tracing::{info, warn};
use zmq::Context;

use crate::zmq::pub_sub::Subscriber;

/// ZMQ proxy endpoints — mirrors `frigate/comms/zmq_proxy.py`.
const SOCKET_PUB: &str = "ipc:///tmp/cache/proxy_pub";
const SOCKET_SUB: &str = "ipc:///tmp/cache/proxy_sub";

/// XSUB↔XPUB proxy replacing `ZmqProxyRunner`.
///
/// Blocking run; unblocks on context destroy (same as Python side).
pub struct ZmqProxyRunner {
    context: Arc<Context>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ZmqProxyRunner {
    pub fn new() -> Self {
        let context = Arc::new(Context::new());
        let ctx = context.clone();

        let handle = std::thread::Builder::new()
            .name("detection_proxy".into())
            .spawn(move || {
                let incoming = ctx.socket(zmq::XSUB).expect("create XSUB");
                let outgoing = ctx.socket(zmq::XPUB).expect("create XPUB");

                incoming.bind(SOCKET_PUB).expect("bind proxy PUB");
                outgoing.bind(SOCKET_SUB).expect("bind proxy SUB");

                info!("ZMQ proxy bound (PUB={SOCKET_PUB}, SUB={SOCKET_SUB})");

                // Blocking proxy — stops when sockets are closed.
                zmq::proxy(&incoming, &outgoing).unwrap_or_else(|e| {
                    warn!("ZMQ proxy stopped: {e}");
                });
            })
            .expect("spawn proxy thread");

        Self {
            context,
            handle: Some(handle),
        }
    }

    pub fn publisher(&self, topic: &str) -> Subscriber {
        let _ = topic;
        Subscriber::new_with_context(&self.context, "")
    }

    pub fn subscriber(&self, topic: &str) -> crate::zmq::pub_sub::Subscriber {
        crate::zmq::pub_sub::Subscriber::new_with_context(&self.context, topic)
    }
}

impl Drop for ZmqProxyRunner {
    fn drop(&mut self) {
        info!("Stopping ZMQ proxy");
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        // Context is dropped after the thread, so sockets close first.
    }
}

/// REQ/REP pair replacing `InterProcessCommunicator` (REP side).
pub struct RepSocket {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    _socket: zmq::Socket,
}

impl RepSocket {
    pub fn bind() -> Self {
        let ctx = Context::new();
        let socket = ctx.socket(zmq::REP).expect("create REP socket");
        socket.bind("ipc:///tmp/cache/comms").expect("bind REQ/REP");

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();

        let handle = std::thread::Builder::new()
            .name("ipc_rep_reader".into())
            .spawn(move || {
                while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    let pollitem = socket.as_poll_item(zmq::POLLIN);
                    if zmq::poll(&mut [pollitem], 1000)
                        .map(|n| n > 0)
                        .unwrap_or(false)
                    {
                        match socket.recv_string(zmq::DONTWAIT) {
                            Ok(Ok(raw)) => {
                                let response =
                                    serde_json::from_str::<serde_json::Value>(raw.as_str());
                                let reply = match response {
                                    Ok(v) => v
                                        .as_object()
                                        .and_then(|o| o.get("payload"))
                                        .map(|p| serde_json::to_string(p).unwrap_or_default())
                                        .unwrap_or_default(),
                                    Err(_) => String::new(),
                                };
                                let _ = socket.send(&reply, zmq::DONTWAIT);
                            }
                            _ => {}
                        }
                    }
                }
            })
            .expect("spawn REP reader");

        let dummy = ctx.socket(zmq::REP).expect("create dummy REP");
        Self {
            stop,
            handle: Some(handle),
            _socket: dummy,
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// REQ side of the IPC communication channel.
pub struct ReqSocket {
    socket: zmq::Socket,
}

impl ReqSocket {
    pub fn connect() -> Self {
        let ctx = Context::new();
        let socket = ctx.socket(zmq::REQ).expect("create REQ socket");
        socket.connect("ipc:///tmp/cache/comms").expect("connect REQ/REP");
        Self { socket }
    }

    pub fn send(&self, topic: &str, data: &serde_json::Value) -> Result<serde_json::Value, zmq::Error> {
        let msg = serde_json::json!({"topic": topic, "payload": data}).to_string();
        self.socket.send_str(&msg, 0)?;
        let reply = self.socket.recv_string(0)?;
        match reply {
            Ok(s) => serde_json::from_str(&s).map_err(|_| zmq::Error::EPROTO),
            Err(_) => Err(zmq::Error::EPROTO),
        }
    }
}
