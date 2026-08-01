use zmq::Context;

/// REQ socket for `InterProcessRequestor` — mirrors `inter_process.py::InterProcessRequestor`.
pub struct ReqClient {
    socket: zmq::Socket,
}

impl ReqClient {
    pub fn connect() -> Self {
        let ctx = Context::new();
        let socket = ctx.socket(zmq::REQ).expect("create REQ");
        socket.connect("ipc:///tmp/cache/comms").expect("connect REQ/REP");
        Self { socket }
    }

    pub fn send(&self, topic: &str, data: &serde_json::Value) -> Result<serde_json::Value, zmq::Error> {
        let msg = serde_json::json!({"topic": topic, "data": data}).to_string();
        self.socket.send_str(&msg, 0)?;
        let reply = self.socket.recv_string(0)?;
        match reply {
            Ok(s) => serde_json::from_str(&s).map_err(|_| zmq::Error::EPROTO),
            Err(_) => Err(zmq::Error::EPROTO),
        }
    }
}


