use std::sync::Arc;

use tokio::sync::mpsc;

/// A bounded channel replacing `multiprocessing.Queue(maxsize=N)`.
///
/// Same backpressure semantics: `send` blocks when the channel is full,
/// `recv` returns None on drop (mimics Queue empty-after-close).
pub struct Queue<T> {
    sender: Arc<mpsc::Sender<T>>,
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<T>>>,
}

impl<T> Queue<T> {
    pub fn new(maxsize: usize) -> Self {
        let (sender, receiver) = mpsc::channel(maxsize);
        Self {
            sender: Arc::new(sender),
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
        }
    }

    pub async fn send(&self, item: T) -> Result<(), SendError> {
        self.sender
            .send(item)
            .await
            .map_err(|_| SendError)
    }

    pub async fn recv(&self) -> Option<T> {
        self.receiver.lock().await.recv().await
    }

    pub fn sender_count(&self) -> usize {
        self.sender.clone().strong_count()
    }
}

impl<T> Clone for Queue<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
        }
    }
}

#[derive(Debug)]
pub struct SendError;

/// Shared-state wrapper replacing `multiprocessing.managers.DictProxy`.
///
/// `SyncManager.dict()` / `ValueProxy` counters become `Arc<RwLock<T>>` —
/// same get/set semantics, no manager process.
#[derive(Clone)]
pub struct SharedDict<K, V>
where
    K: std::hash::Hash + Eq + Send + Sync + 'static,
    V: Send + Sync,
{
    inner: Arc<std::sync::RwLock<std::collections::HashMap<K, V>>>,
}

impl<K, V> SharedDict<K, V>
where
    K: std::hash::Hash + Eq + Send + Sync + 'static,
    V: Send + Sync + Clone,
{
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    pub fn insert(&self, key: K, value: V) {
        self.inner.write().unwrap().insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.read().unwrap().get(key).cloned()
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.write().unwrap().remove(key)
    }

    pub fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        self.inner.read().unwrap().keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }
}

/// A oneshot-style event replacing `multiprocessing.Event` / `MpEvent`.
///
/// `tokio::sync::Notify` with a stored boolean — `wait()` mirrors the
/// blocking `Event.wait()` pattern; `set()` mirrors `Event.set()`.
#[derive(Clone)]
pub struct Event {
    notify: Arc<tokio::sync::Notify>,
    state: Arc<tokio::sync::RwLock<bool>>,
}

impl Event {
    pub fn new() -> Self {
        Self {
            notify: Arc::new(tokio::sync::Notify::new()),
            state: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    pub async fn is_set(&self) -> bool {
        *self.state.read().await
    }

    pub async fn set(&self) {
        let mut s = self.state.write().await;
        *s = true;
        drop(s);
        self.notify.notify_waiters();
    }

    pub async fn wait(&self) {
        loop {
            if *self.state.read().await {
                return;
            }
            self.notify.notified().await;
        }
    }

    pub async fn clear(&self) {
        let mut s = self.state.write().await;
        *s = false;
    }
}


