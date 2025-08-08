use std::collections::HashMap;
use std::sync::Arc;

use super::{StreamingConfig, StreamingManager};
use crate::models::local_manager::LocalModelManager;

#[tokio::test]
async fn test_stream_id_generation() {
    let manager =
        StreamingManager::new(Arc::new(LocalModelManager::new().await.unwrap()), HashMap::new());

    let id1 = manager.generate_stream_id();
    let id2 = manager.generate_stream_id();

    assert_ne!(id1, id2);
    assert!(id1.starts_with("stream_"));
    assert!(id2.starts_with("stream_"));
}

#[tokio::test]
async fn test_streamingconfig_defaults() {
    let config = StreamingConfig::default();

    assert_eq!(config.max_concurrent_streams, 50);
    assert_eq!(config.default_buffer_size, 1024);
    assert_eq!(config.default_timeout_ms, 30000);
    assert!(config.enable_backpressure);
}
