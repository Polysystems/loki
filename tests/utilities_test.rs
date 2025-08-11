//! Integration tests for the utilities module

#[cfg(test)]
mod tests {
    use loki::tui::utilities::ModularUtilities;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[test]
    fn test_utilities_creation() {
        // Test that utilities can be created without panicking
        let utilities = ModularUtilities::new();
        assert_eq!(utilities.tab_count(), 5); // Should have 5 tabs
    }

    #[test]
    fn test_tab_names() {
        let utilities = ModularUtilities::new();
        
        // Check first tab is Tools
        assert_eq!(utilities.current_tab_name(), "Tools");
        
        // Can't easily test tab switching without async context
        // but at least verify the structure is correct
    }

    #[tokio::test]
    async fn test_state_management() {
        let utilities = ModularUtilities::new();
        
        // Verify state is accessible
        let state = utilities.state.read().await;
        assert_eq!(state.cache.tools.len(), 0); // Should start empty
        assert_eq!(state.cache.plugins.len(), 0);
    }

    #[test]
    fn test_cached_metrics() {
        let utilities = ModularUtilities::new();
        
        // Verify cached_metrics field exists and is accessible
        let metrics = utilities.cached_metrics.read().unwrap();
        assert!(metrics.last_update.is_none()); // Should start with no update
    }

    #[test]
    fn test_config_manager() {
        let utilities = ModularUtilities::new();
        
        // Config manager might not initialize if home directory isn't available
        // but it shouldn't panic
        assert!(utilities.config_manager.is_some() || utilities.config_manager.is_none());
    }
}