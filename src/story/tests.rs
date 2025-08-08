#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::cognitive::{CognitiveConfig, CognitiveSystem};
    use crate::config::ApiKeysConfig;
    use std::path::PathBuf;
    
    
    #[test]
    fn test_story_types() {
        // Test codebase story type
        let codebase_story = StoryType::Codebase {
            root_path: PathBuf::from("/test/path"),
            language: "Rust".to_string(),
        };
        
        match codebase_story {
            StoryType::Codebase { root_path, language } => {
                assert_eq!(root_path, PathBuf::from("/test/path"));
                assert_eq!(language, "Rust");
            }
            _ => panic!("Wrong story type"),
        }
        
        // Test agent story type
        let agent_story = StoryType::Agent {
            agent_id: "agent-001".to_string(),
            agent_type: "Assistant".to_string(),
        };
        
        match agent_story {
            StoryType::Agent { agent_id, agent_type } => {
                assert_eq!(agent_id, "agent-001");
                assert_eq!(agent_type, "Assistant");
            }
            _ => panic!("Wrong story type"),
        }
    }
    
    #[test]
    fn test_task_status_mapping() {
        let statuses = vec![
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Blocked,
            TaskStatus::Completed,
            TaskStatus::Cancelled,
        ];
        
        for status in statuses {
            match status {
                TaskStatus::Pending => assert!(true),
                TaskStatus::InProgress => assert!(true),
                TaskStatus::Blocked => assert!(true),
                TaskStatus::Completed => assert!(true),
                TaskStatus::Cancelled => assert!(true),
            }
        }
    }
    
    #[test]
    fn test_context_chain_creation() {
        let chain_id = ChainId(uuid::Uuid::new_v4());
        let story_id = StoryId::new();
        let chain = ContextChain::new(chain_id, story_id);
        
        assert_eq!(chain.id, chain_id);
        assert_eq!(chain.story_id, story_id);
    }
}