//! Thread branching logic

use super::MessageThread;

/// Manages thread branching
#[derive(Debug)]
pub struct ThreadBranch {
    /// Main thread
    pub main: MessageThread,
    
    /// Branch threads
    pub branches: Vec<MessageThread>,
}

impl ThreadBranch {
    /// Create a new branch manager
    pub fn new(main: MessageThread) -> Self {
        Self {
            main,
            branches: Vec::new(),
        }
    }
    
    /// Create a branch at a specific point
    pub fn create_branch(&mut self, branch_point: usize, name: String) -> String {
        let branch = self.main.branch_at(branch_point, name);
        let branch_id = branch.id.clone();
        self.branches.push(branch);
        branch_id
    }
    
    /// Get all branch IDs
    pub fn branch_ids(&self) -> Vec<String> {
        self.branches.iter().map(|b| b.id.clone()).collect()
    }
}