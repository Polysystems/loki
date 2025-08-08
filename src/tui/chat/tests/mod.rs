//! Test suite for chat system refactoring

#[cfg(test)]
pub mod integration;

#[cfg(test)]
pub mod state;

#[cfg(test)]
pub mod orchestration;

#[cfg(test)]
pub mod bridge_test;

#[cfg(test)]
pub mod integration_refactor;

#[cfg(test)]
pub mod cognitive_integration_test;

#[cfg(test)]
pub mod tool_execution_test;

#[cfg(test)]
pub mod model_registry_test;

#[cfg(test)]
pub mod nlp_enhancement_test;

#[cfg(test)]
pub mod task_priority_test;

#[cfg(test)]
pub mod settings_string_test;