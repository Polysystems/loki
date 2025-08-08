//! Input and command handling module
//! 
//! Processes user input, commands, and natural language

pub mod input;
pub mod commands;
pub mod natural_language;
pub mod orchestration_commands;
pub mod app_integration;
pub mod edit_handler;
pub mod message_handler;

// Re-export commonly used types
pub use input::InputProcessor;
pub use commands::CommandProcessor;
pub use natural_language::{NaturalLanguageHandler, NlpCommand, NlpIntent};
pub use orchestration_commands::OrchestrationCommandHandler;
pub use app_integration::handle_chat_key_event;
pub use edit_handler::{EditHandler, EditingExt};
pub use message_handler::{MessageHandler, MessageHandlingExt, MessageTypeCount};