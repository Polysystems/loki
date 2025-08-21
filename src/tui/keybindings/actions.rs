//! Action definitions for keybindings

use serde::{Deserialize, Serialize};

/// Categories of actions for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionCategory {
    Navigation,
    Editing,
    Commands,
    Panels,
    System,
    Custom,
}

/// All possible actions that can be triggered by keybindings
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    // Navigation actions
    NextTab,
    PrevTab,
    NextSubTab,
    PrevSubTab,
    GoToTab(usize),
    GoToView(String),
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    PageUp,
    PageDown,
    Home,
    End,
    
    // Editing actions
    Copy,
    Paste,
    Cut,
    Undo,
    Redo,
    SelectAll,
    ClearInput,
    DeleteChar,
    DeleteWord,
    DeleteLine,
    InsertNewLine,
    
    // Command actions
    ExecuteCommand,
    CommandHistoryUp,
    CommandHistoryDown,
    AutoComplete,
    CancelCommand,
    SearchForward,
    SearchBackward,
    
    // Panel control actions
    TogglePanel(String),
    ToggleContextPanel,
    ToggleToolPanel,
    ToggleWorkflowPanel,
    TogglePreviewPanel,
    ToggleStoryPanel,
    CyclePanelFocus,
    MaximizePanel,
    RestorePanels,
    
    // File operations (for editor)
    OpenFile,
    SaveFile,
    SaveFileAs,
    CloseFile,
    NewFile,
    OpenFileBrowser,
    QuickOpen,
    
    // System actions
    Quit,
    ShowHelp,
    ShowKeybindings,
    Refresh,
    SaveState,
    LoadState,
    ToggleDebugMode,
    ShowCommandPalette,
    
    // Mode changes
    EnterEditMode,
    ExitEditMode,
    EnterCommandMode,
    EnterSearchMode,
    EnterVisualMode,
    
    // Custom actions (for extensions/plugins)
    Custom(String),
}

impl Action {
    /// Get the category of this action
    pub fn category(&self) -> ActionCategory {
        match self {
            Action::NextTab | Action::PrevTab | Action::NextSubTab | Action::PrevSubTab 
            | Action::GoToTab(_) | Action::GoToView(_) 
            | Action::MoveUp | Action::MoveDown | Action::MoveLeft | Action::MoveRight
            | Action::PageUp | Action::PageDown | Action::Home | Action::End => {
                ActionCategory::Navigation
            }
            
            Action::Copy | Action::Paste | Action::Cut | Action::Undo | Action::Redo
            | Action::SelectAll | Action::ClearInput | Action::DeleteChar
            | Action::DeleteWord | Action::DeleteLine | Action::InsertNewLine => {
                ActionCategory::Editing
            }
            
            Action::ExecuteCommand | Action::CommandHistoryUp | Action::CommandHistoryDown
            | Action::AutoComplete | Action::CancelCommand 
            | Action::SearchForward | Action::SearchBackward => {
                ActionCategory::Commands
            }
            
            Action::TogglePanel(_) | Action::ToggleContextPanel | Action::ToggleToolPanel
            | Action::ToggleWorkflowPanel | Action::TogglePreviewPanel | Action::ToggleStoryPanel
            | Action::CyclePanelFocus | Action::MaximizePanel | Action::RestorePanels => {
                ActionCategory::Panels
            }
            
            Action::OpenFile | Action::SaveFile | Action::SaveFileAs | Action::CloseFile
            | Action::NewFile | Action::OpenFileBrowser | Action::QuickOpen => {
                ActionCategory::Commands
            }
            
            Action::Quit | Action::ShowHelp | Action::ShowKeybindings | Action::Refresh
            | Action::SaveState | Action::LoadState | Action::ToggleDebugMode
            | Action::ShowCommandPalette => {
                ActionCategory::System
            }
            
            Action::EnterEditMode | Action::ExitEditMode | Action::EnterCommandMode
            | Action::EnterSearchMode | Action::EnterVisualMode => {
                ActionCategory::Commands
            }
            
            Action::Custom(_) => ActionCategory::Custom,
        }
    }
    
    /// Get a human-readable description of this action
    pub fn description(&self) -> String {
        match self {
            Action::NextTab => "Switch to next tab".to_string(),
            Action::PrevTab => "Switch to previous tab".to_string(),
            Action::NextSubTab => "Switch to next subtab".to_string(),
            Action::PrevSubTab => "Switch to previous subtab".to_string(),
            Action::GoToTab(n) => format!("Go to tab {}", n),
            Action::GoToView(view) => format!("Go to {} view", view),
            Action::MoveUp => "Move cursor up".to_string(),
            Action::MoveDown => "Move cursor down".to_string(),
            Action::MoveLeft => "Move cursor left".to_string(),
            Action::MoveRight => "Move cursor right".to_string(),
            Action::PageUp => "Page up".to_string(),
            Action::PageDown => "Page down".to_string(),
            Action::Home => "Go to beginning".to_string(),
            Action::End => "Go to end".to_string(),
            
            Action::Copy => "Copy selection".to_string(),
            Action::Paste => "Paste from clipboard".to_string(),
            Action::Cut => "Cut selection".to_string(),
            Action::Undo => "Undo last action".to_string(),
            Action::Redo => "Redo last action".to_string(),
            Action::SelectAll => "Select all".to_string(),
            Action::ClearInput => "Clear input field".to_string(),
            Action::DeleteChar => "Delete character".to_string(),
            Action::DeleteWord => "Delete word".to_string(),
            Action::DeleteLine => "Delete line".to_string(),
            Action::InsertNewLine => "Insert new line".to_string(),
            
            Action::ExecuteCommand => "Execute command".to_string(),
            Action::CommandHistoryUp => "Previous command in history".to_string(),
            Action::CommandHistoryDown => "Next command in history".to_string(),
            Action::AutoComplete => "Auto-complete".to_string(),
            Action::CancelCommand => "Cancel command".to_string(),
            Action::SearchForward => "Search forward".to_string(),
            Action::SearchBackward => "Search backward".to_string(),
            
            Action::TogglePanel(name) => format!("Toggle {} panel", name),
            Action::ToggleContextPanel => "Toggle context panel".to_string(),
            Action::ToggleToolPanel => "Toggle tool panel".to_string(),
            Action::ToggleWorkflowPanel => "Toggle workflow panel".to_string(),
            Action::TogglePreviewPanel => "Toggle preview panel".to_string(),
            Action::ToggleStoryPanel => "Toggle story panel".to_string(),
            Action::CyclePanelFocus => "Cycle panel focus".to_string(),
            Action::MaximizePanel => "Maximize current panel".to_string(),
            Action::RestorePanels => "Restore panel layout".to_string(),
            
            Action::OpenFile => "Open file".to_string(),
            Action::SaveFile => "Save file".to_string(),
            Action::SaveFileAs => "Save file as".to_string(),
            Action::CloseFile => "Close file".to_string(),
            Action::NewFile => "New file".to_string(),
            Action::OpenFileBrowser => "Open file browser".to_string(),
            Action::QuickOpen => "Quick open file".to_string(),
            
            Action::Quit => "Quit application".to_string(),
            Action::ShowHelp => "Show help".to_string(),
            Action::ShowKeybindings => "Show keybindings".to_string(),
            Action::Refresh => "Refresh view".to_string(),
            Action::SaveState => "Save application state".to_string(),
            Action::LoadState => "Load application state".to_string(),
            Action::ToggleDebugMode => "Toggle debug mode".to_string(),
            Action::ShowCommandPalette => "Show command palette".to_string(),
            
            Action::EnterEditMode => "Enter edit mode".to_string(),
            Action::ExitEditMode => "Exit edit mode".to_string(),
            Action::EnterCommandMode => "Enter command mode".to_string(),
            Action::EnterSearchMode => "Enter search mode".to_string(),
            Action::EnterVisualMode => "Enter visual mode".to_string(),
            
            Action::Custom(name) => format!("Custom: {}", name),
        }
    }
    
    /// Check if this action requires special handling
    pub fn is_system_action(&self) -> bool {
        matches!(self, 
            Action::Quit | 
            Action::SaveState | 
            Action::LoadState
        )
    }
}