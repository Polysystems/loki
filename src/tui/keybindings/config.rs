//! Configuration structures for keybindings

use std::collections::HashMap;
use std::path::PathBuf;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};
use anyhow::Result;

use super::actions::Action;
use super::registry::KeyContext;

/// A key combination (key + modifiers)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyCombo {
    pub code: SerializableKeyCode,
    pub modifiers: SerializableKeyModifiers,
}

impl KeyCombo {
    /// Create a new key combination
    pub fn new(code: KeyCode, modifiers: KeyModifiers) -> Self {
        Self {
            code: SerializableKeyCode::from(code),
            modifiers: SerializableKeyModifiers::from(modifiers),
        }
    }
    
    /// Create from a key event
    pub fn from_event(event: &KeyEvent) -> Self {
        Self::new(event.code, event.modifiers)
    }
    
    /// Convert to a human-readable string
    pub fn to_string(&self) -> String {
        let mut result = String::new();
        
        if self.modifiers.ctrl {
            result.push_str("Ctrl+");
        }
        if self.modifiers.alt {
            result.push_str("Alt+");
        }
        if self.modifiers.shift {
            result.push_str("Shift+");
        }
        if self.modifiers.meta {
            result.push_str("Meta+");
        }
        
        result.push_str(&self.code.to_string());
        result
    }
}

/// Serializable version of KeyCode
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerializableKeyCode {
    Char(char),
    F(u8),
    Enter,
    Escape,
    Backspace,
    Tab,
    BackTab,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    Up,
    Down,
    Left,
    Right,
    Space,
}

impl From<KeyCode> for SerializableKeyCode {
    fn from(code: KeyCode) -> Self {
        match code {
            KeyCode::Char(c) => SerializableKeyCode::Char(c),
            KeyCode::F(n) => SerializableKeyCode::F(n),
            KeyCode::Enter => SerializableKeyCode::Enter,
            KeyCode::Esc => SerializableKeyCode::Escape,
            KeyCode::Backspace => SerializableKeyCode::Backspace,
            KeyCode::Tab => SerializableKeyCode::Tab,
            KeyCode::BackTab => SerializableKeyCode::BackTab,
            KeyCode::Delete => SerializableKeyCode::Delete,
            KeyCode::Insert => SerializableKeyCode::Insert,
            KeyCode::Home => SerializableKeyCode::Home,
            KeyCode::End => SerializableKeyCode::End,
            KeyCode::PageUp => SerializableKeyCode::PageUp,
            KeyCode::PageDown => SerializableKeyCode::PageDown,
            KeyCode::Up => SerializableKeyCode::Up,
            KeyCode::Down => SerializableKeyCode::Down,
            KeyCode::Left => SerializableKeyCode::Left,
            KeyCode::Right => SerializableKeyCode::Right,
            _ => SerializableKeyCode::Space, // Default for unhandled keys
        }
    }
}

impl SerializableKeyCode {
    pub fn to_string(&self) -> String {
        match self {
            SerializableKeyCode::Char(c) => c.to_string(),
            SerializableKeyCode::F(n) => format!("F{}", n),
            SerializableKeyCode::Enter => "Enter".to_string(),
            SerializableKeyCode::Escape => "Esc".to_string(),
            SerializableKeyCode::Backspace => "Backspace".to_string(),
            SerializableKeyCode::Tab => "Tab".to_string(),
            SerializableKeyCode::BackTab => "BackTab".to_string(),
            SerializableKeyCode::Delete => "Delete".to_string(),
            SerializableKeyCode::Insert => "Insert".to_string(),
            SerializableKeyCode::Home => "Home".to_string(),
            SerializableKeyCode::End => "End".to_string(),
            SerializableKeyCode::PageUp => "PageUp".to_string(),
            SerializableKeyCode::PageDown => "PageDown".to_string(),
            SerializableKeyCode::Up => "↑".to_string(),
            SerializableKeyCode::Down => "↓".to_string(),
            SerializableKeyCode::Left => "←".to_string(),
            SerializableKeyCode::Right => "→".to_string(),
            SerializableKeyCode::Space => "Space".to_string(),
        }
    }
}

/// Serializable version of KeyModifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SerializableKeyModifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
}

impl From<KeyModifiers> for SerializableKeyModifiers {
    fn from(mods: KeyModifiers) -> Self {
        Self {
            ctrl: mods.contains(KeyModifiers::CONTROL),
            alt: mods.contains(KeyModifiers::ALT),
            shift: mods.contains(KeyModifiers::SHIFT),
            meta: mods.contains(KeyModifiers::META),
        }
    }
}

impl PartialEq<KeyCombo> for (&KeyCode, &KeyModifiers) {
    fn eq(&self, other: &KeyCombo) -> bool {
        SerializableKeyCode::from(*self.0) == other.code
            && SerializableKeyModifiers::from(*self.1) == other.modifiers
    }
}

/// Complete keybinding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBindingConfig {
    /// Version of the configuration format
    pub version: String,
    
    /// Custom keybinding overrides
    pub custom_overrides: HashMap<(KeyContext, KeyCombo), Action>,
    
    /// Active preset name (if any)
    pub active_preset: Option<String>,
    
    /// User-defined presets
    pub custom_presets: HashMap<String, KeyBindingPreset>,
}

impl KeyBindingConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            custom_overrides: HashMap::new(),
            active_preset: None,
            custom_presets: HashMap::new(),
        }
    }
    
    /// Get the configuration file path
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;
        Ok(config_dir.join("loki").join("keybindings.toml"))
    }
    
    /// Load configuration from file
    pub fn load_from_file() -> Result<Self> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::new());
        }
        
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self) -> Result<()> {
        let path = Self::config_path()?;
        
        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Apply a preset
    pub fn apply_preset(&mut self, preset: KeyBindingPreset) {
        self.custom_overrides = preset.bindings;
        self.active_preset = Some(preset.name);
    }
}

/// A keybinding preset (collection of bindings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBindingPreset {
    pub name: String,
    pub description: String,
    pub bindings: HashMap<(KeyContext, KeyCombo), Action>,
}

impl KeyBindingPreset {
    /// Create a vim-style preset
    pub fn vim_preset() -> Self {
        let mut bindings = HashMap::new();
        
        // Vim-style navigation
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('h'), KeyModifiers::empty())),
            Action::MoveLeft,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('j'), KeyModifiers::empty())),
            Action::MoveDown,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('k'), KeyModifiers::empty())),
            Action::MoveUp,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('l'), KeyModifiers::empty())),
            Action::MoveRight,
        );
        
        Self {
            name: "Vim".to_string(),
            description: "Vim-style navigation and commands".to_string(),
            bindings,
        }
    }
    
    /// Create an emacs-style preset
    pub fn emacs_preset() -> Self {
        let mut bindings = HashMap::new();
        
        // Emacs-style navigation
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('f'), KeyModifiers::CONTROL)),
            Action::MoveRight,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('b'), KeyModifiers::CONTROL)),
            Action::MoveLeft,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('n'), KeyModifiers::CONTROL)),
            Action::MoveDown,
        );
        bindings.insert(
            (KeyContext::Global, KeyCombo::new(KeyCode::Char('p'), KeyModifiers::CONTROL)),
            Action::MoveUp,
        );
        
        Self {
            name: "Emacs".to_string(),
            description: "Emacs-style navigation and commands".to_string(),
            bindings,
        }
    }
}