//! Central registry for keybindings with conflict detection and priority management

use std::collections::{HashMap, HashSet};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};

use super::config::{KeyCombo, KeyBindingConfig};
use super::actions::Action;

/// Result of handling a key input
#[derive(Debug, Clone, PartialEq)]
pub enum InputResult {
    /// The key was consumed and handled
    Consumed,
    /// The key should be passed through to the next handler
    PassThrough,
    /// The key triggered an action that should be propagated
    Action(Action),
}

/// Context in which a keybinding is active
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyContext {
    /// Global context - always active
    Global,
    /// Active only in a specific view
    View(String),
    /// Active only in a specific tab within a view
    Tab { view: String, tab: String },
    /// Active only in a specific subtab
    SubTab { view: String, tab: String, subtab: String },
    /// Active only when a specific mode is active (e.g., edit mode)
    Mode(String),
}

impl KeyContext {
    /// Check if this context is active given the current state
    pub fn is_active(&self, current_view: &str, current_tab: Option<&str>, current_subtab: Option<&str>) -> bool {
        match self {
            KeyContext::Global => true,
            KeyContext::View(view) => view == current_view,
            KeyContext::Tab { view, tab } => {
                view == current_view && current_tab == Some(tab.as_str())
            }
            KeyContext::SubTab { view, tab, subtab } => {
                view == current_view 
                    && current_tab == Some(tab.as_str())
                    && current_subtab == Some(subtab.as_str())
            }
            KeyContext::Mode(_mode) => {
                // Mode checking would require additional state tracking
                false
            }
        }
    }
    
    /// Get the priority of this context (higher = more specific)
    pub fn priority(&self) -> u32 {
        match self {
            KeyContext::Global => 0,
            KeyContext::View(_) => 1,
            KeyContext::Tab { .. } => 2,
            KeyContext::SubTab { .. } => 3,
            KeyContext::Mode(_) => 4,
        }
    }
}

/// Registration entry for a keybinding
#[derive(Debug, Clone)]
struct KeyBinding {
    combo: KeyCombo,
    action: Action,
    context: KeyContext,
    description: String,
    overridable: bool,
}

/// Central registry for all keybindings
pub struct KeyBindingRegistry {
    /// All registered keybindings
    bindings: Vec<KeyBinding>,
    
    /// Quick lookup by key combination
    lookup: HashMap<KeyCombo, Vec<usize>>,
    
    /// User customizations that override defaults
    custom_overrides: HashMap<(KeyContext, KeyCombo), Action>,
    
    /// Blocked key combinations (reserved by system)
    blocked_keys: HashSet<KeyCombo>,
}

impl KeyBindingRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            lookup: HashMap::new(),
            custom_overrides: HashMap::new(),
            blocked_keys: HashSet::new(),
        }
    }
    
    /// Load default keybindings
    pub fn load_defaults(&mut self) -> Result<()> {
        // Navigation
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::Char('j'), KeyModifiers::CONTROL),
            Action::NextSubTab,
            "Navigate to next subtab",
            true,
        )?;
        
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::Char('k'), KeyModifiers::CONTROL),
            Action::PrevSubTab,
            "Navigate to previous subtab",
            true,
        )?;
        
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::Tab, KeyModifiers::empty()),
            Action::NextTab,
            "Navigate to next tab",
            true,
        )?;
        
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            Action::PrevTab,
            "Navigate to previous tab",
            true,
        )?;
        
        // System
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::Char('q'), KeyModifiers::CONTROL),
            Action::Quit,
            "Quit application",
            false, // Not overridable
        )?;
        
        self.register(
            KeyContext::Global,
            KeyCombo::new(KeyCode::F(1), KeyModifiers::empty()),
            Action::ShowHelp,
            "Show help overlay",
            true,
        )?;
        
        // Editor-specific bindings
        self.register(
            KeyContext::SubTab { 
                view: "chat".to_string(), 
                tab: "chat".to_string(), 
                subtab: "editor".to_string() 
            },
            KeyCombo::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            Action::OpenFile,
            "Open file browser",
            true,
        )?;
        
        self.register(
            KeyContext::SubTab { 
                view: "chat".to_string(), 
                tab: "chat".to_string(), 
                subtab: "editor".to_string() 
            },
            KeyCombo::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
            Action::SaveFile,
            "Save current file",
            true,
        )?;
        
        Ok(())
    }
    
    /// Register a new keybinding
    pub fn register(
        &mut self,
        context: KeyContext,
        combo: KeyCombo,
        action: Action,
        description: &str,
        overridable: bool,
    ) -> Result<()> {
        // Check for conflicts in the same context
        if let Some(conflict) = self.find_conflict(&context, &combo) {
            bail!(
                "Key combination {:?} already registered for action {:?} in context {:?}",
                combo, conflict.action, context
            );
        }
        
        // Check if key is blocked
        if self.blocked_keys.contains(&combo) {
            bail!("Key combination {:?} is reserved by the system", combo);
        }
        
        let binding = KeyBinding {
            combo: combo.clone(),
            action,
            context,
            description: description.to_string(),
            overridable,
        };
        
        let index = self.bindings.len();
        self.bindings.push(binding);
        
        // Update lookup table
        self.lookup.entry(combo).or_insert_with(Vec::new).push(index);
        
        Ok(())
    }
    
    /// Find a conflicting binding in the same context
    fn find_conflict(&self, context: &KeyContext, combo: &KeyCombo) -> Option<&KeyBinding> {
        if let Some(indices) = self.lookup.get(combo) {
            for &idx in indices {
                if let Some(binding) = self.bindings.get(idx) {
                    if &binding.context == context {
                        return Some(binding);
                    }
                }
            }
        }
        None
    }
    
    /// Apply a configuration, including custom overrides
    pub fn apply_config(&mut self, config: KeyBindingConfig) -> Result<()> {
        // Apply custom overrides
        for ((context, combo), action) in config.custom_overrides {
            self.override_binding(context, combo, action)?;
        }
        Ok(())
    }
    
    /// Override a keybinding (user customization)
    pub fn override_binding(
        &mut self,
        context: KeyContext,
        combo: KeyCombo,
        action: Action,
    ) -> Result<()> {
        // Check if the binding exists and is overridable
        if let Some(indices) = self.lookup.get(&combo) {
            for &idx in indices {
                if let Some(binding) = self.bindings.get(idx) {
                    if binding.context == context && !binding.overridable {
                        bail!("Cannot override non-overridable keybinding");
                    }
                }
            }
        }
        
        self.custom_overrides.insert((context, combo), action);
        Ok(())
    }
    
    /// Lookup what action a key combination triggers in the current context
    pub fn lookup_action(
        &self,
        key_event: &KeyEvent,
        current_view: &str,
        current_tab: Option<&str>,
        current_subtab: Option<&str>,
    ) -> Option<Action> {
        let combo = KeyCombo::from_event(key_event);
        
        // Check custom overrides first
        let mut best_match: Option<(&KeyContext, &Action)> = None;
        let mut best_priority = 0;
        
        for ((context, key_combo), action) in &self.custom_overrides {
            if key_combo == &combo && context.is_active(current_view, current_tab, current_subtab) {
                let priority = context.priority();
                if priority >= best_priority {
                    best_priority = priority;
                    best_match = Some((context, action));
                }
            }
        }
        
        if let Some((_, action)) = best_match {
            return Some(action.clone());
        }
        
        // Check registered bindings
        if let Some(indices) = self.lookup.get(&combo) {
            let mut best_binding: Option<&KeyBinding> = None;
            let mut best_priority = 0;
            
            for &idx in indices {
                if let Some(binding) = self.bindings.get(idx) {
                    if binding.context.is_active(current_view, current_tab, current_subtab) {
                        let priority = binding.context.priority();
                        if priority >= best_priority {
                            best_priority = priority;
                            best_binding = Some(binding);
                        }
                    }
                }
            }
            
            if let Some(binding) = best_binding {
                return Some(binding.action.clone());
            }
        }
        
        None
    }
    
    /// Get all keybindings for a specific context
    pub fn get_context_bindings(&self, context: &KeyContext) -> Vec<(KeyCombo, Action, String)> {
        let mut result = Vec::new();
        
        for binding in &self.bindings {
            if &binding.context == context {
                // Check if overridden
                let action = if let Some(override_action) = 
                    self.custom_overrides.get(&(context.clone(), binding.combo.clone())) {
                    override_action.clone()
                } else {
                    binding.action.clone()
                };
                
                result.push((
                    binding.combo.clone(),
                    action,
                    binding.description.clone(),
                ));
            }
        }
        
        result
    }
    
    /// Block a key combination from being registered (system reserved)
    pub fn block_key(&mut self, combo: KeyCombo) {
        self.blocked_keys.insert(combo);
    }
    
    /// Check for conflicts across all contexts
    pub fn find_all_conflicts(&self) -> Vec<(KeyCombo, Vec<(KeyContext, Action)>)> {
        let mut conflicts = HashMap::new();
        
        for binding in &self.bindings {
            conflicts
                .entry(binding.combo.clone())
                .or_insert_with(Vec::new)
                .push((binding.context.clone(), binding.action.clone()));
        }
        
        conflicts
            .into_iter()
            .filter(|(_, contexts)| contexts.len() > 1)
            .collect()
    }
}