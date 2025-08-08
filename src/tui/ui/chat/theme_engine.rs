//! Theme engine for the chat interface
//! 
//! Provides customizable color schemes and styling options for all chat components

use ratatui::style::{Color, Modifier, Style};
use std::collections::HashMap;

/// Chat theme configuration
#[derive(Debug, Clone)]
pub struct ChatTheme {
    /// Theme name
    pub name: String,
    
    /// Theme variant
    pub variant: ThemeVariant,
    
    /// Color palette
    pub colors: ThemeColors,
    
    /// Text styles
    pub text_styles: TextStyles,
    
    /// Component styles
    pub component_styles: ComponentStyles,
    
    /// Animation settings
    pub animations: AnimationSettings,
}

/// Theme variant
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThemeVariant {
    Light,
    Dark,
    HighContrast,
    Colorful,
}

/// Theme colors
#[derive(Debug, Clone)]
pub struct ThemeColors {
    // Background colors
    pub background: Color,
    pub surface: Color,
    pub overlay: Color,
    
    // Foreground colors
    pub foreground: Color,
    pub foreground_dim: Color,
    pub foreground_bright: Color,
    
    // Accent colors
    pub primary: Color,
    pub secondary: Color,
    pub tertiary: Color,
    
    // Semantic colors
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    
    // Message author colors
    pub user_author: Color,
    pub assistant_author: Color,
    pub system_author: Color,
    pub default_author: Color,
    
    // Border colors
    pub border: Color,
    pub border_focused: Color,
    pub border_dim: Color,
    
    // Text colors
    pub text: Color,
    pub code: Color,
    
    // Syntax highlighting base
    pub code_background: Color,
    pub code_foreground: Color,
}

/// Text styles
#[derive(Debug, Clone)]
pub struct TextStyles {
    pub normal: Style,
    pub bold: Style,
    pub italic: Style,
    pub underline: Style,
    pub dim: Style,
    pub highlight: Style,
    pub error: Style,
    pub success: Style,
    pub warning: Style,
    pub info: Style,
    pub code: Style,
    pub quote: Style,
    pub heading1: Style,
    pub heading2: Style,
    pub heading3: Style,
}

/// Component-specific styles
#[derive(Debug, Clone)]
pub struct ComponentStyles {
    pub chat_message: MessageStyles,
    pub input_box: InputStyles,
    pub status_bar: StatusBarStyles,
    pub panel: PanelStyles,
    pub menu: MenuStyles,
    pub button: ButtonStyles,
    pub progress: ProgressStyles,
}

/// Message-specific styles
#[derive(Debug, Clone)]
pub struct MessageStyles {
    pub container: Style,
    pub author: Style,
    pub timestamp: Style,
    pub content: Style,
    pub selected: Style,
    pub streaming: Style,
}

/// Input box styles
#[derive(Debug, Clone)]
pub struct InputStyles {
    pub normal: Style,
    pub focused: Style,
    pub error: Style,
    pub placeholder: Style,
    pub suggestion: Style,
}

/// Status bar styles
#[derive(Debug, Clone)]
pub struct StatusBarStyles {
    pub background: Style,
    pub text: Style,
    pub indicator_active: Style,
    pub indicator_inactive: Style,
}

/// Panel styles
#[derive(Debug, Clone)]
pub struct PanelStyles {
    pub border: Style,
    pub border_focused: Style,
    pub title: Style,
    pub background: Style,
}

/// Menu styles
#[derive(Debug, Clone)]
pub struct MenuStyles {
    pub item: Style,
    pub item_selected: Style,
    pub item_disabled: Style,
    pub separator: Style,
}

/// Button styles
#[derive(Debug, Clone)]
pub struct ButtonStyles {
    pub normal: Style,
    pub focused: Style,
    pub pressed: Style,
    pub disabled: Style,
}

/// Progress indicator styles
#[derive(Debug, Clone)]
pub struct ProgressStyles {
    pub bar_filled: Style,
    pub bar_empty: Style,
    pub text: Style,
    pub spinner: Style,
}

/// Animation settings
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    pub enabled: bool,
    pub duration_ms: u64,
    pub easing: String,
    pub fps: u8,
}

impl Default for ChatTheme {
    fn default() -> Self {
        Self::dark_theme()
    }
}

impl ChatTheme {
    /// Create a dark theme
    pub fn dark_theme() -> Self {
        Self {
            name: "Dark".to_string(),
            variant: ThemeVariant::Dark,
            colors: ThemeColors {
                background: Color::Rgb(16, 16, 16),
                surface: Color::Rgb(24, 24, 24),
                overlay: Color::Rgb(32, 32, 32),
                foreground: Color::Rgb(220, 220, 220),
                foreground_dim: Color::Rgb(160, 160, 160),
                foreground_bright: Color::Rgb(255, 255, 255),
                primary: Color::Rgb(0, 188, 212),
                secondary: Color::Rgb(255, 64, 129),
                tertiary: Color::Rgb(255, 193, 7),
                success: Color::Rgb(76, 175, 80),
                warning: Color::Rgb(255, 152, 0),
                error: Color::Rgb(244, 67, 54),
                info: Color::Rgb(33, 150, 243),
                user_author: Color::Rgb(100, 181, 246),
                assistant_author: Color::Rgb(129, 199, 132),
                system_author: Color::Rgb(255, 183, 77),
                default_author: Color::Rgb(189, 189, 189),
                border: Color::Rgb(64, 64, 64),
                border_focused: Color::Rgb(0, 188, 212),
                border_dim: Color::Rgb(48, 48, 48),
                text: Color::Rgb(220, 220, 220),
                code: Color::Rgb(0, 188, 212),
                code_background: Color::Rgb(40, 40, 40),
                code_foreground: Color::Rgb(200, 200, 200),
            },
            text_styles: Self::create_text_styles_dark(),
            component_styles: Self::create_component_styles_dark(),
            animations: AnimationSettings {
                enabled: true,
                duration_ms: 200,
                easing: "ease-in-out".to_string(),
                fps: 60,
            },
        }
    }
    
    /// Create a light theme
    pub fn light_theme() -> Self {
        Self {
            name: "Light".to_string(),
            variant: ThemeVariant::Light,
            colors: ThemeColors {
                background: Color::Rgb(250, 250, 250),
                surface: Color::Rgb(255, 255, 255),
                overlay: Color::Rgb(245, 245, 245),
                foreground: Color::Rgb(33, 33, 33),
                foreground_dim: Color::Rgb(117, 117, 117),
                foreground_bright: Color::Rgb(0, 0, 0),
                primary: Color::Rgb(25, 118, 210),
                secondary: Color::Rgb(194, 24, 91),
                tertiary: Color::Rgb(251, 140, 0),
                success: Color::Rgb(56, 142, 60),
                warning: Color::Rgb(245, 124, 0),
                error: Color::Rgb(211, 47, 47),
                info: Color::Rgb(30, 136, 229),
                user_author: Color::Rgb(25, 118, 210),
                assistant_author: Color::Rgb(56, 142, 60),
                system_author: Color::Rgb(245, 124, 0),
                default_author: Color::Rgb(97, 97, 97),
                border: Color::Rgb(224, 224, 224),
                border_focused: Color::Rgb(25, 118, 210),
                border_dim: Color::Rgb(238, 238, 238),
                text: Color::Rgb(33, 33, 33),
                code: Color::Rgb(25, 118, 210),
                code_background: Color::Rgb(245, 245, 245),
                code_foreground: Color::Rgb(55, 55, 55),
            },
            text_styles: Self::create_text_styles_light(),
            component_styles: Self::create_component_styles_light(),
            animations: AnimationSettings {
                enabled: true,
                duration_ms: 200,
                easing: "ease-in-out".to_string(),
                fps: 60,
            },
        }
    }
    
    /// Create a high contrast theme
    pub fn high_contrast_theme() -> Self {
        Self {
            name: "High Contrast".to_string(),
            variant: ThemeVariant::HighContrast,
            colors: ThemeColors {
                background: Color::Black,
                surface: Color::Rgb(16, 16, 16),
                overlay: Color::Rgb(32, 32, 32),
                foreground: Color::White,
                foreground_dim: Color::Rgb(200, 200, 200),
                foreground_bright: Color::White,
                primary: Color::Cyan,
                secondary: Color::Magenta,
                tertiary: Color::Yellow,
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                info: Color::Blue,
                user_author: Color::Cyan,
                assistant_author: Color::Green,
                system_author: Color::Yellow,
                default_author: Color::White,
                border: Color::White,
                border_focused: Color::Cyan,
                border_dim: Color::Gray,
                text: Color::White,
                code: Color::Cyan,
                code_background: Color::Rgb(16, 16, 16),
                code_foreground: Color::White,
            },
            text_styles: Self::create_text_styles_high_contrast(),
            component_styles: Self::create_component_styles_high_contrast(),
            animations: AnimationSettings {
                enabled: false, // Reduce motion for accessibility
                duration_ms: 0,
                easing: "none".to_string(),
                fps: 30,
            },
        }
    }
    
    /// Create text styles for dark theme
    fn create_text_styles_dark() -> TextStyles {
        TextStyles {
            normal: Style::default().fg(Color::Rgb(220, 220, 220)),
            bold: Style::default().fg(Color::Rgb(220, 220, 220)).add_modifier(Modifier::BOLD),
            italic: Style::default().fg(Color::Rgb(220, 220, 220)).add_modifier(Modifier::ITALIC),
            underline: Style::default().fg(Color::Rgb(220, 220, 220)).add_modifier(Modifier::UNDERLINED),
            dim: Style::default().fg(Color::Rgb(160, 160, 160)),
            highlight: Style::default().bg(Color::Rgb(64, 64, 64)),
            error: Style::default().fg(Color::Rgb(244, 67, 54)),
            success: Style::default().fg(Color::Rgb(76, 175, 80)),
            warning: Style::default().fg(Color::Rgb(255, 152, 0)),
            info: Style::default().fg(Color::Rgb(33, 150, 243)),
            code: Style::default().fg(Color::Rgb(200, 200, 200)).bg(Color::Rgb(40, 40, 40)),
            quote: Style::default().fg(Color::Rgb(160, 160, 160)).add_modifier(Modifier::ITALIC),
            heading1: Style::default().fg(Color::Rgb(0, 188, 212)).add_modifier(Modifier::BOLD),
            heading2: Style::default().fg(Color::Rgb(100, 181, 246)).add_modifier(Modifier::BOLD),
            heading3: Style::default().fg(Color::Rgb(129, 199, 132)).add_modifier(Modifier::BOLD),
        }
    }
    
    /// Create text styles for light theme
    fn create_text_styles_light() -> TextStyles {
        TextStyles {
            normal: Style::default().fg(Color::Rgb(33, 33, 33)),
            bold: Style::default().fg(Color::Rgb(33, 33, 33)).add_modifier(Modifier::BOLD),
            italic: Style::default().fg(Color::Rgb(33, 33, 33)).add_modifier(Modifier::ITALIC),
            underline: Style::default().fg(Color::Rgb(33, 33, 33)).add_modifier(Modifier::UNDERLINED),
            dim: Style::default().fg(Color::Rgb(117, 117, 117)),
            highlight: Style::default().bg(Color::Rgb(255, 235, 59)),
            error: Style::default().fg(Color::Rgb(211, 47, 47)),
            success: Style::default().fg(Color::Rgb(56, 142, 60)),
            warning: Style::default().fg(Color::Rgb(245, 124, 0)),
            info: Style::default().fg(Color::Rgb(30, 136, 229)),
            code: Style::default().fg(Color::Rgb(55, 55, 55)).bg(Color::Rgb(245, 245, 245)),
            quote: Style::default().fg(Color::Rgb(97, 97, 97)).add_modifier(Modifier::ITALIC),
            heading1: Style::default().fg(Color::Rgb(25, 118, 210)).add_modifier(Modifier::BOLD),
            heading2: Style::default().fg(Color::Rgb(30, 136, 229)).add_modifier(Modifier::BOLD),
            heading3: Style::default().fg(Color::Rgb(56, 142, 60)).add_modifier(Modifier::BOLD),
        }
    }
    
    /// Create text styles for high contrast theme
    fn create_text_styles_high_contrast() -> TextStyles {
        TextStyles {
            normal: Style::default().fg(Color::White),
            bold: Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            italic: Style::default().fg(Color::White).add_modifier(Modifier::ITALIC),
            underline: Style::default().fg(Color::White).add_modifier(Modifier::UNDERLINED),
            dim: Style::default().fg(Color::Gray),
            highlight: Style::default().fg(Color::Black).bg(Color::White),
            error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            success: Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            warning: Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            info: Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            code: Style::default().fg(Color::White).bg(Color::Rgb(16, 16, 16)),
            quote: Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
            heading1: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            heading2: Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            heading3: Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        }
    }
    
    /// Create component styles for dark theme
    fn create_component_styles_dark() -> ComponentStyles {
        ComponentStyles {
            chat_message: MessageStyles {
                container: Style::default().bg(Color::Rgb(24, 24, 24)),
                author: Style::default().add_modifier(Modifier::BOLD),
                timestamp: Style::default().fg(Color::Rgb(160, 160, 160)),
                content: Style::default(),
                selected: Style::default().bg(Color::Rgb(48, 48, 48)),
                streaming: Style::default().add_modifier(Modifier::DIM),
            },
            input_box: InputStyles {
                normal: Style::default().bg(Color::Rgb(32, 32, 32)),
                focused: Style::default().bg(Color::Rgb(40, 40, 40)).fg(Color::Rgb(0, 188, 212)),
                error: Style::default().bg(Color::Rgb(40, 16, 16)).fg(Color::Rgb(244, 67, 54)),
                placeholder: Style::default().fg(Color::Rgb(100, 100, 100)),
                suggestion: Style::default().fg(Color::Rgb(120, 120, 120)),
            },
            status_bar: StatusBarStyles {
                background: Style::default().bg(Color::Rgb(24, 24, 24)),
                text: Style::default().fg(Color::Rgb(200, 200, 200)),
                indicator_active: Style::default().fg(Color::Green),
                indicator_inactive: Style::default().fg(Color::Gray),
            },
            panel: PanelStyles {
                border: Style::default().fg(Color::Rgb(64, 64, 64)),
                border_focused: Style::default().fg(Color::Rgb(0, 188, 212)),
                title: Style::default().fg(Color::Rgb(200, 200, 200)),
                background: Style::default().bg(Color::Rgb(16, 16, 16)),
            },
            menu: MenuStyles {
                item: Style::default(),
                item_selected: Style::default().bg(Color::Rgb(48, 48, 48)),
                item_disabled: Style::default().fg(Color::Rgb(100, 100, 100)),
                separator: Style::default().fg(Color::Rgb(64, 64, 64)),
            },
            button: ButtonStyles {
                normal: Style::default().bg(Color::Rgb(48, 48, 48)),
                focused: Style::default().bg(Color::Rgb(64, 64, 64)),
                pressed: Style::default().bg(Color::Rgb(32, 32, 32)),
                disabled: Style::default().fg(Color::Rgb(100, 100, 100)),
            },
            progress: ProgressStyles {
                bar_filled: Style::default().fg(Color::Rgb(0, 188, 212)),
                bar_empty: Style::default().fg(Color::Rgb(64, 64, 64)),
                text: Style::default(),
                spinner: Style::default().fg(Color::Rgb(0, 188, 212)),
            },
        }
    }
    
    /// Create component styles for light theme
    fn create_component_styles_light() -> ComponentStyles {
        ComponentStyles {
            chat_message: MessageStyles {
                container: Style::default().bg(Color::Rgb(255, 255, 255)),
                author: Style::default().add_modifier(Modifier::BOLD),
                timestamp: Style::default().fg(Color::Rgb(117, 117, 117)),
                content: Style::default(),
                selected: Style::default().bg(Color::Rgb(245, 245, 245)),
                streaming: Style::default().add_modifier(Modifier::DIM),
            },
            input_box: InputStyles {
                normal: Style::default().bg(Color::Rgb(245, 245, 245)),
                focused: Style::default().bg(Color::Rgb(250, 250, 250)).fg(Color::Rgb(25, 118, 210)),
                error: Style::default().bg(Color::Rgb(255, 235, 238)).fg(Color::Rgb(211, 47, 47)),
                placeholder: Style::default().fg(Color::Rgb(158, 158, 158)),
                suggestion: Style::default().fg(Color::Rgb(117, 117, 117)),
            },
            status_bar: StatusBarStyles {
                background: Style::default().bg(Color::Rgb(245, 245, 245)),
                text: Style::default().fg(Color::Rgb(66, 66, 66)),
                indicator_active: Style::default().fg(Color::Rgb(56, 142, 60)),
                indicator_inactive: Style::default().fg(Color::Rgb(189, 189, 189)),
            },
            panel: PanelStyles {
                border: Style::default().fg(Color::Rgb(224, 224, 224)),
                border_focused: Style::default().fg(Color::Rgb(25, 118, 210)),
                title: Style::default().fg(Color::Rgb(66, 66, 66)),
                background: Style::default().bg(Color::Rgb(250, 250, 250)),
            },
            menu: MenuStyles {
                item: Style::default(),
                item_selected: Style::default().bg(Color::Rgb(245, 245, 245)),
                item_disabled: Style::default().fg(Color::Rgb(189, 189, 189)),
                separator: Style::default().fg(Color::Rgb(224, 224, 224)),
            },
            button: ButtonStyles {
                normal: Style::default().bg(Color::Rgb(245, 245, 245)),
                focused: Style::default().bg(Color::Rgb(238, 238, 238)),
                pressed: Style::default().bg(Color::Rgb(224, 224, 224)),
                disabled: Style::default().fg(Color::Rgb(189, 189, 189)),
            },
            progress: ProgressStyles {
                bar_filled: Style::default().fg(Color::Rgb(25, 118, 210)),
                bar_empty: Style::default().fg(Color::Rgb(224, 224, 224)),
                text: Style::default(),
                spinner: Style::default().fg(Color::Rgb(25, 118, 210)),
            },
        }
    }
    
    /// Create component styles for high contrast theme
    fn create_component_styles_high_contrast() -> ComponentStyles {
        ComponentStyles {
            chat_message: MessageStyles {
                container: Style::default(),
                author: Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                timestamp: Style::default().fg(Color::Gray),
                content: Style::default(),
                selected: Style::default().add_modifier(Modifier::REVERSED),
                streaming: Style::default().add_modifier(Modifier::DIM),
            },
            input_box: InputStyles {
                normal: Style::default().fg(Color::White),
                focused: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                placeholder: Style::default().fg(Color::Gray),
                suggestion: Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
            },
            status_bar: StatusBarStyles {
                background: Style::default().add_modifier(Modifier::REVERSED),
                text: Style::default(),
                indicator_active: Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                indicator_inactive: Style::default().fg(Color::Gray),
            },
            panel: PanelStyles {
                border: Style::default().fg(Color::White),
                border_focused: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                title: Style::default().add_modifier(Modifier::BOLD),
                background: Style::default(),
            },
            menu: MenuStyles {
                item: Style::default(),
                item_selected: Style::default().add_modifier(Modifier::REVERSED),
                item_disabled: Style::default().fg(Color::Gray),
                separator: Style::default().fg(Color::Gray),
            },
            button: ButtonStyles {
                normal: Style::default(),
                focused: Style::default().add_modifier(Modifier::BOLD),
                pressed: Style::default().add_modifier(Modifier::REVERSED),
                disabled: Style::default().fg(Color::Gray),
            },
            progress: ProgressStyles {
                bar_filled: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                bar_empty: Style::default().fg(Color::Gray),
                text: Style::default(),
                spinner: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            },
        }
    }
    
    /// Get style for a specific token type (for syntax highlighting)
    pub fn get_token_style(&self, token_type: &str) -> Style {
        match token_type {
            "keyword" => self.text_styles.bold.fg(Color::Magenta),
            "string" => Style::default().fg(Color::Green),
            "number" => Style::default().fg(Color::Cyan),
            "comment" => self.text_styles.dim.add_modifier(Modifier::ITALIC),
            "function" => Style::default().fg(Color::Yellow),
            "type" => Style::default().fg(Color::Blue),
            _ => self.text_styles.normal,
        }
    }
}

/// Theme manager for loading and saving themes
pub struct ThemeManager {
    themes: HashMap<String, ChatTheme>,
    current_theme: String,
}

impl ThemeManager {
    pub fn new() -> Self {
        let mut themes = HashMap::new();
        
        // Add default themes
        themes.insert("dark".to_string(), ChatTheme::dark_theme());
        themes.insert("light".to_string(), ChatTheme::light_theme());
        themes.insert("high_contrast".to_string(), ChatTheme::high_contrast_theme());
        
        Self {
            themes,
            current_theme: "dark".to_string(),
        }
    }
    
    /// Get current theme
    pub fn current(&self) -> &ChatTheme {
        self.themes.get(&self.current_theme)
            .unwrap_or_else(|| self.themes.get("dark").unwrap())
    }
    
    /// Switch theme
    pub fn switch_theme(&mut self, name: &str) -> Result<(), String> {
        if self.themes.contains_key(name) {
            self.current_theme = name.to_string();
            Ok(())
        } else {
            Err(format!("Theme '{}' not found", name))
        }
    }
    
    /// Add custom theme
    pub fn add_theme(&mut self, theme: ChatTheme) {
        self.themes.insert(theme.name.clone(), theme);
    }
    
    /// List available themes
    pub fn list_themes(&self) -> Vec<&str> {
        self.themes.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_theme_creation() {
        let dark = ChatTheme::dark_theme();
        assert_eq!(dark.variant, ThemeVariant::Dark);
        
        let light = ChatTheme::light_theme();
        assert_eq!(light.variant, ThemeVariant::Light);
        
        let high_contrast = ChatTheme::high_contrast_theme();
        assert_eq!(high_contrast.variant, ThemeVariant::HighContrast);
    }
    
    #[test]
    fn test_theme_manager() {
        let mut manager = ThemeManager::new();
        
        assert!(manager.list_themes().contains(&"dark"));
        assert!(manager.list_themes().contains(&"light"));
        
        assert!(manager.switch_theme("light").is_ok());
        assert_eq!(manager.current().variant, ThemeVariant::Light);
        
        assert!(manager.switch_theme("invalid").is_err());
    }
}