//! Authentication UI Components
//!
//! Provides login/logout forms and user management interfaces for the TUI.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
    Frame,
};

use crate::auth::{User, UserRole};
use crate::tui::App;

/// Authentication state for UI
#[derive(Debug, Clone)]
pub enum AuthUIState {
    /// Not authenticated, show login form
    LoginForm {
        username_input: String,
        password_input: String,
        focus: LoginFocus,
        error_message: Option<String>,
    },
    /// User authenticated, show user info
    Authenticated {
        user: User,
        show_user_menu: bool,
    },
    /// Show user registration form (admin only)
    RegisterForm {
        username_input: String,
        password_input: String,
        email_input: String,
        role_selection: UserRole,
        focus: RegisterFocus,
        error_message: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoginFocus {
    Username,
    Password,
    SubmitButton,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegisterFocus {
    Username,
    Password,
    Email,
    Role,
    SubmitButton,
    CancelButton,
}

impl Default for AuthUIState {
    fn default() -> Self {
        Self::LoginForm {
            username_input: String::new(),
            password_input: String::new(),
            focus: LoginFocus::Username,
            error_message: None,
        }
    }
}

/// Render authentication UI overlay
pub fn render_auth_overlay(f: &mut Frame, app: &App, area: Rect) {
    if app.is_authenticated() {
        render_user_status(f, app, area);
    } else {
        render_login_form(f, area);
    }
}

/// Render login form
pub fn render_login_form(f: &mut Frame, area: Rect) {
    // Create centered popup area
    let popup_area = centered_rect(60, 40, area);
    
    // Clear the area
    f.render_widget(Clear, popup_area);
    
    // Create main block
    let block = Block::default()
        .title("🔐 Authentication Required")
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));
    
    f.render_widget(block, popup_area);
    
    // Create inner layout
    let inner_area = popup_area.inner(Margin {
        vertical: 1,
        horizontal: 2,
    });
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Username
            Constraint::Length(3), // Password
            Constraint::Length(1), // Spacer
            Constraint::Length(3), // Buttons
            Constraint::Min(1),    // Help text
        ])
        .split(inner_area);
    
    // Username input
    let username_block = Block::default()
        .title("Username")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::White));
    
    let username_input = Paragraph::new("admin") // Default to admin for demo
        .block(username_block)
        .style(Style::default().bg(Color::DarkGray));
    
    f.render_widget(username_input, chunks[0]);
    
    // Password input
    let password_block = Block::default()
        .title("Password")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::White));
    
    let password_input = Paragraph::new("••••••••") // Masked password
        .block(password_block)
        .style(Style::default().bg(Color::DarkGray));
    
    f.render_widget(password_input, chunks[1]);
    
    // Buttons
    let button_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(chunks[3]);
    
    let login_button = Paragraph::new("[ Login ]")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD));
    
    let cancel_button = Paragraph::new("[ Cancel ]")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Red));
    
    f.render_widget(login_button, button_chunks[0]);
    f.render_widget(cancel_button, button_chunks[1]);
    
    // Help text
    let help_text = vec![
        Line::from(vec![
            Span::styled("↑↓", Style::default().fg(Color::Yellow)),
            Span::raw(" Navigate • "),
            Span::styled("Enter", Style::default().fg(Color::Yellow)),
            Span::raw(" Select • "),
            Span::styled("Esc", Style::default().fg(Color::Yellow)),
            Span::raw(" Cancel"),
        ]),
        Line::from(vec![
            Span::raw("Default admin user created on first run"),
        ]),
    ];
    
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray));
    
    f.render_widget(help, chunks[4]);
}

/// Render user status when authenticated
pub fn render_user_status(f: &mut Frame, app: &App, area: Rect) {
    // Only show in top-right corner
    let status_area = Rect {
        x: area.width.saturating_sub(25),
        y: 0,
        width: 25,
        height: 3,
    };
    
    if let Some(auth_session) = &app.current_auth_session {
        let user_info = vec![
            Line::from(vec![
                Span::styled("👤 ", Style::default().fg(Color::Blue)),
                Span::styled(&auth_session.user.username, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("🛡️ ", Style::default().fg(Color::Green)),
                Span::styled(format!("{:?}", auth_session.user.role), Style::default().fg(Color::Green)),
            ]),
        ];
        
        let user_widget = Paragraph::new(user_info)
            .block(Block::default().borders(Borders::ALL).style(Style::default().fg(Color::Blue)))
            .alignment(Alignment::Left);
        
        f.render_widget(user_widget, status_area);
    }
}

/// Render user management interface (admin only)
pub fn render_user_management(f: &mut Frame, app: &App, area: Rect) {
    if !app.is_authenticated() || !matches!(app.current_auth_session.as_ref().map(|s| &s.user.role), Some(UserRole::Admin)) {
        return;
    }
    
    let block = Block::default()
        .title("👥 User Management")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Blue));
    
    let inner_area = block.inner(area);
    f.render_widget(block, area);
    
    // Split into user list and actions
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(inner_area);
    
    // User list
    let user_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("admin", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled("Admin", Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("guest", Style::default().fg(Color::Gray)),
            Span::raw("  "),
            Span::styled("Guest", Style::default().fg(Color::Yellow)),
            Span::raw("  "),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ])),
    ];
    
    let user_list = List::new(user_items)
        .block(Block::default().title("Users").borders(Borders::ALL));
    
    f.render_widget(user_list, chunks[0]);
    
    // Actions
    let actions = vec![
        ListItem::new("🆕 Create User"),
        ListItem::new("✏️ Edit User"),
        ListItem::new("🗑️ Delete User"),
        ListItem::new("🔑 Reset Password"),
        ListItem::new("🚫 Disable User"),
    ];
    
    let action_list = List::new(actions)
        .block(Block::default().title("Actions").borders(Borders::ALL));
    
    f.render_widget(action_list, chunks[1]);
}

/// Render permission indicator for current user
pub fn render_permission_indicator(f: &mut Frame, app: &App, area: Rect, permission: &str) {
    if let Some(auth_session) = &app.current_auth_session {
        let has_permission = auth_session.has_permission(permission);
        let indicator_area = Rect {
            x: area.x,
            y: area.y,
            width: 3,
            height: 1,
        };
        
        let indicator = if has_permission {
            Paragraph::new("✅").style(Style::default().fg(Color::Green))
        } else {
            Paragraph::new("❌").style(Style::default().fg(Color::Red))
        };
        
        f.render_widget(indicator, indicator_area);
    }
}

/// Create a centered rectangle for popups
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Authentication UI event handler
pub enum AuthUIEvent {
    /// Handle input for login form
    LoginInput(char),
    /// Navigate in login form
    LoginNavigate(bool), // true = next, false = previous
    /// Submit login
    LoginSubmit,
    /// Show user menu
    ShowUserMenu,
    /// Logout
    Logout,
    /// Show user registration (admin only)
    ShowRegister,
    /// Cancel current operation
    Cancel,
}

impl AuthUIState {
    /// Handle authentication UI events
    pub fn handle_event(&mut self, event: AuthUIEvent) {
        match (self, event) {
            (Self::LoginForm { focus, .. }, AuthUIEvent::LoginNavigate(next)) => {
                *focus = if next {
                    match focus {
                        LoginFocus::Username => LoginFocus::Password,
                        LoginFocus::Password => LoginFocus::SubmitButton,
                        LoginFocus::SubmitButton => LoginFocus::Username,
                    }
                } else {
                    match focus {
                        LoginFocus::Username => LoginFocus::SubmitButton,
                        LoginFocus::Password => LoginFocus::Username,
                        LoginFocus::SubmitButton => LoginFocus::Password,
                    }
                };
            },
            (Self::LoginForm { username_input, focus: LoginFocus::Username, .. }, AuthUIEvent::LoginInput(ch)) => {
                if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                    username_input.push(ch);
                }
            },
            (Self::LoginForm { password_input, focus: LoginFocus::Password, .. }, AuthUIEvent::LoginInput(ch)) => {
                if ch.is_ascii() && !ch.is_control() {
                    password_input.push(ch);
                }
            },
            (Self::Authenticated { show_user_menu, .. }, AuthUIEvent::ShowUserMenu) => {
                *show_user_menu = !*show_user_menu;
            },
            _ => {}, // Other events handled by parent
        }
    }
    
    /// Reset to default login state
    pub fn reset_login(&mut self) {
        *self = Self::default();
    }
    
    /// Set error message
    pub fn set_error(&mut self, message: String) {
        match self {
            Self::LoginForm { error_message, .. } => {
                *error_message = Some(message);
            },
            Self::RegisterForm { error_message, .. } => {
                *error_message = Some(message);
            },
            _ => {},
        }
    }
    
    /// Clear error message
    pub fn clear_error(&mut self) {
        match self {
            Self::LoginForm { error_message, .. } => {
                *error_message = None;
            },
            Self::RegisterForm { error_message, .. } => {
                *error_message = None;
            },
            _ => {},
        }
    }
}