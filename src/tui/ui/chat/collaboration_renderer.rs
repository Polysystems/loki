//! Collaboration renderer for visual representation of real-time collaboration
//! 
//! Handles rendering of user cursors, selections, typing indicators, and participant lists.

use ratatui::{
    layout::{Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
    Frame,
};
use std::collections::HashMap;

use super::collaboration::{
    UserPresence, UserStatus, CollaborationSession, CursorPosition,
    CollaborationUI, ParticipantListPosition,
};

/// Collaboration renderer
#[derive(Clone)]
pub struct CollaborationRenderer {
    /// UI configuration
    ui_config: CollaborationUI,
    
    /// Cursor animation state
    cursor_blink_state: HashMap<String, bool>,
    
    /// Last cursor update time
    last_cursor_update: std::time::Instant,
    
    /// Typing indicator animation frame
    typing_animation_frame: usize,
}

impl CollaborationRenderer {
    pub fn new() -> Self {
        Self {
            ui_config: CollaborationUI::default(),
            cursor_blink_state: HashMap::new(),
            last_cursor_update: std::time::Instant::now(),
            typing_animation_frame: 0,
        }
    }
    
    /// Update animation states
    pub fn tick(&mut self) {
        // Update cursor blink
        if self.last_cursor_update.elapsed() > std::time::Duration::from_millis(500) {
            for blink in self.cursor_blink_state.values_mut() {
                *blink = !*blink;
            }
            self.last_cursor_update = std::time::Instant::now();
        }
        
        // Update typing animation
        self.typing_animation_frame = (self.typing_animation_frame + 1) % 4;
    }
    
    /// Render collaboration overlay
    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        session: &CollaborationSession,
        current_user_id: &str,
    ) {
        // Render participant list if enabled
        if self.ui_config.show_participants {
            self.render_participant_list(frame, area, session, current_user_id);
        }
        
        // Render typing indicators
        if self.ui_config.show_typing_indicators {
            self.render_typing_indicators(frame, area, session);
        }
    }
    
    /// Render user cursors in chat area
    pub fn render_cursors(
        &mut self,
        frame: &mut Frame,
        chat_area: Rect,
        participants: &[UserPresence],
        current_user_id: &str,
    ) {
        if !self.ui_config.show_cursors {
            return;
        }
        
        for participant in participants {
            if participant.user_id == current_user_id {
                continue; // Don't render own cursor
            }
            
            if let Some(cursor_pos) = &participant.cursor_position {
                self.render_user_cursor(frame, chat_area, cursor_pos, &participant);
            }
        }
    }
    
    /// Render participant list
    fn render_participant_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        session: &CollaborationSession,
        current_user_id: &str,
    ) {
        let participant_area = match self.ui_config.participants_position {
            ParticipantListPosition::Top => {
                Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: 3.min(area.height),
                }
            }
            ParticipantListPosition::Right => {
                Rect {
                    x: area.x + area.width.saturating_sub(30),
                    y: area.y,
                    width: 30.min(area.width),
                    height: area.height,
                }
            }
            ParticipantListPosition::Bottom => {
                Rect {
                    x: area.x,
                    y: area.y + area.height.saturating_sub(3),
                    width: area.width,
                    height: 3.min(area.height),
                }
            }
            ParticipantListPosition::Floating => {
                // Floating in top-right corner
                Rect {
                    x: area.x + area.width.saturating_sub(25),
                    y: area.y + 1,
                    width: 24.min(area.width - 2),
                    height: (session.participants.len() as u16 + 2).min(10),
                }
            }
        };
        
        // Clear area for floating mode
        if self.ui_config.participants_position == ParticipantListPosition::Floating {
            frame.render_widget(Clear, participant_area);
        }
        
        let mut participants: Vec<_> = session.participants.values().collect();
        participants.sort_by_key(|p| &p.username);
        
        let items: Vec<ListItem> = participants
            .iter()
            .map(|p| {
                let is_you = p.user_id == current_user_id;
                
                let mut spans = vec![
                    // User color indicator
                    Span::styled(
                        "● ",
                        Style::default().fg(self.parse_color(&p.color)),
                    ),
                    // Username
                    Span::styled(
                        if is_you {
                            format!("{} (You)", p.username)
                        } else {
                            p.username.clone()
                        },
                        if is_you {
                            Style::default().add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        },
                    ),
                ];
                
                // Status indicator
                let status_icon = match p.status {
                    UserStatus::Active => "✓",
                    UserStatus::Idle => "◐",
                    UserStatus::Away => "◯",
                    UserStatus::Offline => "✗",
                };
                
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    status_icon,
                    Style::default().fg(match p.status {
                        UserStatus::Active => Color::Green,
                        UserStatus::Idle => Color::Yellow,
                        UserStatus::Away => Color::DarkGray,
                        UserStatus::Offline => Color::Red,
                    }),
                ));
                
                // Typing indicator
                if p.is_typing {
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        self.get_typing_animation(),
                        Style::default().fg(Color::Cyan),
                    ));
                }
                
                ListItem::new(Line::from(spans))
            })
            .collect();
        
        let title = format!("Participants ({})", session.participants.len());
        
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .border_style(Style::default().fg(Color::Rgb(100, 100, 100))),
            );
        
        frame.render_widget(list, participant_area);
    }
    
    /// Render typing indicators at bottom of chat
    fn render_typing_indicators(
        &self,
        frame: &mut Frame,
        area: Rect,
        session: &CollaborationSession,
    ) {
        let typing_users: Vec<_> = session.participants
            .values()
            .filter(|p| p.is_typing)
            .collect();
        
        if typing_users.is_empty() {
            return;
        }
        
        let indicator_area = Rect {
            x: area.x + 2,
            y: area.y + area.height.saturating_sub(2),
            width: area.width.saturating_sub(4),
            height: 1,
        };
        
        let text = if typing_users.len() == 1 {
            format!(
                "{} is typing{}",
                typing_users[0].username,
                self.get_typing_animation()
            )
        } else if typing_users.len() == 2 {
            format!(
                "{} and {} are typing{}",
                typing_users[0].username,
                typing_users[1].username,
                self.get_typing_animation()
            )
        } else {
            format!(
                "{} and {} others are typing{}",
                typing_users[0].username,
                typing_users.len() - 1,
                self.get_typing_animation()
            )
        };
        
        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC));
        
        frame.render_widget(paragraph, indicator_area);
    }
    
    /// Render a user's cursor
    fn render_user_cursor(
        &mut self,
        frame: &mut Frame,
        chat_area: Rect,
        cursor: &CursorPosition,
        user: &UserPresence,
    ) {
        // Calculate cursor position in chat area
        let cursor_x = chat_area.x + cursor.column.min(chat_area.width as usize - 1) as u16;
        let cursor_y = chat_area.y + cursor.line.min(chat_area.height as usize - 1) as u16;
        
        // Get or create blink state
        let blink = self.cursor_blink_state
            .entry(user.user_id.clone())
            .or_insert(true);
        
        if *blink {
            // Render cursor
            let cursor_char = "▌";
            let cursor = Paragraph::new(cursor_char)
                .style(Style::default().fg(self.parse_color(&user.color)));
            
            let cursor_area = Rect {
                x: cursor_x,
                y: cursor_y,
                width: 1,
                height: 1,
            };
            
            frame.render_widget(cursor, cursor_area);
            
            // Render username label near cursor
            let label = Paragraph::new(user.username.clone())
                .style(
                    Style::default()
                        .fg(self.parse_color(&user.color))
                        .bg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                );
            
            let label_width = user.username.len() as u16;
            let label_x = if cursor_x + label_width + 2 < chat_area.x + chat_area.width {
                cursor_x + 2
            } else {
                cursor_x.saturating_sub(label_width + 2)
            };
            
            let label_area = Rect {
                x: label_x,
                y: cursor_y.saturating_sub(1),
                width: label_width,
                height: 1,
            };
            
            frame.render_widget(label, label_area);
        }
    }
    

    
    /// Get typing animation string
    fn get_typing_animation(&self) -> &'static str {
        match self.typing_animation_frame / 2 {
            0 => ".",
            1 => "..",
            2 => "...",
            _ => ".",
        }
    }
    
    /// Parse color string to ratatui Color
    fn parse_color(&self, color: &str) -> Color {
        if color.starts_with('#') && color.len() == 7 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&color[1..3], 16),
                u8::from_str_radix(&color[3..5], 16),
                u8::from_str_radix(&color[5..7], 16),
            ) {
                return Color::Rgb(r, g, b);
            }
        }
        Color::White
    }
    
    /// Set UI configuration
    pub fn set_ui_config(&mut self, config: CollaborationUI) {
        self.ui_config = config;
    }
    
    /// Toggle participant list visibility
    pub fn toggle_participants(&mut self) {
        self.ui_config.show_participants = !self.ui_config.show_participants;
    }
    
    /// Cycle participant list position
    pub fn cycle_participant_position(&mut self) {
        use ParticipantListPosition::*;
        self.ui_config.participants_position = match self.ui_config.participants_position {
            Top => Right,
            Right => Bottom,
            Bottom => Floating,
            Floating => Top,
        };
    }
}

impl Default for CollaborationRenderer {
    fn default() -> Self {
        Self::new()
    }
}