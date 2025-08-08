//! Story Visualization Components for TUI
//!
//! This module provides visual representations of code narratives,
//! character relationships, and story arcs in the chat interface.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders,  List, ListItem, Paragraph,canvas::Canvas},
    Frame,
};

use crate::tui::story_driven_code_analysis::{
    CodeNarrative, CodeCharacter, CharacterRole,
    RelationshipType,
};

/// Story visualization panel
pub struct StoryVisualizationPanel {
    /// Current narrative
    narrative: Option<CodeNarrative>,
    
    /// Selected view
    current_view: StoryView,
    
    /// Animation frame
    animation_frame: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoryView {
    CharacterMap,
    RelationshipGraph,
    StoryArc,
    ThemeCloud,
    Timeline,
}

impl StoryVisualizationPanel {
    pub fn new() -> Self {
        Self {
            narrative: None,
            current_view: StoryView::CharacterMap,
            animation_frame: 0,
        }
    }
    
    /// Set narrative to visualize
    pub fn set_narrative(&mut self, narrative: CodeNarrative) {
        self.narrative = Some(narrative);
    }
    
    /// Update animation
    pub fn update(&mut self) {
        self.animation_frame = (self.animation_frame + 1) % 60;
    }
    
    /// Switch view
    pub fn switch_view(&mut self, view: StoryView) {
        self.current_view = view;
    }
    
    /// Next view
    pub fn next_view(&mut self) {
        self.current_view = match self.current_view {
            StoryView::CharacterMap => StoryView::RelationshipGraph,
            StoryView::RelationshipGraph => StoryView::StoryArc,
            StoryView::StoryArc => StoryView::ThemeCloud,
            StoryView::ThemeCloud => StoryView::Timeline,
            StoryView::Timeline => StoryView::CharacterMap,
        };
    }
    
    /// Render the panel
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if let Some(narrative) = &self.narrative {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(format!(" 📖 Story Visualization - {} ", self.view_title()))
                .style(Style::default().fg(Color::Cyan));
            
            let inner = block.inner(area);
            f.render_widget(block, area);
            
            match self.current_view {
                StoryView::CharacterMap => self.render_character_map(f, inner, narrative),
                StoryView::RelationshipGraph => self.render_relationship_graph(f, inner, narrative),
                StoryView::StoryArc => self.render_story_arc(f, inner, narrative),
                StoryView::ThemeCloud => self.render_theme_cloud(f, inner, narrative),
                StoryView::Timeline => self.render_timeline(f, inner, narrative),
            }
        } else {
            self.render_empty(f, area);
        }
    }
    
    /// Get view title
    fn view_title(&self) -> &'static str {
        match self.current_view {
            StoryView::CharacterMap => "Character Map",
            StoryView::RelationshipGraph => "Relationship Graph",
            StoryView::StoryArc => "Story Arc",
            StoryView::ThemeCloud => "Theme Cloud",
            StoryView::Timeline => "Timeline",
        }
    }
    
    /// Render empty state
    fn render_empty(&self, f: &mut Frame, area: Rect) {
        let text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("📖 ", Style::default().fg(Color::Yellow)),
                Span::raw("No story narrative loaded"),
            ]),
            Line::from(""),
            Line::from("Run '/story analyze' to generate a code narrative"),
        ];
        
        let paragraph = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(" Story Visualization "))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(paragraph, area);
    }
    
    /// Render character map
    fn render_character_map(&self, f: &mut Frame, area: Rect, narrative: &CodeNarrative) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);
        
        // Character grid
        self.render_character_grid(f, chunks[0], &narrative.characters);
        
        // Character details
        self.render_character_details(f, chunks[1], &narrative.characters);
    }
    
    /// Render character grid
    fn render_character_grid(&self, f: &mut Frame, area: Rect, characters: &[CodeCharacter]) {
        let items: Vec<ListItem> = characters.iter()
            .map(|c| {
                let icon = match c.role {
                    CharacterRole::Protagonist => "🌟",
                    CharacterRole::Antagonist => "⚠️",
                    CharacterRole::Supporting => "🤝",
                    CharacterRole::Mentor => "🧙",
                    CharacterRole::Sidekick => "🤖",
                };
                
                let style = match c.role {
                    CharacterRole::Protagonist => Style::default().fg(Color::Yellow),
                    CharacterRole::Antagonist => Style::default().fg(Color::Red),
                    CharacterRole::Supporting => Style::default().fg(Color::Green),
                    CharacterRole::Mentor => Style::default().fg(Color::Magenta),
                    CharacterRole::Sidekick => Style::default().fg(Color::Blue),
                };
                
                ListItem::new(vec![
                    Line::from(vec![
                        Span::raw(format!("{} ", icon)),
                        Span::styled(&c.name, style.add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(vec![
                        Span::raw("   "),
                        Span::styled(
                            format!("{:?}", c.role),
                            Style::default().fg(Color::Gray)
                        ),
                    ]),
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Characters"));
        
        f.render_widget(list, area);
    }
    
    /// Render character details
    fn render_character_details(&self, f: &mut Frame, area: Rect, characters: &[CodeCharacter]) {
        // Show details of first protagonist or first character
        let character = characters.iter()
            .find(|c| matches!(c.role, CharacterRole::Protagonist))
            .or_else(|| characters.first());
        
        if let Some(c) = character {
            let mut text = vec![
                Line::from(vec![
                    Span::styled("Character: ", Style::default().fg(Color::Gray)),
                    Span::styled(&c.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Traits:", Style::default().fg(Color::Cyan)),
                ]),
            ];
            
            for trait_name in &c.traits {
                text.push(Line::from(vec![
                    Span::raw("  • "),
                    Span::raw(trait_name),
                ]));
            }
            
            if !c.conflicts.is_empty() {
                text.push(Line::from(""));
                text.push(Line::from(vec![
                    Span::styled("Conflicts:", Style::default().fg(Color::Red)),
                ]));
                
                for conflict in &c.conflicts {
                    text.push(Line::from(vec![
                        Span::raw("  ⚔️ "),
                        Span::raw(conflict),
                    ]));
                }
            }
            
            let paragraph = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Details"))
                .wrap(ratatui::widgets::Wrap { trim: true });
            
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render relationship graph
    fn render_relationship_graph(&self, f: &mut Frame, area: Rect, narrative: &CodeNarrative) {
        // Use Canvas widget for graph visualization
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title("Relationships"))
            .paint(|ctx| {
                // Simple node layout
                let node_count = narrative.characters.len();
                if node_count == 0 {
                    return;
                }
                
                // Draw nodes in a circle
                for (i, character) in narrative.characters.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / node_count as f64;
                    let x = 50.0 + 40.0 * angle.cos();
                    let y = 50.0 + 40.0 * angle.sin();
                    
                    let color = match character.role {
                        CharacterRole::Protagonist => Color::Yellow,
                        CharacterRole::Antagonist => Color::Red,
                        _ => Color::Blue,
                    };
                    
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(x, y)],
                        color,
                    });
                    
                    // Draw character name
                    ctx.print(x, y + 2.0, character.name.clone());
                }
                
                // Draw relationships
                for rel in &narrative.relationships {
                    // Find character positions
                    if let (Some(a_idx), Some(b_idx)) = (
                        narrative.characters.iter().position(|c| c.name == rel.character_a),
                        narrative.characters.iter().position(|c| c.name == rel.character_b),
                    ) {
                        let angle_a = 2.0 * std::f64::consts::PI * a_idx as f64 / node_count as f64;
                        let angle_b = 2.0 * std::f64::consts::PI * b_idx as f64 / node_count as f64;
                        
                        let x1 = 50.0 + 40.0 * angle_a.cos();
                        let y1 = 50.0 + 40.0 * angle_a.sin();
                        let x2 = 50.0 + 40.0 * angle_b.cos();
                        let y2 = 50.0 + 40.0 * angle_b.sin();
                        
                        let color = match rel.relationship_type {
                            RelationshipType::Dependency => Color::Green,
                            RelationshipType::Conflict => Color::Red,
                            _ => Color::Gray,
                        };
                        
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1, y1, x2, y2,
                            color,
                        });
                    }
                }
            })
            .x_bounds([0.0, 100.0])
            .y_bounds([0.0, 100.0]);
        
        f.render_widget(canvas, area);
    }
    
    /// Render story arc
    fn render_story_arc(&self, f: &mut Frame, area: Rect, narrative: &CodeNarrative) {
        let arc = &narrative.story_arc;
        
        let text = vec![
            Line::from(vec![
                Span::styled("Genre: ", Style::default().fg(Color::Gray)),
                Span::styled(&arc.genre, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("🎬 Exposition:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(format!("  {}", arc.exposition)),
            Line::from(""),
            Line::from(vec![
                Span::styled("📈 Rising Action:", Style::default().fg(Color::Green)),
            ]),
        ];
        
        let mut lines = text;
        for action in &arc.rising_action {
            lines.push(Line::from(format!("  • {}", action)));
        }
        
        lines.extend(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("⚡ Climax:", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(format!("  {}", arc.climax)),
            Line::from(""),
            Line::from(vec![
                Span::styled("🌅 Resolution:", Style::default().fg(Color::Magenta)),
            ]),
            Line::from(format!("  {}", arc.resolution)),
        ]);
        
        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Story Arc"))
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        f.render_widget(paragraph, area);
    }
    
    /// Render theme cloud
    fn render_theme_cloud(&self, f: &mut Frame, area: Rect, narrative: &CodeNarrative) {
        let items: Vec<ListItem> = narrative.themes.iter()
            .map(|theme| {
                let size_indicator = match theme.prevalence {
                    p if p > 0.8 => "🔴🔴🔴",
                    p if p > 0.6 => "🟡🟡",
                    p if p > 0.4 => "🟢",
                    _ => "⚪",
                };
                
                let color = match theme.prevalence {
                    p if p > 0.8 => Color::Red,
                    p if p > 0.6 => Color::Yellow,
                    p if p > 0.4 => Color::Green,
                    _ => Color::Gray,
                };
                
                ListItem::new(vec![
                    Line::from(vec![
                        Span::raw(format!("{} ", size_indicator)),
                        Span::styled(
                            &theme.name,
                            Style::default().fg(color).add_modifier(Modifier::BOLD)
                        ),
                        Span::styled(
                            format!(" ({:.0}%)", theme.prevalence * 100.0),
                            Style::default().fg(Color::Gray)
                        ),
                    ]),
                    Line::from(vec![
                        Span::raw("   Examples: "),
                        Span::raw(theme.examples.join(", ")),
                    ]),
                ])
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Theme Cloud"));
        
        f.render_widget(list, area);
    }
    
    /// Render timeline
    fn render_timeline(&self, f: &mut Frame, area: Rect, narrative: &CodeNarrative) {
        let timeline = &narrative.timeline;
        
        if timeline.chapters.is_empty() {
            let paragraph = Paragraph::new("No timeline data available")
                .block(Block::default().borders(Borders::ALL).title("Timeline"))
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            
            f.render_widget(paragraph, area);
            return;
        }
        
        let current_chapter = timeline.chapters.get(timeline.current_chapter);
        
        let mut text = vec![
            Line::from(vec![
                Span::styled(
                    format!("Chapter {} of {}", timeline.current_chapter + 1, timeline.chapters.len()),
                    Style::default().fg(Color::Cyan)
                ),
            ]),
            Line::from(""),
        ];
        
        if let Some(chapter) = current_chapter {
            text.extend(vec![
                Line::from(vec![
                    Span::styled(
                        &chapter.title,
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Period: ", Style::default().fg(Color::Gray)),
                    Span::raw(&chapter.time_period),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Summary:", Style::default().fg(Color::Green)),
                ]),
                Line::from(format!("  {}", chapter.summary)),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Key Events:", Style::default().fg(Color::Magenta)),
                ]),
            ]);
            
            for event in &chapter.key_events {
                text.push(Line::from(format!("  • {}", event)));
            }
        }
        
        let paragraph = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Timeline"))
            .wrap(ratatui::widgets::Wrap { trim: true });
        
        f.render_widget(paragraph, area);
    }
}

/// Mini story indicator for status bar
pub struct StoryMiniIndicator {
    has_narrative: bool,
    character_count: usize,
    theme_count: usize,
}

impl StoryMiniIndicator {
    pub fn new() -> Self {
        Self {
            has_narrative: false,
            character_count: 0,
            theme_count: 0,
        }
    }
    
    /// Update from narrative
    pub fn update(&mut self, narrative: Option<&CodeNarrative>) {
        if let Some(n) = narrative {
            self.has_narrative = true;
            self.character_count = n.characters.len();
            self.theme_count = n.themes.len();
        } else {
            self.has_narrative = false;
            self.character_count = 0;
            self.theme_count = 0;
        }
    }
    
    /// Render as status line
    pub fn render_line(&self) -> Line<'static> {
        if self.has_narrative {
            Line::from(vec![
                Span::styled("📖 ", Style::default().fg(Color::Yellow)),
                Span::raw("Story: "),
                Span::styled(
                    format!("{} chars", self.character_count),
                    Style::default().fg(Color::Cyan)
                ),
                Span::raw(", "),
                Span::styled(
                    format!("{} themes", self.theme_count),
                    Style::default().fg(Color::Magenta)
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled("📖 ", Style::default().fg(Color::DarkGray)),
                Span::styled("No story", Style::default().fg(Color::DarkGray)),
            ])
        }
    }
}

/// Story visualization integration helper
pub struct StoryVisualizationIntegration;

impl StoryVisualizationIntegration {
    /// Create keyboard shortcuts
    pub fn shortcuts() -> Vec<(&'static str, &'static str)> {
        vec![
            ("S", "Toggle story panel"),
            ("Tab", "Next story view (when panel open)"),
            ("1-5", "Jump to specific view"),
        ]
    }
    
    /// Format view help
    pub fn view_help() -> String {
        "📖 Story Views:\n\
        1. Character Map - Components and their roles\n\
        2. Relationship Graph - Dependencies and conflicts\n\
        3. Story Arc - Narrative structure\n\
        4. Theme Cloud - Patterns and principles\n\
        5. Timeline - Evolution chapters".to_string()
    }
}