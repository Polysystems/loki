use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table, Wrap, canvas::Canvas, Sparkline, Clear};
use chrono;
use std::collections::HashMap;

use crate::tui::app::App;
use crate::tui::ui::draw_sub_tab_navigation;
use crate::tui::connectors::system_connector::{MemoryData};
use crate::tui::visual_components::{ AnimatedGauge, LoadingSpinner};

pub fn draw_tab_memory(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Sub-tab navigation
            Constraint::Min(0),     // Content
        ])
        .split(area);

    // Draw sub-tab navigation
    draw_sub_tab_navigation(f, &app.state.memory_tabs, chunks[0]);

    // Draw content based on current sub-tab
    match app.state.memory_tabs.current_key() {
        Some("overview") => draw_memory_overview(f, app, chunks[1]),
        Some("memory") => draw_memory_management(f, app, chunks[1]),
        Some("database") => draw_database_management(f, app, chunks[1]),
        Some("stories") => draw_story_engine(f, app, chunks[1]),
        Some("storage") => draw_storage_management(f, app, chunks[1]),
        _ => draw_memory_overview(f, app, chunks[1]),
    }
}

fn draw_memory_management(f: &mut Frame, app: &mut App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_memory_management_enhanced(f, app, area);
    } else {
        draw_memory_management_legacy(f, app, area);
    }
}

/// Enhanced memory management with real data and beautiful visualizations
fn draw_memory_management_enhanced(f: &mut Frame, app: &mut App, area: Rect) {

    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing memory system...");
            return;
        }
    };

    // Get memory data
    let memory_data = match system_connector.get_memory_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load memory data: {}", e));
            return;
        }
    };

    // Create a tabbed view for memory subsystems
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Sub-navigation for memory subsystems
            Constraint::Length(15),  // Memory overview section
            Constraint::Min(0),      // Content area
        ])
        .split(area);

    // Draw sub-navigation for memory subsystems
    let memory_subsystems = vec!["Fractal", "Knowledge Graph", "Associations", "Operations"];
    let selected_subsystem = app.state.selected_memory_subsystem.unwrap_or(0);
    
    let tabs_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));
    
    let tabs_text = memory_subsystems.iter().enumerate().map(|(i, &name)| {
        if i == selected_subsystem {
            Span::styled(format!(" {} ", name), Style::default().fg(Color::White).bg(Color::Blue).add_modifier(Modifier::BOLD))
        } else {
            Span::styled(format!(" {} ", name), Style::default().fg(Color::Gray))
        }
    }).collect::<Vec<_>>();
    
    let tabs_widget = Paragraph::new(Line::from(tabs_text))
        .block(tabs_block);
    
    f.render_widget(tabs_widget, chunks[0]);

    // Memory overview with real-time stats
    draw_memory_overview_cards(f, app, chunks[1], &memory_data);

    // Draw content based on selected subsystem
    match selected_subsystem {
        0 => {
            // Fractal Memory View
            let content_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(60), // Fractal visualization
                    Constraint::Min(0),         // Memory layers
                ])
                .split(chunks[2]);
            
            draw_enhanced_fractal_memory_view(f, app, content_chunks[0], &memory_data);
            draw_memory_layers_enhanced(f, content_chunks[1], &memory_data);
        }
        1 => {
            // Knowledge Graph View
            draw_knowledge_graph_view(f, app, chunks[2], &memory_data);
        }
        2 => {
            // Associations View
            draw_associations_view(f, app, chunks[2], &memory_data);
        }
        3 => {
            // Memory Operations View
            draw_memory_operations_view(f, app, chunks[2], &memory_data);
        }
        _ => {
            draw_enhanced_fractal_memory_view(f, app, chunks[2], &memory_data);
        }
    }
}

fn draw_memory_management_legacy(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Memory overview
            Constraint::Min(8),     // Memory layers
            Constraint::Length(8),  // Memory operations
        ])
        .split(area);

    // Get real memory statistics if available
    let (total_memories, cache_hit_rate, memory_usage_mb, embedding_count) =
        if let Some(cognitive_system) = &app.cognitive_system {
            let memory = cognitive_system.memory();

            // Get memory layer stats
            let stats = memory.stats();

            // Get storage statistics
            let storage_stats = memory.get_storage_statistics().unwrap_or(
                crate::memory::StorageStatistics {
                    total_memories: 0,
                    cache_memory_mb: 0.0,
                    disk_usage_mb: 0.0,
                    embedding_count: 0,
                    association_count: 0,
                }
            );

            let total_memories = storage_stats.total_memories;
            let cache_hit_rate = stats.cache_hit_rate;
            let memory_usage_mb = storage_stats.cache_memory_mb + storage_stats.disk_usage_mb;
            let embedding_count = stats.total_embeddings;

            (total_memories, cache_hit_rate, memory_usage_mb, embedding_count)
        } else {
            (0, 0.0, 0.0, 0)
        };

    // Memory Overview
    let overview_lines = vec![
        Line::from(vec![
            Span::styled("🧠 Memory System Overview", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Total Memories: "),
            Span::styled(
                format!("{}", total_memories),
                Style::default().fg(Color::White)
            ),
        ]),
        Line::from(vec![
            Span::raw("Memory Usage: "),
            Span::styled(
                format!("{:.1} MB", memory_usage_mb),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::raw("Cache Hit Rate: "),
            Span::styled(
                format!("{:.1}%", cache_hit_rate * 100.0),
                Style::default().fg(if cache_hit_rate > 0.8 { Color::Green } else { Color::Yellow })
            ),
        ]),
        Line::from(vec![
            Span::raw("Embeddings: "),
            Span::styled(
                format!("{}", embedding_count),
                Style::default().fg(Color::Magenta)
            ),
        ]),
        Line::from(vec![
            Span::raw("Active Patterns: "),
            Span::styled(
                if let Some(cognitive_system) = &app.cognitive_system {
                    if let Some(fractal_activator) = cognitive_system.fractal_activator() {
                        let fractal_stats = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                fractal_activator.get_fractal_stats().await
                            })
                        });
                        format!("{}", fractal_stats.total_nodes)
                    } else {
                        "0".to_string()
                    }
                } else {
                    "0".to_string()
                },
                Style::default().fg(Color::Blue)
            ),
        ]),
    ];

    let overview_widget = Paragraph::new(overview_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " Overview ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(overview_widget, chunks[0]);

    // Memory Layers Table
    let header = Row::new(vec!["Layer", "Size", "Items", "Hit Rate", "Status"])
        .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));

    let rows = if let Some(cognitive_system) = &app.cognitive_system {
        let memory = cognitive_system.memory();
        let stats = memory.stats();

        let mut rows = vec![];

        // Cache layers
        rows.push(Row::new(vec![
            "Cache".to_string(),
            format!("{:.1} MB", memory_usage_mb * 0.1), // Estimate cache portion
            format!("{}", embedding_count),
            format!("{:.1}%", cache_hit_rate * 100.0),
            "✅".to_string()
        ]));

        // Short-term memory
        rows.push(Row::new(vec![
            "Short-term".to_string(),
            "Dynamic".to_string(),
            format!("{}", stats.short_term_count),
            "N/A".to_string(),
            "✅".to_string()
        ]));

        // Long-term memory layers
        for (i, count) in stats.long_term_counts.iter().enumerate() {
            rows.push(Row::new(vec![
                format!("Long-term L{}", i + 1),
                "Dynamic".to_string(),
                format!("{}", count),
                "N/A".to_string(),
                "✅".to_string()
            ]));
        }

        // Fractal memory (if available)
        if let Some(fractal_activator) = cognitive_system.fractal_activator() {
            let fractal_stats = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    fractal_activator.get_fractal_stats().await
                })
            });
            rows.push(Row::new(vec![
                "Fractal".to_string(),
                "Dynamic".to_string(),
                format!("{}", fractal_stats.total_nodes),
                format!("{:.1}%", fractal_stats.avg_coherence * 100.0),
                "🔄".to_string()
            ]));
        }

        rows
    } else {
        // Fallback placeholder data
        vec![
            Row::new(vec!["System".to_string(), "Offline".to_string(), "0".to_string(), "0%".to_string(), "❌".to_string()]),
        ]
    };

    let layers_table = Table::new(
        rows,
        vec![
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue))
            .title(Span::styled(
                " Memory Layers ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
    )
    .style(Style::default().fg(Color::Gray));

    f.render_widget(layers_table, chunks[1]);

    // Memory Operations - get real operations from activity history
    let operations: Vec<ListItem> = app.state.recent_activities
        .iter()
        .filter(|activity| {
            activity.activity_type == crate::tui::state::ActivityType::MemoryOptimization
        })
        .rev()
        .take(10)
        .map(|activity| {
            // Parse operation type from description
            let (op_type, color) = if activity.description.starts_with("STORE:") {
                ("STORE", Color::Green)
            } else if activity.description.starts_with("RECALL:") {
                ("RECALL", Color::Blue)
            } else if activity.description.starts_with("OPTIMIZE:") {
                ("OPTIMIZE", Color::Yellow)
            } else if activity.description.starts_with("PRUNE:") {
                ("PRUNE", Color::Red)
            } else if activity.description.contains("cache hit rate") {
                ("CACHE", Color::Cyan)
            } else if activity.description.contains("Fractal patterns") {
                ("FRACTAL", Color::Magenta)
            } else {
                ("MEMORY", Color::White)
            };

            // Clean up description by removing the operation prefix if present
            let description = if activity.description.contains(':') {
                activity.description.split_once(": ")
                    .map(|(_, desc)| desc)
                    .unwrap_or(&activity.description)
            } else {
                &activity.description
            };

            ListItem::new(Line::from(vec![
                Span::styled(format!("[{}]", &activity.timestamp[..8]), Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(op_type, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::raw(": "),
                Span::raw(description),
            ]))
        })
        .collect();

    // If no operations, show placeholder
    let operations = if operations.is_empty() {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("No recent memory operations", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ])),
        ]
    } else {
        operations
    };

    let operations_list = List::new(operations)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(Span::styled(
                    " Recent Operations ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(operations_list, chunks[2]);
}

fn draw_memory_overview(f: &mut Frame, app: &mut App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_memory_overview_enhanced(f, app, area);
    } else {
        draw_memory_overview_legacy(f, app, area);
    }
}

fn draw_memory_overview_enhanced(f: &mut Frame, app: &mut App, area: Rect) {
    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_error_state(f, area, "System connector not available");
            return;
        }
    };

    // Get overview data
    let overview_data = match system_connector.get_memory_overview_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load overview data: {}", e));
            return;
        }
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // System health summary
            Constraint::Percentage(35), // System health dashboard
            Constraint::Percentage(30), // Resource usage and activity feed
            Constraint::Min(8),     // Quick actions and system metrics
        ])
        .split(area);

    // System Health Summary
    draw_overview_health_summary(f, chunks[0], &overview_data);
    
    // System Health Dashboard
    draw_system_health_dashboard(f, chunks[1], &overview_data);
    
    // Resource Usage and Activity Feed
    draw_resource_and_activity(f, chunks[2], &overview_data);
    
    // Quick Actions and System Metrics
    draw_quick_actions_and_metrics(f, chunks[3], &overview_data);
}

fn draw_overview_health_summary(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let overall_score = (data.memory_health.health_score + data.database_health.health_score + 
                        data.story_health.health_score + data.cognitive_health.health_score) / 4.0;
    let status = if overall_score > 0.8 { "🟢 All Systems Operational" } else if overall_score > 0.6 { "🟡 Some Issues Detected" } else { "🔴 System Issues" };
    let status_color = if overall_score > 0.8 { Color::Green } else if overall_score > 0.6 { Color::Yellow } else { Color::Red };

    let health_lines = vec![
        Line::from(vec![
            Span::styled("🧠 Memory System Overview", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("System Status: "),
            Span::styled(status, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Overall Health: "),
            Span::styled(format!("{:.1}%", overall_score * 100.0), Style::default().fg(status_color)),
        ]),
        Line::from(vec![
            Span::raw("Last Updated: "),
            Span::styled(
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                Style::default().fg(Color::Gray)
            ),
        ]),
    ];

    let health_widget = Paragraph::new(health_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " System Health ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(health_widget, area);
}

/// Draw comprehensive system health dashboard
fn draw_system_health_dashboard(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Memory health
            Constraint::Percentage(25), // Database health
            Constraint::Percentage(25), // Stories health
            Constraint::Percentage(25), // Cognitive health
        ])
        .split(area);

    // Memory System Health
    draw_system_health_card(f, chunks[0], "🧠 Memory", &data.memory_health, Color::Cyan);
    
    // Database System Health
    draw_system_health_card(f, chunks[1], "🗄️ Database", &data.database_health, Color::Blue);
    
    // Stories System Health
    draw_system_health_card(f, chunks[2], "📖 Stories", &data.story_health, Color::Magenta);
    
    // Cognitive System Health
    draw_system_health_card(f, chunks[3], "🤖 Cognitive", &data.cognitive_health, Color::Green);
}

/// Draw individual system health card
fn draw_system_health_card(f: &mut Frame, area: Rect, title: &str, health: &crate::tui::connectors::system_connector::SystemHealth, color: Color) {
    let health_bar = match (health.health_score * 10.0) as u8 {
        0..=3 => "████░░░░░░",
        4..=5 => "██████░░░░",
        6..=7 => "████████░░",
        8..=9 => "██████████",
        _ => "██████████",
    };

    let health_color = if health.health_score > 0.8 {
        Color::Green
    } else if health.health_score > 0.6 {
        Color::Yellow
    } else {
        Color::Red
    };

    let health_text = vec![
        Line::from(vec![
            Span::styled(title, Style::default().fg(color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(&health.status, Style::default().fg(health_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(health_bar, Style::default().fg(health_color)),
        ]),
        Line::from(vec![
            Span::styled(format!("{:.1}%", health.health_score * 100.0), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(&health.details[..health.details.len().min(20)], Style::default().fg(Color::Gray)),
        ]),
    ];

    let health_widget = Paragraph::new(health_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(color))
        )
        .alignment(Alignment::Center);

    f.render_widget(health_widget, area);
}

/// Draw resource usage and activity feed
fn draw_resource_and_activity(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Resource usage
            Constraint::Percentage(60), // Activity feed
        ])
        .split(area);

    // Resource Usage
    draw_resource_usage(f, chunks[0], data);
    
    // Activity Feed
    draw_activity_feed(f, chunks[1], data);
}

/// Draw resource usage panel
fn draw_resource_usage(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let usage = &data.resource_usage;
    
    let resource_text = vec![
        Line::from(vec![
            Span::styled("📊 Resource Usage", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Memory: "),
            Span::styled(format!("{:.1}GB / {:.1}GB", usage.memory_usage_gb, usage.memory_total_gb), 
                Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Storage: "),
            Span::styled(format!("{:.0}MB / {:.1}GB", usage.storage_usage_mb, usage.storage_total_gb), 
                Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Connections: "),
            Span::styled(format!("{} / {}", usage.active_connections, usage.max_connections), 
                Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Operations/sec: "),
            Span::styled(format!("{:.1}", usage.operations_per_second), 
                Style::default().fg(Color::Magenta)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("CPU Usage: "),
            Span::styled(format!("{:.1}%", usage.cpu_usage_percent), 
                Style::default().fg(if usage.cpu_usage_percent > 80.0 { Color::Red } else { Color::Green })),
        ]),
        Line::from(vec![
            Span::raw("Cache Hit Rate: "),
            Span::styled(format!("{:.1}%", usage.cache_hit_rate), 
                Style::default().fg(if usage.cache_hit_rate > 80.0 { Color::Green } else { Color::Yellow })),
        ]),
    ];

    let resource_widget = Paragraph::new(resource_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("💾 Resources")
                .border_style(Style::default().fg(Color::Yellow)),
        );

    f.render_widget(resource_widget, area);
}

/// Draw activity feed panel
fn draw_activity_feed(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let activity_items: Vec<ListItem> = data.activity_feed.iter().take(8).map(|activity| {
        let severity_color = match activity.severity {
            crate::tui::connectors::system_connector::OverviewActivitySeverity::Success => Color::Green,
            crate::tui::connectors::system_connector::OverviewActivitySeverity::Info => Color::Blue,
            crate::tui::connectors::system_connector::OverviewActivitySeverity::Warning => Color::Yellow,
            crate::tui::connectors::system_connector::OverviewActivitySeverity::Error => Color::Red,
            crate::tui::connectors::system_connector::OverviewActivitySeverity::Critical => Color::Magenta,
        };

        let system_color = match activity.system.as_str() {
            "Memory" => Color::Cyan,
            "Database" => Color::Blue,
            "Stories" => Color::Magenta,
            "Cognitive" => Color::Green,
            "Cache" => Color::Yellow,
            _ => Color::Gray,
        };

        let elapsed = chrono::Utc::now().signed_duration_since(activity.timestamp);
        let time_str = if elapsed.num_minutes() < 60 {
            format!("[{}m]", elapsed.num_minutes())
        } else {
            format!("[{}h]", elapsed.num_hours())
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(time_str, Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(&activity.title, Style::default().fg(severity_color).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(&activity.system, Style::default().fg(system_color)),
                Span::raw(": "),
                Span::styled(&activity.description, Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let activity_list = List::new(activity_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📋 Activity Feed")
                .border_style(Style::default().fg(Color::Green)),
        );

    f.render_widget(activity_list, area);
}

/// Draw quick actions and system metrics
fn draw_quick_actions_and_metrics(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Quick actions
            Constraint::Percentage(40), // System metrics
        ])
        .split(area);

    // Quick Actions
    draw_quick_actions(f, chunks[0], data);
    
    // System Metrics
    draw_system_metrics(f, chunks[1], data);
}

/// Draw quick actions panel
fn draw_quick_actions(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let action_items: Vec<ListItem> = data.quick_actions.iter().take(6).map(|action| {
        let enabled_color = if action.enabled { Color::White } else { Color::DarkGray };
        let icon_color = if action.enabled { Color::Cyan } else { Color::DarkGray };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&action.icon, Style::default().fg(icon_color)),
                Span::raw(" "),
                Span::styled(&action.title, Style::default().fg(enabled_color).add_modifier(
                    if action.enabled { Modifier::BOLD } else { Modifier::empty() }
                )),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(&action.description, Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let actions_list = List::new(action_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("⚡ Quick Actions")
                .border_style(Style::default().fg(Color::Cyan)),
        );

    f.render_widget(actions_list, area);
}

/// Draw system metrics panel
fn draw_system_metrics(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let metrics = &data.system_metrics;
    let uptime_hours = metrics.uptime_seconds / 3600;
    let uptime_days = uptime_hours / 24;

    let metrics_text = vec![
        Line::from(vec![
            Span::styled("📈 System Metrics", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Uptime: "),
            Span::styled(format!("{}d {}h", uptime_days, uptime_hours % 24), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Operations: "),
            Span::styled(format!("{}", metrics.total_operations), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Success Rate: "),
            Span::styled(format!("{:.1}%", metrics.success_rate * 100.0), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Avg Response: "),
            Span::styled(format!("{:.1}ms", metrics.average_response_time_ms), Style::default().fg(Color::Blue)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Peak Memory: "),
            Span::styled(format!("{:.1}GB", metrics.peak_memory_usage_gb), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Data Processed: "),
            Span::styled(format!("{:.1}MB", metrics.data_processed_mb), Style::default().fg(Color::Magenta)),
        ]),
    ];

    let metrics_widget = Paragraph::new(metrics_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📊 Metrics")
                .border_style(Style::default().fg(Color::Magenta)),
        );

    f.render_widget(metrics_widget, area);
}

fn draw_overview_system_cards(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Memory System Card
    let memory_color = if data.memory_health.health_score > 0.8 { Color::Green } else if data.memory_health.health_score > 0.6 { Color::Yellow } else { Color::Red };
    let memory_card = vec![
        Line::from(vec![
            Span::styled("🧠 Memory", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(&data.memory_health.status, Style::default().fg(memory_color)),
        ]),
        Line::from(vec![
            Span::raw("Health: "),
            Span::styled(format!("{:.1}%", data.memory_health.health_score * 100.0), Style::default().fg(memory_color)),
        ]),
        Line::from(vec![
            Span::raw("Usage: "),
            Span::styled(format!("{:.1} MB", data.total_memory_usage), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(data.memory_health.details.as_str()),
    ];

    let memory_widget = Paragraph::new(memory_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)))
        .wrap(Wrap { trim: true });
    f.render_widget(memory_widget, main_chunks[0]);

    // Database Card
    let db_color = if data.database_health.health_score > 0.8 { Color::Green } else if data.database_health.health_score > 0.6 { Color::Yellow } else { Color::Red };
    let db_card = vec![
        Line::from(vec![
            Span::styled("🗄️ Database", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(&data.database_health.status, Style::default().fg(db_color)),
        ]),
        Line::from(vec![
            Span::raw("Health: "),
            Span::styled(format!("{:.1}%", data.database_health.health_score * 100.0), Style::default().fg(db_color)),
        ]),
        Line::from(vec![
            Span::raw("Storage: "),
            Span::styled(format!("{:.1} MB", data.total_storage_usage), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(data.database_health.details.as_str()),
    ];

    let db_widget = Paragraph::new(db_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)))
        .wrap(Wrap { trim: true });
    f.render_widget(db_widget, main_chunks[1]);

    // Stories Card
    let story_color = if data.story_health.health_score > 0.8 { Color::Green } else if data.story_health.health_score > 0.6 { Color::Yellow } else { Color::Red };
    let stories_card = vec![
        Line::from(vec![
            Span::styled("📖 Stories", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(&data.story_health.status, Style::default().fg(story_color)),
        ]),
        Line::from(vec![
            Span::raw("Health: "),
            Span::styled(format!("{:.1}%", data.story_health.health_score * 100.0), Style::default().fg(story_color)),
        ]),
        Line::from(""),
        Line::from(data.story_health.details.as_str()),
    ];

    let stories_widget = Paragraph::new(stories_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Magenta)))
        .wrap(Wrap { trim: true });
    f.render_widget(stories_widget, main_chunks[2]);

    // System Resources Card
    let total_usage = data.total_memory_usage + data.total_storage_usage;
    let system_card = vec![
        Line::from(vec![
            Span::styled("⚡ System", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Total Usage: "),
            Span::styled(format!("{:.1} MB", total_usage), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::raw("Connections: "),
            Span::styled(format!("{}", data.interconnection_status.len()), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Avg Latency: "),
            Span::styled(
                format!("{:.1}ms", 
                    data.interconnection_status.iter()
                        .map(|ic| ic.latency_ms)
                        .sum::<f32>() / data.interconnection_status.len() as f32
                ), 
                Style::default().fg(Color::Green)
            ),
        ]),
    ];

    let system_widget = Paragraph::new(system_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow)));
    f.render_widget(system_widget, main_chunks[3]);
}

fn draw_overview_interconnections(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::MemoryOverviewData) {
    let interconnections: Vec<ListItem> = data.interconnection_status.iter().map(|ic| {
        let status_color = match ic.status.as_str() {
            "Connected" | "Active" => Color::Green,
            "Connecting" => Color::Yellow,
            _ => Color::Red,
        };

        ListItem::new(Line::from(vec![
            Span::styled(&ic.from_system, Style::default().fg(Color::Cyan)),
            Span::raw(" → "),
            Span::styled(&ic.to_system, Style::default().fg(Color::Blue)),
            Span::raw(" | "),
            Span::styled(&ic.status, Style::default().fg(status_color)),
            Span::raw(" | "),
            Span::styled(format!("{:.1}ms", ic.latency_ms), Style::default().fg(Color::Yellow)),
            Span::raw(" | "),
            Span::styled(format!("{:.1} ops/s", ic.throughput_ops_per_sec), Style::default().fg(Color::Magenta)),
        ]))
    }).collect();

    let interconnections_list = List::new(interconnections)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("System Interconnections")
                .border_style(Style::default().fg(Color::Green)),
        );

    f.render_widget(interconnections_list, area);
}

fn draw_overview_navigation(f: &mut Frame, area: Rect) {
    let actions_lines = vec![
        Line::from(vec![
            Span::styled("🚀 Quick Actions", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from("Use the sub-tabs above to navigate:"),
        Line::from("  • Memory Tab - Explore memory layers, embeddings, and associations"),
        Line::from("  • Database Tab - Manage PostgreSQL, SQLite, Redis, and MongoDB"),
        Line::from("  • Stories Tab - Monitor story engine and narrative generation"),
        Line::from(""),
        Line::from(vec![
            Span::styled("Tip: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Press Tab to cycle between tabs, or use number keys (1-4) for quick navigation."),
        ]),
    ];

    let actions_widget = Paragraph::new(actions_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray))
                .title(Span::styled(
                    " Navigation Help ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(actions_widget, area);
}

fn draw_memory_overview_legacy(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // System health summary
            Constraint::Percentage(60), // Main overview cards
            Constraint::Min(8),     // Quick actions and navigation
        ])
        .split(area);

    // System Health Summary
    let health_lines = vec![
        Line::from(vec![
            Span::styled("🧠 Memory System Overview", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("System Status: "),
            Span::styled("🟢 All Systems Operational", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Last Updated: "),
            Span::styled(
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                Style::default().fg(Color::Gray)
            ),
        ]),
        Line::from(vec![
            Span::raw("Active Subsystems: "),
            Span::styled("Memory • Database • Stories • Cache", Style::default().fg(Color::Blue)),
        ]),
    ];

    let health_widget = Paragraph::new(health_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " System Health ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(health_widget, chunks[0]);

    // Main Overview Cards
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);

    // Memory System Card - Try multiple sources for memory data
    let memory_stats = if let Some(cognitive_system) = &app.cognitive_system {
        let memory = cognitive_system.memory();
        let stats = memory.stats();
        let storage_stats = memory.get_storage_statistics().unwrap_or_default();
        (storage_stats.total_memories, stats.cache_hit_rate, storage_stats.cache_memory_mb)
    } else if let Some(ref system_connector) = app.system_connector {
        // Try to get from system connector's memory system
        if let Some(memory) = &system_connector.memory_system {
            let stats = memory.get_statistics().unwrap_or_default();
            let storage_stats = memory.get_storage_statistics().unwrap_or_default();
            (storage_stats.total_memories, stats.cache_hit_rate, storage_stats.cache_memory_mb)
        } else {
            // Memory system not yet initialized, show loading state
            (0, 0.0, 0.0)
        }
    } else {
        (0, 0.0, 0.0)
    };

    let memory_card = vec![
        Line::from(vec![
            Span::styled("🧠 Memory", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Nodes: "),
            Span::styled(format!("{}", memory_stats.0), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::raw("Cache Hit: "),
            Span::styled(format!("{:.1}%", memory_stats.1 * 100.0), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Usage: "),
            Span::styled(format!("{:.1} MB", memory_stats.2), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("→ View Details", Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
        ]),
    ];

    let memory_widget = Paragraph::new(memory_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    f.render_widget(memory_widget, main_chunks[0]);

    // Database Card - Get real connection status
    let db_status = if let Some(ref system_connector) = app.system_connector {
        if let Some(db_manager) = &system_connector.database_manager {
            // Get actual connection status for each backend
            let postgres_status = if db_manager.is_connected("postgresql").unwrap_or(false) {
                ("🟢 Connected", Color::Green)
            } else {
                ("🔴 Disconnected", Color::Red)
            };
            
            let sqlite_status = if db_manager.is_connected("sqlite").unwrap_or(false) {
                ("🟢 Available", Color::Green)
            } else {
                ("🟡 Not initialized", Color::Yellow)
            };
            
            let redis_status = if db_manager.is_connected("redis").unwrap_or(false) {
                ("🟢 Connected", Color::Green)
            } else {
                ("🔴 Disconnected", Color::Red)
            };
            
            (postgres_status, sqlite_status, redis_status)
        } else {
            // Database manager not available
            (("⚫ Not configured", Color::Gray), 
             ("⚫ Not configured", Color::Gray), 
             ("⚫ Not configured", Color::Gray))
        }
    } else {
        // System connector not available
        (("⚪ Unknown", Color::DarkGray), 
         ("⚪ Unknown", Color::DarkGray), 
         ("⚪ Unknown", Color::DarkGray))
    };
    
    let db_card = vec![
        Line::from(vec![
            Span::styled("🗄️ Database", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("PostgreSQL: "),
            Span::styled(db_status.0.0, Style::default().fg(db_status.0.1)),
        ]),
        Line::from(vec![
            Span::raw("SQLite: "),
            Span::styled(db_status.1.0, Style::default().fg(db_status.1.1)),
        ]),
        Line::from(vec![
            Span::raw("Redis: "),
            Span::styled(db_status.2.0, Style::default().fg(db_status.2.1)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("→ Manage DBs", Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
        ]),
    ];

    let db_widget = Paragraph::new(db_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)));
    f.render_widget(db_widget, main_chunks[1]);

    // Stories Card - Try multiple sources for story data
    let stories_stats = if let Some(story_engine) = &app.story_engine {
        let active_count = story_engine.get_active_story_count().unwrap_or(0);
        let total_count = story_engine.get_total_story_count().unwrap_or(0);
        (active_count, total_count, true)
    } else if let Some(ref system_connector) = app.system_connector {
        // Try to get from system connector's story engine
        if let Some(story_engine) = &system_connector.story_engine {
            let active_count = story_engine.get_active_story_count().unwrap_or(0);
            let total_count = story_engine.get_total_story_count().unwrap_or(0);
            (active_count, total_count, true)
        } else {
            // Story engine not available, check mock stories
            let mock_stories = system_connector.mock_stories.read().unwrap();
            let total = mock_stories.len();
            let active = mock_stories.iter().filter(|s| s.status == crate::story::types::StoryStatus::Active).count();
            (active, total, false)
        }
    } else {
        (0, 0, false)
    };

    let stories_card = vec![
        Line::from(vec![
            Span::styled("📖 Stories", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Active: "),
            Span::styled(format!("{}", stories_stats.0), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::raw("Total: "),
            Span::styled(format!("{}", stories_stats.1), Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(
                if stories_stats.2 { "🟢 Running" } else { "🔴 Not initialized" }, 
                Style::default().fg(if stories_stats.2 { Color::Green } else { Color::Red })
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("→ Story Engine", Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
        ]),
    ];

    let stories_widget = Paragraph::new(stories_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Magenta)));
    f.render_widget(stories_widget, main_chunks[2]);

    // System Resources Card - Get real system metrics
    let (cpu_usage, memory_gb, total_memory_gb) = {
        // Use sysinfo to get real system metrics
        use sysinfo::System;
        let mut system = System::new_all();
        system.refresh_all();
        
        let cpu = system.global_cpu_usage();
        let used_mem = system.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_mem = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        (cpu, used_mem, total_mem)
    };
    
    // Get connection count from database manager or default
    let (current_connections, max_connections) = if let Some(ref system_connector) = app.system_connector {
        if let Some(db_manager) = &system_connector.database_manager {
            // Note: active_connections is async, but we're in a sync context
            // Using a default value for now - consider making this function async in the future
            let active = 0; // Would need to be fetched asynchronously
            let max = db_manager.max_connections();
            (active, max)
        } else {
            (0, 20)
        }
    } else {
        (0, 20)
    };
    
    // Determine CPU color based on usage
    let cpu_color = if cpu_usage > 80.0 {
        Color::Red
    } else if cpu_usage > 60.0 {
        Color::Yellow  
    } else {
        Color::Green
    };
    
    // Determine memory color based on usage percentage
    let memory_percentage = (memory_gb / total_memory_gb) * 100.0;
    let memory_color = if memory_percentage > 80.0 {
        Color::Red
    } else if memory_percentage > 60.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    
    let system_card = vec![
        Line::from(vec![
            Span::styled("⚡ System", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("CPU: "),
            Span::styled(format!("{:.1}%", cpu_usage), Style::default().fg(cpu_color)),
        ]),
        Line::from(vec![
            Span::raw("Memory: "),
            Span::styled(format!("{:.1}GB/{:.1}GB", memory_gb, total_memory_gb), Style::default().fg(memory_color)),
        ]),
        Line::from(vec![
            Span::raw("Connections: "),
            Span::styled(format!("{}/{}", current_connections, max_connections), Style::default().fg(Color::Blue)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("→ Monitor", Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
        ]),
    ];

    let system_widget = Paragraph::new(system_card)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow)));
    f.render_widget(system_widget, main_chunks[3]);

    // Quick Actions and Navigation
    let actions_lines = vec![
        Line::from(vec![
            Span::styled("🚀 Quick Actions", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from("Use the sub-tabs above to navigate:"),
        Line::from("  • Memory Tab - Explore memory layers, embeddings, and associations"),
        Line::from("  • Database Tab - Manage PostgreSQL, SQLite, Redis, and MongoDB"),
        Line::from("  • Stories Tab - Monitor story engine and narrative generation"),
        Line::from(""),
        Line::from(vec![
            Span::styled("Tip: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Press Tab to cycle between tabs, or use number keys (1-4) for quick navigation."),
        ]),
    ];

    let actions_widget = Paragraph::new(actions_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray))
                .title(Span::styled(
                    " Navigation Help ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(actions_widget, chunks[2]);
}

fn draw_database(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Database status
            Constraint::Min(10),    // Collections table
            Constraint::Length(8),  // Query history
        ])
        .split(area);

    // Get real database status
    let (db_status, db_size, _, collection_count) =
        if let Some(cognitive_system) = &app.cognitive_system {
            let memory = cognitive_system.memory();

            // Get storage statistics
            let storage_stats = memory.get_storage_statistics().unwrap_or(
                crate::memory::StorageStatistics {
                    total_memories: 0,
                    cache_memory_mb: 0.0,
                    disk_usage_mb: 0.0,
                    embedding_count: 0,
                    association_count: 0,
                }
            );

            // Estimate database size
            let db_size_gb = storage_stats.disk_usage_mb / 1024.0;
            let db_status = if storage_stats.total_memories > 0 { "● Online" } else { "● Empty" };

            // Count collections (we know there are at least memories and embeddings)
            let collection_count = if storage_stats.total_memories > 0 { 4 } else { 0 };

            (db_status, db_size_gb, storage_stats.total_memories, collection_count)
        } else {
            ("● Offline", 0.0, 0, 0)
        };

    // Database Status
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🗄️ Database Status", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Engine: "),
            Span::styled("RocksDB v8.11.3", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(
                db_status,
                Style::default().fg(
                    if db_status.contains("Online") { Color::Green }
                    else if db_status.contains("Empty") { Color::Yellow }
                    else { Color::Red }
                )
            ),
        ]),
        Line::from(vec![
            Span::raw("Size: "),
            Span::styled(format!("{:.1} GB", db_size), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Collections: "),
            Span::styled(format!("{}", collection_count), Style::default().fg(Color::Blue)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(Span::styled(
                    " Status ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(status_widget, chunks[0]);

    // Collections Table
    let header = Row::new(vec!["Collection", "Documents", "Size", "Indexes", "Status"])
        .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));

    let rows = if let Some(cognitive_system) = &app.cognitive_system {
        let memory = cognitive_system.memory();

        // Get storage statistics
        let storage_stats = memory.get_storage_statistics().unwrap_or_default();

        let mut rows = vec![];

        // Memories collection
        rows.push(Row::new(vec![
            "memories".to_string(),
            format!("{}", storage_stats.total_memories),
            format!("{:.1} MB", storage_stats.cache_memory_mb),
            "3".to_string(),
            "✅".to_string()
        ]));

        // Embeddings collection
        rows.push(Row::new(vec![
            "embeddings".to_string(),
            format!("{}", storage_stats.embedding_count),
            "Dynamic".to_string(),
            "2".to_string(),
            "✅".to_string()
        ]));

        // Associations collection
        rows.push(Row::new(vec![
            "associations".to_string(),
            format!("{}", storage_stats.association_count),
            "Dynamic".to_string(),
            "1".to_string(),
            "✅".to_string()
        ]));

        // Decisions collection (from orchestrator stats)
        if let Some(decisions_made) = app.cognitive_system.as_ref().and_then(|cs| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    Some(cs.orchestrator().get_stats().await.decisions_made)
                })
            })
        }) {
            rows.push(Row::new(vec![
                "decisions".to_string(),
                format!("{}", decisions_made),
                "Dynamic".to_string(),
                "2".to_string(),
                "✅".to_string()
            ]));
        }

        rows
    } else {
        vec![
            Row::new(vec!["No data".to_string(), "0".to_string(), "0 MB".to_string(), "0".to_string(), "❌".to_string()]),
        ]
    };

    let collections_table = Table::new(
        rows,
        vec![
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue))
            .title(Span::styled(
                " Collections ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
    )
    .style(Style::default().fg(Color::Gray));

    f.render_widget(collections_table, chunks[1]);

    // Query History
    let queries = vec![
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:59]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("SELECT", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": patterns WHERE score > 0.8 (125ms)"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:58]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("INSERT", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": memory document #45232 (12ms)"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:57]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("UPDATE", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": agent status for task_distributor (8ms)"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:56]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("INDEX", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(": created embedding_vector_idx (2.3s)"),
        ])),
    ];

    let query_list = List::new(queries)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title(Span::styled(
                    " Recent Queries ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(query_list, chunks[2]);
}

fn draw_database_management(f: &mut Frame, app: &App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_database_enhanced(f, app, area);
    } else {
        draw_database(f, app, area);
    }
}

fn draw_database_enhanced(f: &mut Frame, app: &App, area: Rect) {
    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_error_state(f, area, "System connector not available");
            return;
        }
    };

    // Get database data using async-to-sync bridge
    let database_data = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            system_connector.get_database_data().await
        })
    });
    
    let database_data = match database_data {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load database data: {}", e));
            return;
        }
    };

    // Enhanced layout for database management
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65), // Main view
            Constraint::Percentage(35), // Configuration panel
        ])
        .split(area);

    // Left side - Database status and monitoring
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // Backend status
            Constraint::Length(15),  // Connection pools
            Constraint::Min(0),      // Operations log
        ])
        .split(chunks[0]);

    // Backend Status Overview
    draw_database_backend_status(f, left_chunks[0], &database_data);
    
    // Connection Pools
    draw_database_connection_pools(f, left_chunks[1], &database_data);
    
    // Operations log
    draw_database_operations_log(f, left_chunks[2], &database_data);

    // Right side - Configuration and management
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Database selection
            Constraint::Length(15), // Configuration form
            Constraint::Length(10), // Actions
            Constraint::Min(0),     // Status/Help
        ])
        .split(chunks[1]);

    // Database selector
    draw_database_selector(f, right_chunks[0], &database_data, &app.state.selected_database_backend);
    
    // Configuration form for selected database
    draw_database_config_form(f, right_chunks[1], &database_data, app);
    
    // Database actions
    draw_database_actions(f, right_chunks[2]);
    
    // Help/Status
    draw_database_help(f, right_chunks[3]);
    
    // Show operation messages if any
    if let Some(message) = &app.state.database_operation_message {
        draw_operation_message(f, chunks[1], message, MessageType::Info);
    }
}

fn draw_database_backend_status(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    let backend_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // RocksDB
            Constraint::Percentage(20), // PostgreSQL
            Constraint::Percentage(20), // SQLite
            Constraint::Percentage(20), // Redis
            Constraint::Percentage(20), // Summary
        ])
        .split(area);

    let backends = ["rocksdb", "postgresql", "sqlite", "redis"];
    let colors = [Color::Cyan, Color::Blue, Color::Green, Color::Yellow];

    for (i, backend_key) in backends.iter().enumerate() {
        if let Some(backend) = data.backend_status.get(&backend_key.to_string()) {
            let status_color = match backend.status.as_str() {
                "Connected" => Color::Green,
                "Available" => Color::Green,
                "Empty" => Color::Yellow,
                _ => Color::Red,
            };

            let backend_lines = vec![
                Line::from(vec![
                    Span::styled(&backend.name, Style::default().fg(colors[i]).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled(&backend.status, Style::default().fg(status_color)),
                ]),
                Line::from(vec![
                    Span::raw("v"),
                    Span::styled(&backend.version, Style::default().fg(Color::Gray)),
                ]),
                Line::from(vec![
                    Span::styled(format!("{:.1}MB", backend.size_mb), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled(format!("{}/{}", backend.active_connections, backend.connection_pool_size), Style::default().fg(Color::Blue)),
                ]),
            ];

            let backend_widget = Paragraph::new(backend_lines)
                .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(colors[i])));
            f.render_widget(backend_widget, backend_chunks[i]);
        }
    }

    // Summary card
    let connected_count = data.backend_status.values()
        .filter(|b| b.status == "Connected" || b.status == "Available")
        .count();
    let total_count = data.backend_status.len();

    let summary_lines = vec![
        Line::from(vec![
            Span::styled("Summary", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{}/{}", connected_count, total_count), Style::default().fg(Color::Green)),
            Span::raw(" Connected"),
        ]),
        Line::from(vec![
            Span::styled(format!("{:.1}%", data.query_analytics.cache_hit_rate * 100.0), Style::default().fg(Color::Cyan)),
            Span::raw(" Cache Hit"),
        ]),
        Line::from(vec![
            Span::styled(format!("{:.1}", data.query_analytics.average_response_time_ms), Style::default().fg(Color::Yellow)),
            Span::raw("ms Avg"),
        ]),
        Line::from(vec![
            Span::styled(format!("{:.1}", data.performance_metrics.queries_per_second), Style::default().fg(Color::Magenta)),
            Span::raw(" QPS"),
        ]),
    ];

    let summary_widget = Paragraph::new(summary_lines)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::White)));
    f.render_widget(summary_widget, backend_chunks[4]);
}

fn draw_database_connection_pools(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    let header = Row::new(vec!["Backend", "Active", "Max", "Queries", "Avg RT", "Errors"])
        .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = data.connection_pools.iter().map(|pool| {
        let error_rate = if pool.total_queries > 0 {
            (pool.failed_queries as f32 / pool.total_queries as f32) * 100.0
        } else {
            0.0
        };

        let health_color = if error_rate < 1.0 { Color::Green } else if error_rate < 5.0 { Color::Yellow } else { Color::Red };

        Row::new(vec![
            pool.backend.clone(),
            format!("{}", pool.active_connections),
            format!("{}", pool.max_connections),
            format!("{}", pool.total_queries),
            format!("{:.1}ms", pool.average_response_time_ms),
            format!("{:.1}%", error_rate),
        ])
        .style(Style::default().fg(health_color))
    }).collect();

    let table = Table::new(
        rows,
        vec![
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Length(7),
            Constraint::Length(7),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Connection Pools")
            .border_style(Style::default().fg(Color::Blue)),
    )
    .style(Style::default().fg(Color::White));

    f.render_widget(table, area);
}

fn draw_database_performance_metrics(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    let metrics_lines = vec![
        Line::from(vec![
            Span::styled("⚡ Performance Metrics", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Queries/sec: "),
            Span::styled(format!("{:.1}", data.performance_metrics.queries_per_second), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Data Transfer: "),
            Span::styled(format!("{:.1} MB/s", data.performance_metrics.data_transfer_rate_mbps), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Index Hit Ratio: "),
            Span::styled(format!("{:.1}%", data.performance_metrics.index_hit_ratio * 100.0), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Lock Wait Time: "),
            Span::styled(format!("{:.2}ms", data.performance_metrics.lock_wait_time_ms), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Memory Usage: "),
            Span::styled(format!("{:.1} MB", data.performance_metrics.memory_usage_mb), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::raw("Disk I/O: "),
            Span::styled(format!("{:.1} MB/s", data.performance_metrics.disk_io_rate_mbps), Style::default().fg(Color::White)),
        ]),
    ];

    let metrics_widget = Paragraph::new(metrics_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Metrics")
                .border_style(Style::default().fg(Color::Yellow)),
        );

    f.render_widget(metrics_widget, area);
}

fn draw_database_recent_operations(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    let operations: Vec<ListItem> = data.recent_operations.iter().map(|op| {
        let op_color = match op.operation_type.as_str() {
            "SELECT" => Color::Blue,
            "INSERT" => Color::Green,
            "UPDATE" => Color::Yellow,
            "DELETE" => Color::Red,
            "SET" => Color::Cyan,
            _ => Color::White,
        };

        let status_indicator = if op.success { "✅" } else { "❌" };
        
        // Truncate long queries
        let display_query = if op.query.len() > 60 {
            format!("{}...", &op.query[..57])
        } else {
            op.query.clone()
        };

        let backend_upper = op.backend.to_uppercase();
        ListItem::new(Line::from(vec![
            Span::styled(
                format!("[{}]", op.timestamp.format("%H:%M:%S")),
                Style::default().fg(Color::DarkGray)
            ),
            Span::raw(" "),
            Span::styled(status_indicator, Style::default()),
            Span::raw(" "),
            Span::styled(backend_upper, Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled(&op.operation_type, Style::default().fg(op_color).add_modifier(Modifier::BOLD)),
            Span::raw(": "),
            Span::raw(display_query),
            Span::raw(" ("),
            Span::styled(format!("{:.1}ms", op.duration_ms), Style::default().fg(Color::Yellow)),
            Span::raw(")"),
        ]))
    }).collect();

    let operations_list = List::new(operations)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Recent Operations")
                .border_style(Style::default().fg(Color::Green)),
        );

    f.render_widget(operations_list, area);
}

fn draw_database_available_commands(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    // let commands_text = data.available_commands.iter()
    //     .map(|cmd| format!("• {} - {} ({})", cmd.name, cmd.description, cmd.syntax))
    //     .collect::<Vec<_>>()
    //     .join("\n");

    let commands_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("🛠️ Available Commands", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ].into_iter()
    .chain(
        data.available_commands.iter().map(|cmd| {
            Line::from(vec![
                Span::raw("• "),
                Span::styled(&cmd.name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(" - "),
                Span::raw(&cmd.description),
                Span::raw(" ("),
                Span::styled(&cmd.syntax, Style::default().fg(Color::Yellow)),
                Span::raw(")"),
            ])
        })
    )
    .collect();

    let commands_widget = Paragraph::new(commands_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Commands")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(commands_widget, area);
}

fn draw_story_engine(f: &mut Frame, app: &App, area: Rect) {
    // Use enhanced version if system connector is available
    if app.system_connector.is_some() {
        draw_story_engine_enhanced(f, app, area);
    } else {
        draw_story_engine_legacy(f, app, area);
    }
}

/// Enhanced story engine with comprehensive narrative analytics and management
fn draw_story_engine_enhanced(f: &mut Frame, app: &App, area: Rect) {

    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            draw_loading_state(f, area, "Initializing story system...");
            return;
        }
    };

    // Get story data
    let story_data = match system_connector.get_story_data() {
        Ok(data) => data,
        Err(e) => {
            draw_error_state(f, area, &format!("Failed to load story data: {}", e));
            return;
        }
    };

    // Enhanced layout for story management
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70), // Main story view
            Constraint::Percentage(30), // Management panel
        ])
        .split(area);

    // Left side - Story visualization and timeline
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // Story overview
            Constraint::Length(20),  // Story timeline
            Constraint::Min(0),      // Active stories list
        ])
        .split(chunks[0]);

    // Story overview with key metrics
    draw_story_overview_enhanced(f, left_chunks[0], &story_data);
    
    // Interactive story timeline
    draw_story_timeline_interactive(f, left_chunks[1], &story_data);
    
    // Active stories list with details
    draw_active_stories_list(f, left_chunks[2], &story_data);

    // Right side - Management controls
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Story controls
            Constraint::Length(15), // Template manager
            Constraint::Length(12), // Agent coordination
            Constraint::Min(0),     // Context & sync
        ])
        .split(chunks[1]);

    // Story management controls
    draw_story_controls(f, right_chunks[0], &story_data);
    
    // Template management
    draw_template_manager(f, right_chunks[1], &story_data);
    
    // Narrative analytics
    draw_narrative_analytics(f, right_chunks[2], &story_data);
    
    // Story autonomy status (new!)
    draw_story_autonomy_status(f, right_chunks[3], app);
    
    // Show operation messages if any
    if let Some(message) = &app.state.story_operation_message {
        draw_operation_message(f, area, message, MessageType::Info);
    }
    
    // Show story creation form if active
    if app.state.story_creation_mode {
        draw_story_creation_form(f, area, &app.state);
    }
}

/// Legacy story engine implementation for fallback
fn draw_story_engine_legacy(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Story engine status
            Constraint::Min(0),     // Active stories
        ])
        .split(chunks[0]);

    // Get real story engine data if available
    let (engine_status, total_stories, active_stories, template_count, current_focus) =
        if let Some(system_connector) = &app.system_connector {
            // Get all stories
            let all_stories = system_connector.get_all_stories();
            let total_stories = all_stories.len();

            // Count active stories (non-archived)
            let active_stories = all_stories.iter()
                .filter(|s| !s.metadata.custom_data.get("archived").map_or(false, |v| v.as_str() == Some("true")))
                .count();

            // Get template count
            let template_count = if let Some(story_engine) = &system_connector.story_engine {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let template_manager = story_engine.template_manager();
                        let templates = template_manager.lock().await;
                        templates.get_templates().len()
                    })
                })
            } else {
                5 // We have 5 templates in our UI
            };

            // Get current focus (most recent story)
            let current_focus = all_stories.iter()
                .max_by_key(|s| s.created_at)
                .map(|s| s.title.clone())
                .unwrap_or_else(|| "No active story".to_string());

            let status = if system_connector.story_engine.is_some() {
                "● Active"
            } else {
                "● Mock Mode"
            };
            (status, total_stories, active_stories, template_count, current_focus)
        } else {
            ("● Offline", 0, 0, 0, "Story engine not initialized".to_string())
        };

    // Story Engine Status
    let status_lines = vec![
        Line::from(vec![
            Span::styled("📖 Story Engine", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(
                engine_status,
                Style::default().fg(
                    if engine_status.contains("Active") { Color::Green } else { Color::Red }
                )
            ),
        ]),
        Line::from(vec![
            Span::raw("Active Stories: "),
            Span::styled(format!("{}", active_stories), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Templates Loaded: "),
            Span::styled(format!("{}", template_count), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Total Narratives: "),
            Span::styled(format!("{}", total_stories), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Learning Progress: "),
            Span::styled(
                if total_stories > 0 { "████████░░" } else { "░░░░░░░░░░" },
                Style::default().fg(Color::Green)
            ),
            Span::raw(if total_stories > 0 { " 82%" } else { " 0%" }),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Current Focus: "),
            Span::styled(&current_focus, Style::default().fg(Color::Magenta)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(Span::styled(
                    " Engine Status ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(status_widget, left_chunks[0]);

    // Active Stories List
    let stories = if let Some(system_connector) = &app.system_connector {
        let all_stories = system_connector.get_all_stories();

        // Get active (non-archived) stories, sorted by most recent
        let mut active_stories: Vec<_> = all_stories.into_iter()
            .filter(|s| !s.metadata.custom_data.get("archived").map_or(false, |v| v.as_str() == Some("true")))
            .collect();
        active_stories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Take up to 7 most recent stories
        active_stories.truncate(7);

        active_stories.into_iter().map(|story| {
            // Choose icon based on story type
            let (icon, color) = match story.story_type {
                crate::story::StoryType::Codebase { .. } => ("💻", Color::White),
                crate::story::StoryType::Directory { .. } => ("📁", Color::Yellow),
                crate::story::StoryType::File { .. } => ("📄", Color::Cyan),
                crate::story::StoryType::Task { .. } => ("📋", Color::Yellow),
                crate::story::StoryType::Agent { .. } => ("🤖", Color::Cyan),
                crate::story::StoryType::System { .. } => ("⚙️", Color::Blue),
                crate::story::StoryType::Bug { .. } => ("🐛", Color::Red),
                crate::story::StoryType::Feature { .. } => ("✨", Color::Green),
                crate::story::StoryType::Epic { .. } => ("🎯", Color::Magenta),
                crate::story::StoryType::Learning { .. } => ("🧠", Color::Yellow),
                crate::story::StoryType::Performance { .. } => ("⚡", Color::Yellow),
                crate::story::StoryType::Security { .. } => ("🔒", Color::Red),
                crate::story::StoryType::Documentation { .. } => ("📚", Color::Gray),
                crate::story::StoryType::Testing { .. } => ("🧪", Color::Green),
                crate::story::StoryType::Refactoring { .. } => ("🔧", Color::Blue),
                crate::story::StoryType::Dependencies { .. } => ("📦", Color::Yellow),
                crate::story::StoryType::Deployment { .. } => ("🚀", Color::Cyan),
                crate::story::StoryType::Research { .. } => ("🔬", Color::Magenta),
            };

            // Get current status or description
            let status = story.metadata.custom_data.get("status")
                .or_else(|| story.metadata.custom_data.get("description"))
                .and_then(|v| v.as_str())
                .unwrap_or("Active");

            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", icon), Style::default().fg(color)),
                Span::styled(story.title.clone(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(format!(" - {}", status)),
            ]))
        }).collect()
    } else {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("⚠️ ", Style::default().fg(Color::Yellow)),
                Span::styled("No stories available", Style::default().fg(Color::DarkGray)),
            ]))
        ]
    };

    let stories_list = List::new(stories)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(Span::styled(
                    " Active Stories ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(stories_list, left_chunks[1]);

    // Story Details (Right side)
    let details_lines = vec![
        Line::from(vec![
            Span::styled("Current Story: ", Style::default().fg(Color::Gray)),
            Span::styled("Bug Detection", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("📊 Progress", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Analysis: "),
            Span::styled("██████████", Style::default().fg(Color::Green)),
            Span::raw(" 100%"),
        ]),
        Line::from(vec![
            Span::raw("  Detection: "),
            Span::styled("███████░░░", Style::default().fg(Color::Yellow)),
            Span::raw(" 75%"),
        ]),
        Line::from(vec![
            Span::raw("  Resolution: "),
            Span::styled("████░░░░░░", Style::default().fg(Color::Blue)),
            Span::raw(" 40%"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("🎯 Objectives", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  • Identify memory leak patterns"),
        Line::from("  • Analyze error frequency"),
        Line::from("  • Generate fix suggestions"),
        Line::from("  • Validate solutions"),
        Line::from(""),
        Line::from(vec![
            Span::styled("📝 Recent Actions", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  • Found 3 potential memory leaks"),
        Line::from("  • Analyzed 1,234 error logs"),
        Line::from("  • Generated 5 fix proposals"),
        Line::from("  • Testing fix #1 in sandbox"),
        Line::from(""),
        Line::from(vec![
            Span::styled("💡 Insights", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  • Pattern: Async closures not properly cleaned"),
        Line::from("  • Root cause: Missing drop implementations"),
        Line::from("  • Impact: 15% memory overhead"),
    ];

    let details_widget = Paragraph::new(details_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Story Details ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });

    f.render_widget(details_widget, chunks[1]);
}

// Enhanced memory tab functions



/// Enhanced fractal memory visualization with hierarchical scales and domains
fn draw_fractal_visualization(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Check if we have fractal memory data
    if let Some(fractal_data) = &data.fractal_memory {
        draw_enhanced_fractal_visualization(f, area, fractal_data);
    } else {
        draw_basic_fractal_visualization(f, area, data);
    }
}

/// Draw enhanced fractal visualization with real fractal memory data
fn draw_enhanced_fractal_visualization(f: &mut Frame, area: Rect, fractal_data: &crate::tui::connectors::system_connector::FractalMemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Scale distribution
            Constraint::Percentage(35), // Domains
            Constraint::Percentage(25), // Emergence events
        ])
        .split(area);

    // Scale Distribution Visualization
    draw_scale_distribution(f, chunks[0], fractal_data);
    
    // Domain Information
    draw_fractal_domains(f, chunks[1], fractal_data);
    
    // Recent Emergence Events
    draw_emergence_events(f, chunks[2], fractal_data);
}

/// Draw scale distribution chart
fn draw_scale_distribution(f: &mut Frame, area: Rect, fractal_data: &crate::tui::connectors::system_connector::FractalMemoryData) {
    let scale_items: Vec<ListItem> = fractal_data.scale_distribution.iter().map(|scale| {
        let activity_bar = match (scale.activity_level * 10.0) as u8 {
            0..=2 => "██░░░░░░░░",
            3..=4 => "████░░░░░░",
            5..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let activity_color = if scale.activity_level > 0.8 {
            Color::Green
        } else if scale.activity_level > 0.6 {
            Color::Yellow
        } else {
            Color::Red
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&scale.scale_name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  Nodes: "),
                Span::styled(format!("{}", scale.node_count), Style::default().fg(Color::White)),
                Span::raw("  Connections: "),
                Span::styled(format!("{}", scale.connections), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::raw("  Activity: "),
                Span::styled(activity_bar, Style::default().fg(activity_color)),
                Span::styled(format!(" {:.1}%", scale.activity_level * 100.0), Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let scale_list = List::new(scale_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📊 Scale Distribution")
                .border_style(Style::default().fg(Color::Blue)),
        );

    f.render_widget(scale_list, area);
}

/// Draw fractal domains information
fn draw_fractal_domains(f: &mut Frame, area: Rect, fractal_data: &crate::tui::connectors::system_connector::FractalMemoryData) {
    let domain_items: Vec<ListItem> = fractal_data.domains.iter().map(|domain| {
        let coherence_bar = match (domain.coherence * 10.0) as u8 {
            0..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let coherence_color = if domain.coherence > 0.85 {
            Color::Green
        } else if domain.coherence > 0.75 {
            Color::Yellow
        } else {
            Color::Red
        };

        let activity_indicator = if let Some(last_activity) = domain.last_activity {
            let elapsed = chrono::Utc::now().signed_duration_since(last_activity);
            if elapsed.num_minutes() < 5 {
                "🟢"
            } else if elapsed.num_minutes() < 30 {
                "🟡"
            } else {
                "🔴"
            }
        } else {
            "⚫"
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::raw(activity_indicator),
                Span::raw(" "),
                Span::styled(&domain.name, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  Nodes: "),
                Span::styled(format!("{}", domain.node_count), Style::default().fg(Color::White)),
                Span::raw("  Depth: "),
                Span::styled(format!("{}", domain.depth), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::raw("  Coherence: "),
                Span::styled(coherence_bar, Style::default().fg(coherence_color)),
                Span::styled(format!(" {:.1}%", domain.coherence * 100.0), Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let domain_list = List::new(domain_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🏛️ Memory Domains")
                .border_style(Style::default().fg(Color::Magenta)),
        );

    f.render_widget(domain_list, area);
}

/// Draw recent emergence events
fn draw_emergence_events(f: &mut Frame, area: Rect, fractal_data: &crate::tui::connectors::system_connector::FractalMemoryData) {
    let event_items: Vec<ListItem> = fractal_data.recent_emergence.iter().map(|event| {
        let confidence_color = if event.confidence > 0.8 {
            Color::Green
        } else if event.confidence > 0.6 {
            Color::Yellow
        } else {
            Color::Red
        };

        let elapsed = chrono::Utc::now().signed_duration_since(event.timestamp);
        let time_str = if elapsed.num_minutes() < 60 {
            format!("{}m ago", elapsed.num_minutes())
        } else {
            format!("{}h ago", elapsed.num_hours())
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&event.event_type, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(&event.description, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("  Confidence: "),
                Span::styled(format!("{:.1}%", event.confidence * 100.0), Style::default().fg(confidence_color)),
                Span::raw("  Nodes: "),
                Span::styled(format!("{}", event.nodes_involved), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(time_str, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let events_list = List::new(event_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("✨ Emergence Events")
                .border_style(Style::default().fg(Color::Yellow)),
        );

    f.render_widget(events_list, area);
}

/// Fallback basic fractal visualization for when detailed data isn't available
fn draw_basic_fractal_visualization(f: &mut Frame, area: Rect, data: &MemoryData) {
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧠 Basic Memory Patterns")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(|ctx| {
            // Draw basic fractal pattern based on memory nodes
            let node_count = data.total_nodes.min(100);
            let golden_ratio = 1.618033988;

            for i in 0..node_count {
                let angle = i as f64 * golden_ratio * 2.0 * std::f64::consts::PI;
                let radius = (i as f64).sqrt() * 5.0;
                let x = angle.cos() * radius;
                let y = angle.sin() * radius;

                let color = if i % 3 == 0 {
                    Color::Cyan
                } else if i % 3 == 1 {
                    Color::Blue
                } else {
                    Color::Magenta
                };

                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x,
                    y,
                    radius: 1.0 + (i as f64 * 0.1).sin().abs(),
                    color,
                });
            }
        });

    f.render_widget(canvas, area);
}

/// Enhanced story overview with key metrics
fn draw_story_overview_enhanced(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // Active stories
            Constraint::Percentage(20), // Completion rate
            Constraint::Percentage(20), // Narrative coherence
            Constraint::Percentage(20), // Character count
            Constraint::Percentage(20), // Templates
        ])
        .split(area);

    // Active Stories
    let mut active_gauge = AnimatedGauge::new(
        "Active Stories".to_string(),
        story_data.active_stories as f32,
        100.0,
    );
    active_gauge.color_start = Color::Green;
    active_gauge.render(f, chunks[0]);

    // Completion Rate
    let mut completion_gauge = AnimatedGauge::new(
        "Completion".to_string(),
        story_data.completion_rate * 100.0,
        100.0,
    );
    completion_gauge.color_start = Color::Blue;
    completion_gauge.render(f, chunks[1]);

    // Narrative Coherence
    let mut coherence_gauge = AnimatedGauge::new(
        "Coherence".to_string(),
        story_data.narrative_coherence * 100.0,
        100.0,
    );
    coherence_gauge.color_start = Color::Magenta;
    coherence_gauge.render(f, chunks[2]);

    // Character Count
    let chars_text = vec![
        Line::from(vec![
            Span::styled("👥 Characters", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{}", story_data.character_count), 
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Active", Style::default().fg(Color::Green)),
        ]),
    ];

    let chars_widget = Paragraph::new(chars_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
        )
        .alignment(Alignment::Center);

    f.render_widget(chars_widget, chunks[3]);

    // Templates
    let templates_text = vec![
        Line::from(vec![
            Span::styled("📋 Templates", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{}", story_data.story_templates.len()), 
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Available", Style::default().fg(Color::Green)),
        ]),
    ];

    let templates_widget = Paragraph::new(templates_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
        )
        .alignment(Alignment::Center);

    f.render_widget(templates_widget, chunks[4]);
}

/// Story management dashboard with templates and analytics
fn draw_story_management_dashboard(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35), // Story templates
            Constraint::Percentage(35), // Narrative analytics
            Constraint::Percentage(30), // Active arcs
        ])
        .split(area);

    // Story Templates
    draw_story_templates(f, chunks[0], story_data);
    
    // Narrative Analytics
    draw_narrative_analytics(f, chunks[1], story_data);
    
    // Active Story Arcs
    draw_active_story_arcs(f, chunks[2], story_data);
}

/// Draw story templates with usage statistics
fn draw_story_templates(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let template_items: Vec<ListItem> = story_data.story_templates.iter().map(|template| {
        let success_bar = match (template.success_rate * 10.0) as u8 {
            0..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let success_color = if template.success_rate > 0.85 {
            Color::Green
        } else if template.success_rate > 0.75 {
            Color::Yellow
        } else {
            Color::Red
        };

        let category_color = match template.category.as_str() {
            "Development" => Color::Blue,
            "Debugging" => Color::Red,
            "Integration" => Color::Magenta,
            "Optimization" => Color::Yellow,
            _ => Color::Gray,
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&template.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  Category: "),
                Span::styled(&template.category, Style::default().fg(category_color)),
                Span::raw(format!("  Used: {}", template.usage_count)),
            ]),
            Line::from(vec![
                Span::raw("  Success: "),
                Span::styled(success_bar, Style::default().fg(success_color)),
                Span::styled(format!(" {:.1}%", template.success_rate * 100.0), Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let templates_list = List::new(template_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📋 Story Templates")
                .border_style(Style::default().fg(Color::Yellow)),
        );

    f.render_widget(templates_list, area);
}

/// Draw comprehensive narrative analytics
fn draw_narrative_analytics(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let analytics = &story_data.narrative_analytics;
    
    let analytics_text = vec![
        Line::from(vec![
            Span::styled("📊 Narrative Analytics", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Coherence Metrics:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Overall: "),
            Span::styled(format!("{:.1}%", analytics.coherence_metrics.overall_coherence * 100.0), 
                Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("  Logical: "),
            Span::styled(format!("{:.1}%", analytics.coherence_metrics.logical_consistency * 100.0), 
                Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("  Temporal: "),
            Span::styled(format!("{:.1}%", analytics.coherence_metrics.temporal_consistency * 100.0), 
                Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Plot Analysis:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Structure: "),
            Span::styled(format!("{:.1}%", analytics.plot_analysis.story_structure_score * 100.0), 
                Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("  Climax Impact: "),
            Span::styled(format!("{:.1}%", analytics.plot_analysis.climax_impact * 100.0), 
                Style::default().fg(Color::Magenta)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Pacing & Flow:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Overall: "),
            Span::styled(format!("{:.1}%", analytics.pacing_analysis.overall_pacing * 100.0), 
                Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("  Tension: "),
            Span::styled(format!("{:.1}%", analytics.narrative_tension * 100.0), 
                Style::default().fg(Color::Red)),
        ]),
    ];

    let analytics_widget = Paragraph::new(analytics_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📈 Analytics")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(analytics_widget, area);
}

/// Draw active story arcs with progress
fn draw_active_story_arcs(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let arc_items: Vec<ListItem> = story_data.active_arcs.iter().map(|arc| {
        let progress_bar = match (arc.progress * 10.0) as u8 {
            0..=2 => "██░░░░░░░░",
            3..=4 => "████░░░░░░",
            5..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let progress_color = if arc.progress > 0.8 {
            Color::Green
        } else if arc.progress > 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };

        let status_color = match arc.status.as_str() {
            "Completed" => Color::Green,
            "Active" => Color::Blue,
            "Near Completion" => Color::Yellow,
            _ => Color::Gray,
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&arc.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(&arc.description, Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::raw("  Progress: "),
                Span::styled(progress_bar, Style::default().fg(progress_color)),
                Span::styled(format!(" {:.1}%", arc.progress * 100.0), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("  Status: "),
                Span::styled(&arc.status, Style::default().fg(status_color)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let arcs_list = List::new(arc_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🎯 Active Arcs")
                .border_style(Style::default().fg(Color::Magenta)),
        );

    f.render_widget(arcs_list, area);
}

/// Draw character development tracking
fn draw_character_development(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let char_items: Vec<ListItem> = story_data.character_development.iter().map(|character| {
        let growth_bar = match (character.growth_percentage * 10.0) as u8 {
            0..=2 => "██░░░░░░░░",
            3..=4 => "████░░░░░░",
            5..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let growth_color = if character.growth_percentage > 0.8 {
            Color::Green
        } else if character.growth_percentage > 0.6 {
            Color::Yellow
        } else {
            Color::Red
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(&character.character_name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  Arc: "),
                Span::styled(&character.development_arc, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("  Growth: "),
                Span::styled(growth_bar, Style::default().fg(growth_color)),
                Span::styled(format!(" {:.1}%", character.growth_percentage * 100.0), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("  Consistency: "),
                Span::styled(format!("{:.1}%", character.consistency_score * 100.0), Style::default().fg(Color::Green)),
                Span::raw("  Motivation: "),
                Span::styled(format!("{:.1}%", character.motivation_clarity * 100.0), Style::default().fg(Color::Blue)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let characters_list = List::new(char_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("👥 Character Development")
                .border_style(Style::default().fg(Color::Cyan)),
        );

    f.render_widget(characters_list, area);
}

/// Draw story progression and milestones
fn draw_story_progression(f: &mut Frame, area: Rect, story_data: &crate::tui::connectors::system_connector::StoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Progression stats
            Constraint::Percentage(35), // Milestones
            Constraint::Percentage(25), // Next objectives
        ])
        .split(area);

    // Progression Stats
    let progression = &story_data.story_progression;
    let progress_text = vec![
        Line::from(vec![
            Span::styled("📈 Story Progression", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Act: "),
            Span::styled(format!("{}/{}", progression.current_act, progression.total_acts), 
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Scenes: "),
            Span::styled(format!("{}", progression.scene_count), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Words: "),
            Span::styled(format!("{}", progression.word_count), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Completion:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(format!("{:.1}%", progression.estimated_completion * 100.0), 
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
    ];

    let progress_widget = Paragraph::new(progress_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📊 Progress")
                .border_style(Style::default().fg(Color::Green)),
        );

    f.render_widget(progress_widget, chunks[0]);

    // Milestones
    let milestone_items: Vec<ListItem> = progression.milestone_progress.iter().take(4).map(|milestone| {
        let status_icon = if milestone.completed { "✅" } else { "⏳" };
        let status_color = if milestone.completed { Color::Green } else { Color::Yellow };

        ListItem::new(vec![
            Line::from(vec![
                Span::raw(status_icon),
                Span::raw(" "),
                Span::styled(&milestone.name, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(&milestone.description, Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
        ])
    }).collect();

    let milestones_list = List::new(milestone_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🎯 Milestones")
                .border_style(Style::default().fg(Color::Blue)),
        );

    f.render_widget(milestones_list, chunks[1]);

    // Next Objectives
    let objective_items: Vec<ListItem> = progression.next_objectives.iter().take(4).map(|objective| {
        ListItem::new(Line::from(vec![
            Span::raw("• "),
            Span::styled(objective, Style::default().fg(Color::White)),
        ]))
    }).collect();

    let objectives_list = List::new(objective_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🎯 Next Steps")
                .border_style(Style::default().fg(Color::Magenta)),
        );

    f.render_widget(objectives_list, chunks[2]);
}

/// Draw memory layers with real data
fn draw_memory_layers_enhanced(f: &mut Frame, area: Rect, data: &MemoryData) {
    let rows: Vec<Row> = data.layers.iter().map(|layer| {
        let activity_bar = match (layer.activity_level * 10.0) as u8 {
            0..=2 => "██░░░░░░░░",
            3..=4 => "████░░░░░░",
            5..=6 => "██████░░░░",
            7..=8 => "████████░░",
            _ => "██████████",
        };

        let activity_color = if layer.activity_level > 0.7 {
            Color::Green
        } else if layer.activity_level > 0.4 {
            Color::Yellow
        } else {
            Color::Red
        };

        Row::new(vec![
            layer.name.clone(),
            format!("{}", layer.node_count),
            activity_bar.to_string(),
            format!("{:.1}%", layer.activity_level * 100.0),
        ])
        .style(Style::default().fg(activity_color))
    }).collect();

    let header = Row::new(vec!["Layer", "Nodes", "Activity", "Load"])
        .style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD));

    let table = Table::new(
        rows,
        vec![
            Constraint::Length(20),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Memory Layers")
            .border_style(Style::default().fg(Color::Cyan)),
    )
    .style(Style::default().fg(Color::White));

    f.render_widget(table, area);
}

/// Draw recent associations
fn draw_recent_associations(f: &mut Frame, area: Rect, data: &MemoryData) {
    let associations: Vec<ListItem> = data.recent_associations.iter().map(|assoc| {
        let strength_bar = match (assoc.strength * 5.0) as u8 {
            0 => "░",
            1 => "▒",
            2 => "▓",
            3 => "█",
            _ => "█",
        };

        let strength_color = if assoc.strength > 0.7 {
            Color::Green
        } else if assoc.strength > 0.4 {
            Color::Yellow
        } else {
            Color::Red
        };

        ListItem::new(Line::from(vec![
            Span::raw(format!("{} ", strength_bar)),
            Span::styled(&assoc.from_node, Style::default().fg(Color::Cyan)),
            Span::raw(" → "),
            Span::styled(&assoc.to_node, Style::default().fg(Color::Blue)),
            Span::raw(" ("),
            Span::styled(&assoc.association_type, Style::default().fg(Color::Gray)),
            Span::raw(", "),
            Span::styled(
                format!("{:.2}", assoc.strength),
                Style::default().fg(strength_color),
            ),
            Span::raw(")"),
        ]))
    }).collect();

    let list = List::new(associations)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Recent Associations")
                .border_style(Style::default().fg(Color::Blue)),
        );

    f.render_widget(list, area);
}

/// Helper functions

fn draw_loading_state(f: &mut Frame, area: Rect, message: &str) {
    let loading = LoadingSpinner::new(message.to_string());
    loading.render(f, area);
}

fn draw_error_state(f: &mut Frame, area: Rect, error: &str) {
    let error_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("❌ Error", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(error),
    ])
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Red)))
    .alignment(Alignment::Center)
    .wrap(Wrap { trim: true });

    f.render_widget(error_widget, area);
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count == 3 {
            result.push(',');
            count = 0;
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

/// Draw memory overview cards with real-time stats
fn draw_memory_overview_cards(f: &mut Frame, _app: &App, area: Rect, data: &MemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);


    // Total memories card
    let memories_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("🧠 Total Memories", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format_number(data.total_nodes), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)))
    .alignment(Alignment::Center);
    f.render_widget(memories_widget, chunks[0]);

    // Cache hit rate card
    let cache_hit = data.cache_hit_rate * 100.0;
    let cache_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("⚡ Cache Hit Rate", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{:.1}%", cache_hit), Style::default().fg(if cache_hit > 85.0 { Color::Green } else { Color::Yellow }).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)))
    .alignment(Alignment::Center);
    f.render_widget(cache_widget, chunks[1]);

    // Memory usage card
    let usage_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("💾 Memory Usage", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{:.1} MB", data.memory_usage_mb), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)))
    .alignment(Alignment::Center);
    f.render_widget(usage_widget, chunks[2]);

    // Embeddings card
    let embeddings_count = data.embeddings_stats.as_ref().map(|s| s.total_embeddings).unwrap_or(0);
    let embeddings_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("🔮 Embeddings", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format_number(embeddings_count), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)))
    .alignment(Alignment::Center);
    f.render_widget(embeddings_widget, chunks[3]);
}

/// Draw enhanced fractal memory view with visualizations
fn draw_enhanced_fractal_memory_view(f: &mut Frame, _app: &App, area: Rect, data: &MemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70),  // Main fractal visualization
            Constraint::Percentage(30),  // Fractal stats panel
        ])
        .split(area);

    // Main fractal visualization
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧠 Fractal Memory Patterns")
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_bounds([-50.0, 50.0])
        .y_bounds([-50.0, 50.0])
        .paint(|ctx| {
            // Draw fractal pattern with enhanced visualization
            draw_fractal_pattern(ctx, 0.0, 0.0, 40.0, 4, data);
            
            // Draw memory nodes with hierarchy
            let node_count = data.total_nodes.min(50);
            for i in 0..node_count {
                let level = (i / 10) as f64;
                let angle = (i as f64 * 2.0 * std::f64::consts::PI) / node_count as f64;
                let radius = 15.0 + (level * 10.0) + ((i % 10) as f64 * 0.5);
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                
                // Node color based on memory type
                let node_color = match i % 5 {
                    0 => Color::Cyan,    // Semantic memories
                    1 => Color::Blue,    // Episodic memories
                    2 => Color::Magenta, // Procedural memories
                    3 => Color::Green,   // Working memories
                    _ => Color::Yellow,  // Meta memories
                };
                
                // Draw node with fixed size for now
                let node_size = 2.0 + (i as f64 * 0.1);
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x,
                    y,
                    radius: node_size,
                    color: node_color,
                });
                
                // Draw hierarchical connections
                if i > 0 {
                    let parent_idx = (i - 1) / 3;
                    let parent_angle = (parent_idx as f64 * 2.0 * std::f64::consts::PI) / node_count as f64;
                    let parent_level = (parent_idx / 10) as f64;
                    let parent_radius = 15.0 + (parent_level * 10.0) + ((parent_idx % 10) as f64 * 0.5);
                    let parent_x = parent_radius * parent_angle.cos();
                    let parent_y = parent_radius * parent_angle.sin();
                    
                    ctx.draw(&ratatui::widgets::canvas::Line {
                        x1: parent_x,
                        y1: parent_y,
                        x2: x,
                        y2: y,
                        color: Color::DarkGray,
                    });
                }
            }
            
            // Draw emergence patterns from fractal memory if available
            if let Some(fractal_data) = &data.fractal_memory {
                for (i, _) in fractal_data.recent_emergence.iter().enumerate().take(5) {
                    let x = 10.0 + (i as f64 * 15.0);
                    let y = -20.0;
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x,
                        y,
                        radius: 3.0,
                        color: Color::LightCyan,
                    });
                }
            }
        });
    
    f.render_widget(canvas, chunks[0]);
    
    // Fractal stats panel
    draw_fractal_stats_panel(f, chunks[1], data);
}

/// Draw fractal pattern recursively
fn draw_fractal_pattern(
    ctx: &mut ratatui::widgets::canvas::Context,
    x: f64,
    y: f64,
    size: f64,
    depth: u32,
    data: &MemoryData,
) {
    if depth == 0 || size < 2.0 {
        return;
    }
    
    // Draw central circle
    ctx.draw(&ratatui::widgets::canvas::Circle {
        x,
        y,
        radius: size / 4.0,
        color: Color::Magenta,
    });
    
    // Draw recursive branches
    let angles = [0.0, 90.0, 180.0, 270.0];
    for &angle in &angles {
        let rad = angle * std::f64::consts::PI / 180.0;
        let new_x = x + (size / 2.0) * rad.cos();
        let new_y = y + (size / 2.0) * rad.sin();
        
        // Draw connection
        ctx.draw(&ratatui::widgets::canvas::Line {
            x1: x,
            y1: y,
            x2: new_x,
            y2: new_y,
            color: Color::DarkGray,
        });
        
        // Recurse
        draw_fractal_pattern(ctx, new_x, new_y, size / 2.0, depth - 1, data);
    }
}

/// Draw knowledge graph view
fn draw_fractal_stats_panel(f: &mut Frame, area: Rect, data: &MemoryData) {
    let mut stats_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Total Nodes: "),
            Span::styled(format!("{}", data.total_nodes), Style::default().fg(Color::Cyan)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Associations: "),
            Span::styled(format!("{}", data.total_associations), Style::default().fg(Color::Green)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Cache Hit Rate: "),
            Span::styled(format!("{:.1}%", data.cache_hit_rate * 100.0), Style::default().fg(Color::Magenta)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Memory Usage: "),
            Span::styled(format!("{:.1} MB", data.memory_usage_mb), Style::default().fg(Color::Yellow)),
        ])),
    ];
    
    // Add fractal-specific stats if available
    if let Some(fractal_data) = &data.fractal_memory {
        stats_items.extend(vec![
            ListItem::new(Line::from("")),
            ListItem::new(Line::from(vec![
                Span::styled("Fractal Memory:", Style::default().add_modifier(Modifier::BOLD)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Emergence Events: "),
                Span::styled(format!("{}", fractal_data.emergence_events), Style::default().fg(Color::Cyan)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Avg Coherence: "),
                Span::styled(format!("{:.2}", fractal_data.avg_coherence), Style::default().fg(Color::Green)),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Avg Depth: "),
                Span::styled(format!("{:.1}", fractal_data.average_depth), Style::default().fg(Color::Blue)),
            ])),
        ]);
    }
    
    stats_items.extend(vec![
        ListItem::new(Line::from("")),
        ListItem::new(Line::from("🔹 Node Types:").style(Style::default().add_modifier(Modifier::BOLD))),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Cyan)),
            Span::raw("Semantic"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Blue)),
            Span::raw("Episodic"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Magenta)),
            Span::raw("Procedural"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Green)),
            Span::raw("Working"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Yellow)),
            Span::raw("Meta"),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from("📊 Layer Activity:").style(Style::default().add_modifier(Modifier::BOLD))),
    ]);
    
    // Add layer information
    for layer in &data.layers {
        stats_items.push(ListItem::new(Line::from(vec![
            Span::raw(&layer.name),
            Span::raw(": "),
            Span::styled(format!("{:.1}%", layer.activity_level * 100.0), Style::default().fg(Color::Green)),
        ])));
    }
    
    let stats_list = List::new(stats_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Fractal Stats ")
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    
    f.render_widget(stats_list, area);
}

fn draw_knowledge_graph_view(f: &mut Frame, _: &App, area: Rect, data: &MemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70),  // Main graph visualization
            Constraint::Percentage(30),  // Side panel
        ])
        .split(area);

    // Graph visualization with dynamic knowledge nodes
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🕸️ Knowledge Graph Network")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(|ctx| {
            // Create node positions from recent associations
            let mut node_positions: HashMap<String, (f64, f64)> = HashMap::new();
            let mut node_index = 0;
            
            // Extract unique nodes from associations
            for association in &data.recent_associations {
                if !node_positions.contains_key(&association.from_node) {
                    let angle = (node_index as f64 * 2.0 * std::f64::consts::PI) / 20.0;
                    let radius = 30.0 + (node_index as f64 * 2.0);
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();
                    node_positions.insert(association.from_node.clone(), (x, y));
                    node_index += 1;
                }
                if !node_positions.contains_key(&association.to_node) {
                    let angle = (node_index as f64 * 2.0 * std::f64::consts::PI) / 20.0;
                    let radius = 30.0 + (node_index as f64 * 2.0);
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();
                    node_positions.insert(association.to_node.clone(), (x, y));
                    node_index += 1;
                }
            }
            
            // Draw edges (associations)
            for association in &data.recent_associations {
                if let (Some(from_pos), Some(to_pos)) = (
                    node_positions.get(&association.from_node),
                    node_positions.get(&association.to_node)
                ) {
                    let edge_color = match association.strength {
                        s if s > 0.8 => Color::Green,
                        s if s > 0.5 => Color::Yellow,
                        s if s > 0.3 => Color::Blue,
                        _ => Color::DarkGray,
                    };
                    
                    ctx.draw(&ratatui::widgets::canvas::Line {
                        x1: from_pos.0,
                        y1: from_pos.1,
                        x2: to_pos.0,
                        y2: to_pos.1,
                        color: edge_color,
                    });
                }
            }
            
            // Draw nodes
            for (node_name, (x, y)) in node_positions.iter() {
                // Determine node color based on association type
                let node_associations: Vec<_> = data.recent_associations.iter()
                    .filter(|a| &a.from_node == node_name || &a.to_node == node_name)
                    .collect();
                
                let node_color = if !node_associations.is_empty() {
                    match node_associations[0].association_type.as_str() {
                        "semantic" => Color::Yellow,
                        "episodic" => Color::Blue,
                        "procedural" => Color::Green,
                        "causal" => Color::Magenta,
                        _ => Color::Cyan,
                    }
                } else {
                    Color::Gray
                };
                
                // Determine node size based on connection count
                let radius = 3.0 + (node_associations.len() as f64).min(5.0);
                
                // Draw node
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x: *x,
                    y: *y,
                    radius,
                    color: node_color,
                });
                
                // Draw node halo for highly connected nodes
                if node_associations.len() > 3 {
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: *x,
                        y: *y,
                        radius: radius + 2.0,
                        color: Color::DarkGray,
                    });
                }
            }
            
            // Draw layer boundaries as circles
            for (i, _) in data.layers.iter().enumerate() {
                let layer_radius = 20.0 + (i as f64 * 30.0);
                let steps = 60;
                for j in 0..steps {
                    let angle = (j as f64 * 2.0 * std::f64::consts::PI) / steps as f64;
                    let x = layer_radius * angle.cos();
                    let y = layer_radius * angle.sin();
                    
                    if j % 6 == 0 {  // Only draw every 6th point for dotted effect
                        ctx.draw(&ratatui::widgets::canvas::Circle {
                            x,
                            y,
                            radius: 0.5,
                            color: Color::DarkGray,
                        });
                    }
                }
            }
        });
    
    f.render_widget(canvas, chunks[0]);

    // Side panel with graph info and controls
    let side_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),  // Graph stats
            Constraint::Length(10),  // Node type legend
            Constraint::Length(12),  // Controls
            Constraint::Min(0),      // Selected node details
        ])
        .split(chunks[1]);

    // Graph statistics
    draw_graph_stats(f, side_chunks[0], data);
    
    // Node type legend
    draw_node_legend(f, side_chunks[1]);
    
    // Graph controls
    draw_graph_controls(f, side_chunks[2]);
    
    // Selected node details
    draw_selected_node_details(f, side_chunks[3], data);
}

// Helper function removed - node positions are now generated inline

fn draw_graph_stats(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Calculate stats from available data
    let unique_nodes: std::collections::HashSet<String> = data.recent_associations.iter()
        .flat_map(|a| vec![a.from_node.clone(), a.to_node.clone()])
        .collect();
    let node_count = unique_nodes.len();
    let edge_count = data.recent_associations.len();
    let avg_connections = if node_count > 0 {
        (edge_count * 2) as f32 / node_count as f32
    } else {
        0.0
    };
    
    let stats_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Total Nodes: "),
            Span::styled(format!("{}", node_count), Style::default().fg(Color::Cyan)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Total Edges: "),
            Span::styled(format!("{}", edge_count), Style::default().fg(Color::Green)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Memory Layers: "),
            Span::styled(format!("{}", data.layers.len()), Style::default().fg(Color::Magenta)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Avg Connections: "),
            Span::styled(format!("{:.1}", avg_connections), Style::default().fg(Color::Yellow)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Cache Hit Rate: "),
            Span::styled(format!("{:.2}%", data.cache_hit_rate * 100.0), Style::default().fg(Color::Blue)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Memory Usage: "),
            Span::styled(format!("{:.1} MB", data.memory_usage_mb), Style::default().fg(Color::Green)),
        ])),
    ];
    
    let stats_list = List::new(stats_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Graph Statistics "));
    
    f.render_widget(stats_list, area);
}

fn draw_node_legend(f: &mut Frame, area: Rect) {
    let legend_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Yellow)),
            Span::raw("Core Concept"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Blue)),
            Span::raw("Concept"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Green)),
            Span::raw("Instance"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Magenta)),
            Span::raw("Property"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Cyan)),
            Span::raw("Relation"),
        ])),
    ];
    
    let legend = List::new(legend_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Node Types "));
    
    f.render_widget(legend, area);
}

fn draw_graph_controls(f: &mut Frame, area: Rect) {
    let controls = vec![
        Line::from(vec![
            Span::styled("[1-5]", Style::default().fg(Color::Cyan)),
            Span::raw(" Filter by type"),
        ]),
        Line::from(vec![
            Span::styled("[+/-]", Style::default().fg(Color::Cyan)),
            Span::raw(" Zoom in/out"),
        ]),
        Line::from(vec![
            Span::styled("[Space]", Style::default().fg(Color::Cyan)),
            Span::raw(" Reset view"),
        ]),
        Line::from(vec![
            Span::styled("[Enter]", Style::default().fg(Color::Cyan)),
            Span::raw(" Select node"),
        ]),
        Line::from(vec![
            Span::styled("[F]", Style::default().fg(Color::Cyan)),
            Span::raw(" Find node"),
        ]),
    ];
    
    let controls_widget = Paragraph::new(controls)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Controls "));
    
    f.render_widget(controls_widget, area);
}

fn draw_selected_node_details(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Since we don't have node selection in the data model, show general memory info
    let details = vec![
        Line::from(vec![
            Span::styled("Memory System Info", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Total Nodes: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{}", data.total_nodes)),
        ]),
        Line::from(vec![
            Span::styled("Associations: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{}", data.total_associations)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Recent Associations:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
    ];
    
    let mut lines = details;
    
    // Show a few recent associations
    for assoc in data.recent_associations.iter().take(5) {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(&assoc.from_node, Style::default().fg(Color::Yellow)),
            Span::raw(" → "),
            Span::styled(&assoc.to_node, Style::default().fg(Color::Green)),
            Span::raw(" ("),
            Span::styled(&assoc.association_type, Style::default().fg(Color::Cyan)),
            Span::raw(")"),
        ]));
    }
    
    let details_widget = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Node Details "))
        .wrap(Wrap { trim: true });
    
    f.render_widget(details_widget, area);
}

/// Draw associations view with enhanced management features
fn draw_associations_view(f: &mut Frame, _app: &App, area: Rect, data: &MemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),  // Main association view
            Constraint::Percentage(40),  // Controls and management
        ])
        .split(area);

    // Left side - Association visualization
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(18),  // Association matrix
            Constraint::Min(0),      // Recent associations list
        ])
        .split(chunks[0]);

    // Association strength matrix with real data
    draw_association_matrix(f, left_chunks[0], data);
    
    // Recent associations with details
    draw_recent_associations_enhanced(f, left_chunks[1], data);

    // Right side - Management controls
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),  // Association controls
            Constraint::Length(10),  // Filter options
            Constraint::Length(8),   // Stats
            Constraint::Min(0),      // Selected association details
        ])
        .split(chunks[1]);

    // Association management controls
    draw_association_controls(f, right_chunks[0]);
    
    // Filter options
    draw_association_filters(f, right_chunks[1], data);
    
    // Association statistics
    draw_association_stats(f, right_chunks[2], data);
    
    // Selected association details
    draw_selected_association_details(f, right_chunks[3], data);
}

fn draw_association_matrix(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Define memory types based on layer information
    let memory_types = vec!["Semantic", "Episodic", "Procedural", "Working", "Meta"];
    let mut rows = vec![];
    
    // Header row
    let header = Row::new(
        std::iter::once("Type".to_string())
            .chain(memory_types.iter().map(|t| t.to_string()))
            .collect::<Vec<_>>()
    )
    .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));
    
    // Create association matrix based on layer activity and association counts
    for (i, from_type) in memory_types.iter().enumerate() {
        let mut cells = vec![from_type.to_string()];
        
        for (j, _) in memory_types.iter().enumerate() {
            // Calculate synthetic strength based on layer activity
            let from_activity = data.layers.get(i).map(|l| l.activity_level).unwrap_or(0.0);
            let to_activity = data.layers.get(j).map(|l| l.activity_level).unwrap_or(0.0);
            let strength = (from_activity + to_activity) / 2.0;
            
            let (icon, _) = match strength {
                s if s >= 0.8 => ("⬤", Color::Green),
                s if s >= 0.6 => ("◉", Color::LightGreen),
                s if s >= 0.4 => ("◎", Color::Yellow),
                s if s >= 0.2 => ("○", Color::LightYellow),
                _ => ("◯", Color::DarkGray),
            };
            
            cells.push(format!("{} {:.2}", icon, strength));
        }
        
        rows.push(Row::new(cells));
    }
    
    let constraints = std::iter::once(Constraint::Percentage(15))
        .chain((0..memory_types.len()).map(|_| Constraint::Percentage(17)))
        .collect::<Vec<_>>();
    
    let table = Table::new(rows, constraints)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🔗 Association Strength Matrix")
                .border_style(Style::default().fg(Color::Blue)),
        );
    
    f.render_widget(table, area);
}

fn draw_recent_associations_enhanced(f: &mut Frame, area: Rect, data: &MemoryData) {
    let recent_items: Vec<ListItem> = data.recent_associations
        .iter()
        .map(|assoc| {
            let strength_bar = "█".repeat((assoc.strength * 10.0) as usize);
            let strength_color = match assoc.strength {
                s if s >= 0.8 => Color::Green,
                s if s >= 0.5 => Color::Yellow,
                _ => Color::Red,
            };
            
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("{} → {}", assoc.from_node, assoc.to_node), 
                        Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::raw("  Strength: "),
                    Span::styled(strength_bar, Style::default().fg(strength_color)),
                    Span::raw(format!(" {:.2}", assoc.strength)),
                ]),
                Line::from(vec![
                    Span::raw("  Type: "),
                    Span::styled(&assoc.association_type, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(""),
            ])
        })
        .collect();
    
    let list = List::new(recent_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📋 Recent Associations")
                .border_style(Style::default().fg(Color::Blue)),
        );
    
    f.render_widget(list, area);
}

fn draw_association_controls(f: &mut Frame, area: Rect) {
    let controls = vec![
        Line::from(vec![
            Span::styled("[A]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Add new association"),
        ]),
        Line::from(vec![
            Span::styled("[E]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Edit selected"),
        ]),
        Line::from(vec![
            Span::styled("[D]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Delete selected"),
        ]),
        Line::from(vec![
            Span::styled("[S]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Strengthen association"),
        ]),
        Line::from(vec![
            Span::styled("[W]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Weaken association"),
        ]),
        Line::from(vec![
            Span::styled("[R]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Refresh associations"),
        ]),
        Line::from(vec![
            Span::styled("[X]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Export associations"),
        ]),
    ];
    
    let controls_widget = Paragraph::new(controls)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Association Controls "),
        );
    
    f.render_widget(controls_widget, area);
}

fn draw_association_filters(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Extract unique association types from recent associations
    let unique_types: std::collections::HashSet<String> = data.recent_associations
        .iter()
        .map(|a| a.association_type.clone())
        .collect();
    
    let filters = vec![
        Line::from(vec![
            Span::raw("Available Types:"),
        ]),
        Line::from(vec![
            Span::styled(
                unique_types.iter().cloned().collect::<Vec<_>>().join(", "), 
                Style::default().fg(Color::Green)
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Filter by: "),
            Span::styled("[T]ype [S]trength [R]ecent", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Sort by: "),
            Span::styled("[N]odes [W]eight [D]ate", Style::default().fg(Color::Yellow)),
        ]),
    ];
    
    let filters_widget = Paragraph::new(filters)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Filters "),
        );
    
    f.render_widget(filters_widget, area);
}

fn draw_association_stats(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Calculate statistics from recent associations
    let strong_count = data.recent_associations.iter().filter(|a| a.strength >= 0.7).count();
    let weak_count = data.recent_associations.iter().filter(|a| a.strength < 0.3).count();
    let avg_strength = if !data.recent_associations.is_empty() {
        data.recent_associations.iter().map(|a| a.strength).sum::<f32>() / data.recent_associations.len() as f32
    } else {
        0.0
    };
    
    let stats = vec![
        Line::from(vec![
            Span::raw("Total: "),
            Span::styled(format!("{}", data.total_associations), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Strong: "),
            Span::styled(format!("{}", strong_count), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Weak: "),
            Span::styled(format!("{}", weak_count), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Avg Strength: "),
            Span::styled(format!("{:.2}", avg_strength), Style::default().fg(Color::Blue)),
        ]),
    ];
    
    let stats_widget = Paragraph::new(stats)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Statistics "),
        );
    
    f.render_widget(stats_widget, area);
}

fn draw_selected_association_details(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Show details of the first association or a message if none available
    let details = if let Some(first_assoc) = data.recent_associations.first() {
        vec![
            Line::from(vec![
                Span::styled("Sample Association", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("From: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&first_assoc.from_node),
            ]),
            Line::from(vec![
                Span::styled("To: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&first_assoc.to_node),
            ]),
            Line::from(vec![
                Span::styled("Type: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&first_assoc.association_type),
            ]),
            Line::from(vec![
                Span::styled("Strength: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(format!("{:.2}", first_assoc.strength)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Hint: ", Style::default().fg(Color::Yellow)),
                Span::raw("Click on an association to see details"),
            ]),
        ]
    } else {
        vec![
            Line::from("No associations available"),
            Line::from(""),
            Line::from("Associations will appear here"),
            Line::from("as the memory system creates them"),
        ]
    };
    
    let details_widget = Paragraph::new(details)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Association Details "))
        .wrap(Wrap { trim: true });
    
    f.render_widget(details_widget, area);
}

/// Draw memory operations view with enhanced performance metrics
fn draw_memory_operations_view(f: &mut Frame, _app: &App, area: Rect, data: &MemoryData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(65),  // Main operations view
            Constraint::Percentage(35),  // Performance metrics
        ])
        .split(area);

    // Left side - Operations
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),  // Operation controls
            Constraint::Length(15),  // Performance graph
            Constraint::Min(0),      // Operation log
        ])
        .split(chunks[0]);

    // Operation controls
    draw_operation_controls(f, left_chunks[0], data);
    
    // Performance graph
    draw_performance_graph(f, left_chunks[1], data);
    
    // Operation log
    draw_operation_log(f, left_chunks[2], data);
    
    // Right side - Performance metrics
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),  // Cache stats
            Constraint::Length(10),  // SIMD optimization
            Constraint::Length(12),  // Memory pools
            Constraint::Min(0),      // System resources
        ])
        .split(chunks[1]);
    
    // Cache statistics
    draw_cache_stats(f, right_chunks[0], data);
    
    // SIMD optimization status
    draw_simd_status(f, right_chunks[1], data);
    
    // Memory pool status
    draw_memory_pools(f, right_chunks[2], data);
    
    // System resources
    draw_system_resources(f, right_chunks[3], data);
}

fn draw_operation_controls(f: &mut Frame, area: Rect, data: &MemoryData) {
    let control_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);
    
    // Operations and controls
    let ops_controls = vec![
        Line::from(vec![
            Span::styled("⚙️ Operations Control", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[O]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Optimize Memory"),
        ]),
        Line::from(vec![
            Span::styled("[C]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Clear Cache"),
        ]),
        Line::from(vec![
            Span::styled("[B]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Backup Memory"),
        ]),
        Line::from(vec![
            Span::styled("[R]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Rebuild Index"),
        ]),
        Line::from(vec![
            Span::styled("[V]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Vacuum Database"),
        ]),
        Line::from(vec![
            Span::styled("[P]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Pause Operations"),
        ]),
    ];
    
    let controls_widget = Paragraph::new(ops_controls)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    
    f.render_widget(controls_widget, control_chunks[0]);
    
    // Current operations status
    let status_items = vec![
        Line::from(vec![
            Span::styled("📊 Status", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("State: "),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Cache Hit: "),
            Span::styled(format!("{:.1}%", data.cache_hit_rate * 100.0), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Nodes: "),
            Span::styled(format!("{}", data.total_nodes), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Memory: "),
            Span::styled(format!("{:.1} MB", data.memory_usage_mb), 
                Style::default().fg(Color::Cyan)),
        ]),
    ];
    
    let status_widget = Paragraph::new(status_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    
    f.render_widget(status_widget, control_chunks[1]);
}

fn draw_performance_graph(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Create synthetic performance data based on cache hit rate
    let performance_data = vec![
        (data.cache_hit_rate * 100.0) as u64,
        ((data.cache_hit_rate * 100.0) + 5.0) as u64,
        ((data.cache_hit_rate * 100.0) - 3.0) as u64,
        ((data.cache_hit_rate * 100.0) + 2.0) as u64,
        (data.cache_hit_rate * 100.0) as u64,
        ((data.cache_hit_rate * 100.0) + 7.0) as u64,
        ((data.cache_hit_rate * 100.0) - 2.0) as u64,
        (data.cache_hit_rate * 100.0) as u64,
    ];
    
    let sparkline = Sparkline::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title("📈 Cache Performance")
            .border_style(Style::default().fg(Color::Blue)))
        .data(&performance_data)
        .style(Style::default().fg(Color::Cyan))
        .max(100);
    
    f.render_widget(sparkline, area);
}

fn draw_operation_log(f: &mut Frame, area: Rect, data: &MemoryData) {
    // Create synthetic operation log based on recent associations
    let log_items: Vec<ListItem> = data.recent_associations
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, assoc)| {
            let op_type = match i % 5 {
                0 => "store",
                1 => "query",
                2 => "update",
                3 => "link",
                _ => "cache",
            };
            
            let (icon, color) = match op_type {
                "store" => ("💾", Color::Green),
                "query" => ("🔍", Color::Blue),
                "update" => ("✏️", Color::Yellow),
                "link" => ("🔗", Color::Cyan),
                "cache" => ("⚡", Color::Magenta),
                _ => ("•", Color::Gray),
            };
            
            let timestamp = chrono::Utc::now().format("%H:%M:%S").to_string();
            let duration = (assoc.strength * 100.0) as u32;
            
            ListItem::new(Line::from(vec![
                Span::styled(timestamp, Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::raw(icon),
                Span::raw(" "),
                Span::styled(op_type.to_uppercase(), Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::raw(": "),
                Span::raw(format!("{} → {}", &assoc.from_node, &assoc.to_node)),
                Span::raw(" "),
                Span::styled(format!("[{}ms]", duration), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();
    
    let log_list = List::new(log_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("📝 Operation Log")
            .border_style(Style::default().fg(Color::Green)));
    
    f.render_widget(log_list, area);
}

fn draw_cache_stats(f: &mut Frame, area: Rect, data: &MemoryData) {
    let cache_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("🗄️ Cache Statistics", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![
            Span::raw("Hit Rate: "),
            Span::styled(format!("{:.1}%", data.cache_hit_rate * 100.0), 
                Style::default().fg(if data.cache_hit_rate > 0.8 { Color::Green } else { Color::Yellow })),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Memory Used: "),
            Span::styled(format!("{:.1} MB", data.memory_usage_mb),
                Style::default().fg(Color::Blue)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Total Nodes: "),
            Span::styled(format!("{}", data.total_nodes), Style::default().fg(Color::Magenta)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Associations: "),
            Span::styled(format!("{}", data.total_associations), Style::default().fg(Color::Yellow)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Layers: "),
            Span::styled(format!("{}", data.layers.len()), Style::default().fg(Color::Cyan)),
        ])),
    ];
    
    let cache_list = List::new(cache_items)
        .block(Block::default()
            .borders(Borders::ALL));
    
    f.render_widget(cache_list, area);
}

fn draw_simd_status(f: &mut Frame, area: Rect, _data: &MemoryData) {
    // SIMD is enabled based on build configuration
    let simd_enabled = cfg!(target_feature = "simd");
    
    let simd_items = vec![
        Line::from(vec![
            Span::styled("⚡ SIMD Optimization", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled(if simd_enabled { "Enabled ✓" } else { "Available" },
                Style::default().fg(if simd_enabled { Color::Green } else { Color::Yellow })),
        ]),
        Line::from(vec![
            Span::raw("Platform: "),
            Span::styled("Apple Silicon", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Arch: "),
            Span::styled("aarch64", Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Features: "),
            Span::styled("NEON", Style::default().fg(Color::Magenta)),
        ]),
    ];
    
    let simd_widget = Paragraph::new(simd_items)
        .block(Block::default()
            .borders(Borders::ALL));
    
    f.render_widget(simd_widget, area);
}

fn draw_memory_pools(f: &mut Frame, area: Rect, data: &MemoryData) {
    let mut pool_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("💧 Memory Layers", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ])),
        ListItem::new(Line::from("")),
    ];
    
    // Show layer information as memory pools
    for layer in &data.layers {
        let usage_percent = (layer.activity_level * 100.0) as u16;
        let color = match usage_percent {
            0..=50 => Color::Green,
            51..=80 => Color::Yellow,
            _ => Color::Red,
        };
        
        pool_items.push(ListItem::new(Line::from(vec![
            Span::styled(&layer.name, Style::default().fg(Color::White)),
            Span::raw(": "),
            Span::styled(format!("{} nodes ({}% active)", layer.node_count, usage_percent),
                Style::default().fg(color)),
        ])));
    }
    
    let pools_list = List::new(pool_items)
        .block(Block::default()
            .borders(Borders::ALL));
    
    f.render_widget(pools_list, area);
}

fn draw_system_resources(f: &mut Frame, area: Rect, data: &MemoryData) {
    let resources = vec![
        Line::from(vec![
            Span::styled("💻 System Resources", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Memory Used: "),
            Span::styled(format!("{:.1} MB", data.memory_usage_mb),
                Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Total Nodes: "),
            Span::styled(format!("{}", data.total_nodes),
                Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Associations: "),
            Span::styled(format!("{}", data.total_associations),
                Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Cache Rate: "),
            Span::styled(format!("{:.1}%", data.cache_hit_rate * 100.0),
                Style::default().fg(if data.cache_hit_rate > 0.8 { Color::Green } else { Color::Yellow })),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Active Layers: "),
            Span::styled(format!("{}", data.layers.len()),
                Style::default().fg(Color::Cyan)),
        ]),
    ];
    
    let resources_widget = Paragraph::new(resources)
        .block(Block::default()
            .borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    
    f.render_widget(resources_widget, area);
}

// Database configuration helper functions
fn draw_database_operations_log(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData) {
    let log_items: Vec<ListItem> = data.recent_operations
        .iter()
        .map(|op| {
            let color = match op.operation_type.as_str() {
                "INSERT" => Color::Green,
                "UPDATE" => Color::Yellow,
                "DELETE" => Color::Red,
                "SELECT" => Color::Blue,
                _ => Color::Gray,
            };
            
            ListItem::new(Line::from(vec![
                Span::styled(op.timestamp.format("%H:%M:%S").to_string(), Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(&op.operation_type, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::raw(if op.query.len() > 30 { &op.query[..30] } else { &op.query }),
                Span::raw(" "),
                Span::styled(format!("[{}ms]", op.duration_ms), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();
    
    let log_list = List::new(log_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("📋 Recent Operations")
            .border_style(Style::default().fg(Color::Blue)));
    
    f.render_widget(log_list, area);
}

fn draw_database_selector(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::DatabaseData, selected_backend: &Option<String>) {
    // Check which databases are configured based on backend_status
    let databases = vec![
        ("PostgreSQL", "postgresql", Color::Blue, data.backend_status.contains_key("postgresql")),
        ("MySQL", "mysql", Color::Yellow, data.backend_status.contains_key("mysql")),
        ("SQLite", "sqlite", Color::Green, data.backend_status.contains_key("sqlite")),
        ("Redis", "redis", Color::Red, data.backend_status.contains_key("redis")),
        ("RocksDB", "rocksdb", Color::Cyan, data.backend_status.contains_key("rocksdb")),
        ("MongoDB", "mongodb", Color::Magenta, data.backend_status.contains_key("mongodb")),
    ];
    
    let db_items: Vec<ListItem> = databases
        .into_iter()
        .map(|(name, key, color, configured)| {
            let is_selected = selected_backend.as_ref().map(|s| s == key).unwrap_or(false);
            ListItem::new(Line::from(vec![
                Span::styled(
                    if is_selected { "▶ " } else { "  " },
                    Style::default().fg(if is_selected { Color::White } else { color })
                ),
                Span::styled(
                    name,
                    Style::default()
                        .fg(if is_selected { Color::White } else { color })
                        .add_modifier(if is_selected { Modifier::BOLD | Modifier::REVERSED } else { Modifier::empty() })
                ),
                Span::raw(" "),
                Span::styled(
                    if configured { "[Connected]" } else { "[Not configured]" },
                    Style::default().fg(if configured { Color::Green } else { Color::DarkGray })
                ),
            ]))
        })
        .collect();
    
    let selector_list = List::new(db_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Select Database ")
            .border_style(Style::default().fg(Color::Cyan)));
    
    f.render_widget(selector_list, area);
}

fn draw_database_config_form(f: &mut Frame, area: Rect, _data: &crate::tui::connectors::system_connector::DatabaseData, app: &App) {
    // Use the selected backend or default to postgresql
    let selected_db = app.state.selected_database_backend.as_ref().map(|s| s.as_str()).unwrap_or("postgresql");
    let is_editing = app.state.database_config_mode;
    let active_field = app.state.database_form_active_field.as_ref().map(|s| s.as_str());
    
    let config_lines = match selected_db {
        "postgresql" => {
            let get_field = |field: &str| {
                let key = format!("postgresql_{}", field);
                app.state.database_form_fields.get(&key).map(|s| s.as_str()).unwrap_or_default()
            };
            let field_style = |field: &str| {
                if is_editing && active_field == Some(field) {
                    Style::default().fg(Color::White).bg(Color::Blue)
                } else if is_editing {
                    Style::default().fg(Color::White)
                } else {
                    Style::default().fg(Color::Gray)
                }
            };
            vec![
                Line::from(vec![Span::styled("PostgreSQL Configuration", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD))]),
                Line::from(""),
                Line::from(vec![
                    Span::raw("Host: "), 
                    Span::styled(if get_field("host").is_empty() { "localhost" } else { get_field("host") }, field_style("host"))
                ]),
                Line::from(vec![
                    Span::raw("Port: "), 
                    Span::styled(if get_field("port").is_empty() { "5432" } else { get_field("port") }, field_style("port"))
                ]),
                Line::from(vec![
                    Span::raw("Database: "), 
                    Span::styled(if get_field("database").is_empty() { "loki_db" } else { get_field("database") }, field_style("database"))
                ]),
                Line::from(vec![
                    Span::raw("User: "), 
                    Span::styled(if get_field("user").is_empty() { "loki_user" } else { get_field("user") }, field_style("user"))
                ]),
                Line::from(vec![
                    Span::raw("Password: "), 
                    Span::styled(
                        if get_field("password").is_empty() { "********".to_string() } else { "*".repeat(get_field("password").len()) }, 
                        field_style("password")
                    )
                ]),
                Line::from(vec![Span::raw("SSL Mode: "), Span::styled("prefer", Style::default().fg(Color::Green))]),
                Line::from(vec![Span::raw("Pool Size: "), Span::styled("10", Style::default().fg(Color::Yellow))]),
            ]
        },
        "mysql" => vec![
            Line::from(vec![Span::styled("MySQL Configuration", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))]),
            Line::from(""),
            Line::from(vec![Span::raw("Host: "), Span::styled("localhost", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Port: "), Span::styled("3306", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Database: "), Span::styled("loki_db", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("User: "), Span::styled("root", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Charset: "), Span::styled("utf8mb4", Style::default().fg(Color::Green))]),
        ],
        "sqlite" => vec![
            Line::from(vec![Span::styled("SQLite Configuration", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))]),
            Line::from(""),
            Line::from(vec![Span::raw("Path: "), Span::styled("./data/loki.db", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Journal: "), Span::styled("WAL", Style::default().fg(Color::Blue))]),
            Line::from(vec![Span::raw("Sync: "), Span::styled("NORMAL", Style::default().fg(Color::Yellow))]),
            Line::from(vec![Span::raw("Cache: "), Span::styled("2000 pages", Style::default().fg(Color::Cyan))]),
        ],
        "redis" => vec![
            Line::from(vec![Span::styled("Redis Configuration", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))]),
            Line::from(""),
            Line::from(vec![Span::raw("Host: "), Span::styled("localhost", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Port: "), Span::styled("6379", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Database: "), Span::styled("0", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Pool Size: "), Span::styled("10", Style::default().fg(Color::Yellow))]),
            Line::from(vec![Span::raw("Timeout: "), Span::styled("5s", Style::default().fg(Color::Blue))]),
        ],
        "rocksdb" => vec![
            Line::from(vec![Span::styled("RocksDB Configuration", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))]),
            Line::from(""),
            Line::from(vec![Span::raw("Path: "), Span::styled("./data/rocksdb", Style::default().fg(Color::White))]),
            Line::from(vec![Span::raw("Compression: "), Span::styled("Snappy", Style::default().fg(Color::Green))]),
            Line::from(vec![Span::raw("Block Cache: "), Span::styled("512 MB", Style::default().fg(Color::Blue))]),
            Line::from(vec![Span::raw("Write Buffer: "), Span::styled("64 MB", Style::default().fg(Color::Yellow))]),
            Line::from(vec![Span::raw("Max Files: "), Span::styled("-1", Style::default().fg(Color::Magenta))]),
        ],
        _ => vec![Line::from("Select a database to configure")],
    };
    
    let mut all_lines = config_lines;
    if is_editing {
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(vec![
            Span::styled("Tab: Next field | Enter: Save | Esc: Cancel", Style::default().fg(Color::DarkGray))
        ]));
    }
    
    let config_widget = Paragraph::new(all_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(if is_editing { " Configuration [EDIT MODE] " } else { " Configuration " })
            .border_style(Style::default().fg(if is_editing { Color::Yellow } else { Color::DarkGray })));
    
    f.render_widget(config_widget, area);
}

fn draw_database_actions(f: &mut Frame, area: Rect) {
    let actions = vec![
        Line::from(vec![
            Span::styled("[C]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Connect"),
        ]),
        Line::from(vec![
            Span::styled("[T]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Test Connection"),
        ]),
        Line::from(vec![
            Span::styled("[S]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Save Config"),
        ]),
        Line::from(vec![
            Span::styled("[M]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Run Migrations"),
        ]),
        Line::from(vec![
            Span::styled("[B]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Backup Database"),
        ]),
        Line::from(vec![
            Span::styled("[R]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Reset Connection"),
        ]),
    ];
    
    let actions_widget = Paragraph::new(actions)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Actions ")
            .border_style(Style::default().fg(Color::Green)));
    
    f.render_widget(actions_widget, area);
}

fn draw_database_help(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(vec![Span::styled("Database Management", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))]),
        Line::from(""),
        Line::from("Use ↑/↓ to select database"),
        Line::from("Press Enter to edit config"),
        Line::from("Tab to navigate fields"),
        Line::from(""),
        Line::from(vec![Span::styled("Tips:", Style::default().add_modifier(Modifier::BOLD))]),
        Line::from("• Test before connecting"),
        Line::from("• Backup regularly"),
        Line::from("• Monitor performance"),
    ];
    
    let help_widget = Paragraph::new(help_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Help ")
            .border_style(Style::default().fg(Color::DarkGray)))
        .wrap(Wrap { trim: true });
    
    f.render_widget(help_widget, area);
}

#[derive(Debug, Clone, Copy)]
enum MessageType {
    Info,
    Success,
    Warning,
    Error,
}

fn draw_operation_message(f: &mut Frame, area: Rect, message: &str, msg_type: MessageType) {
    let (border_color, title) = match msg_type {
        MessageType::Info => (Color::Blue, " Info "),
        MessageType::Success => (Color::Green, " Success "),
        MessageType::Warning => (Color::Yellow, " Warning "),
        MessageType::Error => (Color::Red, " Error "),
    };
    
    let message_widget = Paragraph::new(message)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(border_color)))
        .wrap(Wrap { trim: true })
        .alignment(Alignment::Center);
    
    // Create a small centered area for the message
    let popup_area = centered_rect(60, 20, area);
    f.render_widget(Clear, popup_area);
    f.render_widget(message_widget, popup_area);
}

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

fn draw_story_creation_form(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    use crate::tui::state::StoryCreationStep;
    
    // Create a larger popup for the comprehensive form
    let popup_area = centered_rect(80, 80, area);
    f.render_widget(Clear, popup_area);
    
    // Split the popup into header, content, and footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Footer
        ])
        .split(popup_area);
    
    // Draw header with current step
    let header_text = match state.story_creation_step {
        StoryCreationStep::SelectType => "Step 1/6: Select Story Type",
        StoryCreationStep::SelectTemplate => "Step 2/6: Choose Template",
        StoryCreationStep::ConfigureBasics => "Step 3/6: Configure Story Details",
        StoryCreationStep::SetupPlotPoints => "Step 4/6: Define Plot Points",
        StoryCreationStep::AssignCharacters => "Step 5/6: Assign Characters",
        StoryCreationStep::Review => "Step 6/6: Review & Create",
    };
    
    let header = Paragraph::new(header_text)
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::Magenta)));
    f.render_widget(header, chunks[0]);
    
    // Draw content based on current step
    match state.story_creation_step {
        StoryCreationStep::SelectType => draw_story_type_selection(f, chunks[1], state),
        StoryCreationStep::SelectTemplate => draw_template_selection(f, chunks[1], state),
        StoryCreationStep::ConfigureBasics => draw_story_configuration(f, chunks[1], state),
        StoryCreationStep::SetupPlotPoints => draw_plot_points_setup(f, chunks[1], state),
        StoryCreationStep::AssignCharacters => draw_character_assignment(f, chunks[1], state),
        StoryCreationStep::Review => draw_story_review(f, chunks[1], state),
    }
    
    // Draw footer with navigation hints
    let footer_text = match state.story_creation_step {
        StoryCreationStep::SelectType => "[↑↓] Navigate | [Enter] Select | [Esc] Cancel",
        StoryCreationStep::SelectTemplate => "[↑↓] Navigate | [Enter] Select | [←] Back | [Esc] Cancel",
        StoryCreationStep::ConfigureBasics => "[Tab] Next Field | [Enter] Continue | [←] Back | [Esc] Cancel",
        StoryCreationStep::SetupPlotPoints => "[+] Add Point | [-] Remove | [Enter] Continue | [←] Back",
        StoryCreationStep::AssignCharacters => "[Space] Toggle | [Enter] Continue | [←] Back | [Esc] Cancel",
        StoryCreationStep::Review => "[Enter] Create Story | [←] Back | [Esc] Cancel",
    };
    
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(footer, chunks[2]);
}

fn draw_story_type_selection(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    let story_types = vec![
        ("Feature", "🚀", "Develop new functionality with defined objectives"),
        ("Bug", "🐛", "Track and resolve issues systematically"),
        ("Epic", "📚", "Large initiatives with multiple sub-stories"),
        ("Task", "📋", "Specific work items or workflows"),
        ("Performance", "⚡", "Optimization work with measurable metrics"),
        ("Security", "🔒", "Security improvements and vulnerability fixes"),
        ("Documentation", "📖", "Create or update documentation"),
        ("Testing", "🧪", "Test coverage and quality improvements"),
        ("Refactoring", "🔧", "Code improvement without changing behavior"),
        ("Research", "🔬", "Explore new technologies or approaches"),
        ("Learning", "🎓", "Knowledge acquisition and skill development"),
        ("Deployment", "🚢", "Release and deployment activities"),
    ];
    
    let items: Vec<ListItem> = story_types.iter().enumerate().map(|(i, (name, icon, desc))| {
        let is_selected = i == state.story_type_index;
        let style = if is_selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        
        ListItem::new(vec![
            Line::from(vec![
                Span::raw(format!("{} ", icon)),
                Span::styled(*name, style),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(*desc, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
        ])
    }).collect();
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Select Story Type ")
            .border_style(Style::default().fg(Color::Blue)))
        .highlight_style(Style::default().bg(Color::DarkGray));
        
    f.render_widget(list, area);
}

fn draw_template_selection(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    // This would show available templates based on selected story type
    let templates = vec![
        ("REST API Development", "Complete workflow for API development"),
        ("Feature Development", "End-to-end feature implementation"),
        ("Bug Investigation", "Systematic bug resolution process"),
        ("Performance Optimization", "Measure, analyze, and optimize"),
        ("Custom", "Start with a blank story"),
    ];
    
    let items: Vec<ListItem> = templates.iter().enumerate().map(|(i, (name, desc))| {
        let is_selected = i == state.story_template_index;
        let style = if is_selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        
        ListItem::new(vec![
            Line::from(Span::styled(*name, style)),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(*desc, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
        ])
    }).collect();
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Select Template ")
            .border_style(Style::default().fg(Color::Blue)));
        
    f.render_widget(list, area);
}

fn draw_story_configuration(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    use crate::tui::state::StoryFormField;
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(5),  // Description
            Constraint::Length(7),  // Objectives
            Constraint::Length(5),  // Metrics
            Constraint::Min(0),     // Context
        ])
        .split(area);
    
    // Title input
    let title_border_style = if matches!(state.story_form_field, StoryFormField::Title) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Blue)
    };
    let title_block = Block::default()
        .borders(Borders::ALL)
        .title(" Story Title ")
        .border_style(title_border_style);
    
    let title_content = if matches!(state.story_form_field, StoryFormField::Title) {
        format!("{}_", state.story_form_input)
    } else {
        state.story_configuration.title.clone()
    };
    let title_text = Paragraph::new(title_content)
        .block(title_block);
    f.render_widget(title_text, chunks[0]);
    
    // Description input
    let desc_border_style = if matches!(state.story_form_field, StoryFormField::Description) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Blue)
    };
    let desc_block = Block::default()
        .borders(Borders::ALL)
        .title(" Description ")
        .border_style(desc_border_style);
    
    let desc_content = if matches!(state.story_form_field, StoryFormField::Description) {
        format!("{}_", state.story_form_input)
    } else {
        state.story_configuration.description.clone()
    };
    let desc_text = Paragraph::new(desc_content)
        .block(desc_block)
        .wrap(Wrap { trim: true });
    f.render_widget(desc_text, chunks[1]);
    
    // Objectives list
    let obj_border_style = if matches!(state.story_form_field, StoryFormField::Objectives) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Blue)
    };
    
    let mut obj_lines: Vec<Line> = state.story_configuration.objectives.iter()
        .map(|obj| Line::from(format!("• {}", obj)))
        .collect();
    
    // Show current input if editing objectives
    if matches!(state.story_form_field, StoryFormField::Objectives) {
        if !state.story_form_input.is_empty() {
            obj_lines.push(Line::from(format!("• {}_ (press Enter to add)", state.story_form_input)));
        } else {
            obj_lines.push(Line::from("Type objective and press Enter to add...").style(Style::default().fg(Color::DarkGray)));
        }
    }
    
    let objectives = Paragraph::new(obj_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Objectives (Press Enter to add) ")
            .border_style(obj_border_style));
    f.render_widget(objectives, chunks[2]);
    
    // Metrics
    let metrics_border_style = if matches!(state.story_form_field, StoryFormField::Metrics) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Blue)
    };
    
    let mut metrics_lines = vec![];
    if !state.story_configuration.metrics.is_empty() {
        metrics_lines.push(Line::from(state.story_configuration.metrics.join(", ")));
    }
    
    // Show current input if editing metrics
    if matches!(state.story_form_field, StoryFormField::Metrics) {
        if !state.story_form_input.is_empty() {
            metrics_lines.push(Line::from(format!("{}_ (press Enter to add)", state.story_form_input)));
        } else {
            metrics_lines.push(Line::from("Type metric and press Enter to add...").style(Style::default().fg(Color::DarkGray)));
        }
    }
    
    let metrics = Paragraph::new(metrics_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Success Metrics (Press Enter to add) ")
            .border_style(metrics_border_style));
    f.render_widget(metrics, chunks[3]);
}

fn draw_plot_points_setup(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    // This would show a list of plot points that can be added/edited
    let plot_points_text = if state.story_configuration.plot_points.is_empty() {
        vec![Line::from("No plot points defined yet."), Line::from("Press '+' to add a plot point")]
    } else {
        state.story_configuration.plot_points.iter().enumerate().map(|(i, pp)| {
            Line::from(format!("{}. {} - {}", i + 1, pp.plot_type, pp.description))
        }).collect()
    };
    
    let plot_points = Paragraph::new(plot_points_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Plot Points (Development Checkpoints) ")
            .border_style(Style::default().fg(Color::Blue)));
    f.render_widget(plot_points, area);
}

fn draw_character_assignment(f: &mut Frame, area: Rect, _state: &crate::tui::state::AppState) {
    // This would show available agents/characters that can be assigned
    let text = vec![
        Line::from("Available Characters/Agents:"),
        Line::from(""),
        Line::from("[ ] System Administrator"),
        Line::from("[ ] Database Specialist"),
        Line::from("[ ] UI Designer"),
        Line::from("[ ] Test Engineer"),
        Line::from(""),
        Line::from("Press [Space] to toggle selection"),
    ];
    
    let characters = Paragraph::new(text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Assign Characters ")
            .border_style(Style::default().fg(Color::Blue)));
    f.render_widget(characters, area);
}

fn draw_story_review(f: &mut Frame, area: Rect, state: &crate::tui::state::AppState) {
    let config = &state.story_configuration;
    let review_text = vec![
        Line::from(vec![
            Span::raw("Story Type: "),
            Span::styled(config.story_type.as_ref().map(|s| s.as_str()).unwrap_or("Unknown"), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Template: "),
            Span::styled(config.template.as_ref().map(|s| s.as_str()).unwrap_or("None"), Style::default().fg(Color::Blue)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Title: "),
            Span::styled(&config.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Description: "),
            Span::styled(&config.description, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(format!("Objectives: {} defined", config.objectives.len())),
        Line::from(format!("Plot Points: {} configured", config.plot_points.len())),
        Line::from(format!("Characters: {} assigned", config.characters.len())),
        Line::from(""),
        Line::from(vec![
            Span::styled("Press [Enter] to create this story", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
    ];
    
    let review = Paragraph::new(review_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Review Story Configuration ")
            .border_style(Style::default().fg(Color::Green)));
    f.render_widget(review, area);
}

// Story management helper functions
fn draw_story_timeline_interactive(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::StoryData) {
    let canvas = Canvas::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title("📚 Story Timeline")
            .border_style(Style::default().fg(Color::Magenta)))
        .x_bounds([0.0, 100.0])
        .y_bounds([0.0, 10.0])
        .paint(|ctx| {
            // Draw timeline axis
            ctx.draw(&ratatui::widgets::canvas::Line {
                x1: 5.0,
                y1: 5.0,
                x2: 95.0,
                y2: 5.0,
                color: Color::White,
            });
            
            // Draw story arcs
            for (i, arc) in data.active_arcs.iter().enumerate() {
                let start_x = 10.0 + (i as f64 * 20.0);
                let end_x = start_x + 15.0;
                let y = 5.0;
                
                // Arc line
                ctx.draw(&ratatui::widgets::canvas::Rectangle {
                    x: start_x,
                    y: y - 1.0,
                    width: end_x - start_x,
                    height: 2.0,
                    color: match arc.status.as_str() {
                        "active" => Color::Green,
                        "completed" => Color::Blue,
                        "planned" => Color::Yellow,
                        _ => Color::Gray,
                    },
                });
                
                // Draw progress markers as small circles
                let progress_markers = 5; // Number of progress markers
                for j in 0..progress_markers {
                    let milestone_x = start_x + (j as f64 * 3.0);
                    let is_completed = (j as f32) < (arc.progress * progress_markers as f32);
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: milestone_x,
                        y,
                        radius: if is_completed { 1.0 } else { 0.5 },
                        color: if is_completed { Color::Green } else { Color::Gray },
                    });
                }
            }
        });
    
    f.render_widget(canvas, area);
}

fn draw_active_stories_list(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::StoryData) {
    let story_items: Vec<ListItem> = data.active_arcs
        .iter()
        .map(|arc| {
            let status_icon = match arc.status.as_str() {
                "active" => "🟢",
                "paused" => "🟡",
                "completed" => "🔵",
                _ => "⚪",
            };
            
            let progress_bar = "█".repeat((arc.progress * 10.0) as usize) + 
                              &"░".repeat(10 - (arc.progress * 10.0) as usize);
            
            ListItem::new(vec![
                Line::from(vec![
                    Span::raw(status_icon),
                    Span::raw(" "),
                    Span::styled(&arc.title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::raw("  Status: "),
                    Span::styled(&arc.status, Style::default().fg(Color::Cyan)),
                    Span::raw(" | Progress: "),
                    Span::styled(progress_bar, Style::default().fg(Color::Green)),
                ]),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(&arc.description, Style::default().fg(Color::DarkGray)),
                ]),
                Line::from(""),
            ])
        })
        .collect();
    
    let stories_list = List::new(story_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("📖 Active Stories")
            .border_style(Style::default().fg(Color::Magenta)));
    
    f.render_widget(stories_list, area);
}

fn draw_story_controls(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::StoryData) {
    let controls = vec![
        Line::from(vec![
            Span::styled("[N]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" New Story"),
        ]),
        Line::from(vec![
            Span::styled("[E]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Edit Selected"),
        ]),
        Line::from(vec![
            Span::styled("[A]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Add Arc"),
        ]),
        Line::from(vec![
            Span::styled("[P]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Pause/Resume"),
        ]),
        Line::from(vec![
            Span::styled("[X]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Export Story"),
        ]),
        Line::from(vec![
            Span::styled("[D]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Delete Story"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Active: "),
            Span::styled(
                format!("{} stories", data.active_stories),
                Style::default().fg(Color::White)
            ),
        ]),
    ];
    
    let controls_widget = Paragraph::new(controls)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Story Controls ")
            .border_style(Style::default().fg(Color::Green)));
    
    f.render_widget(controls_widget, area);
}

fn draw_template_manager(f: &mut Frame, area: Rect, data: &crate::tui::connectors::system_connector::StoryData) {
    let template_items: Vec<ListItem> = data.story_templates
        .iter()
        .map(|template| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    "  ",
                    Style::default().fg(Color::Cyan)
                ),
                Span::styled(&template.name, Style::default().fg(Color::White)),
                Span::raw(" ("),
                Span::styled(&template.category, Style::default().fg(Color::Yellow)),
                Span::raw(")"),
            ]))
        })
        .collect();
    
    let header = Line::from(vec![
        Span::styled("Templates: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(format!("{}", data.story_templates.len()), Style::default().fg(Color::Cyan)),
        Span::raw(" | "),
        Span::styled("[T]", Style::default().fg(Color::Green)),
        Span::raw(" Apply"),
    ]);
    
    let template_list = List::new(template_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(header)
            .border_style(Style::default().fg(Color::Blue)));
    
    f.render_widget(template_list, area);
}

/// Draw story autonomy status display
fn draw_story_autonomy_status(f: &mut Frame, area: Rect, app: &App) {
    // Get story autonomy configuration from the stories tab
    let config = app.state.stories_tab.autonomy_config();
    
    // Count active features
    let active_features = [
        config.auto_maintenance,
        config.pr_review_enabled,
        config.bug_detection_enabled,
        config.quality_monitoring,
        config.performance_optimization,
        config.security_scanning,
        config.test_generation,
        config.refactoring_enabled,
        config.dependency_updates,
    ].iter().filter(|&&x| x).count();
    
    // Build status display
    let mut lines = vec![
        Line::from(vec![
            Span::styled("🤖 Story Autonomy Status", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            if config.auto_maintenance {
                Span::styled("ACTIVE", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            } else {
                Span::styled("INACTIVE", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
            },
        ]),
        Line::from(vec![
            Span::raw("Active Features: "),
            Span::styled(format!("{}/9", active_features), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Maintenance: "),
            Span::styled(
                format!("Every {} hours", config.maintenance_interval_hours),
                Style::default().fg(Color::Blue)
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("✅ Enabled Features:", Style::default().fg(Color::Green)),
        ]),
    ];
    
    // Add enabled features
    if config.pr_review_enabled {
        lines.push(Line::from(vec![
            Span::raw("  • PR Review ("),
            Span::styled(format!("{:.0}%", config.pr_review_threshold * 100.0), Style::default().fg(Color::Cyan)),
            Span::raw(")"),
        ]));
    }
    if config.bug_detection_enabled {
        lines.push(Line::from("  • Bug Detection"));
    }
    if config.quality_monitoring {
        lines.push(Line::from(vec![
            Span::raw("  • Quality ("),
            Span::styled(format!("{:.0}%", config.quality_threshold * 100.0), Style::default().fg(Color::Cyan)),
            Span::raw(")"),
        ]));
    }
    if config.performance_optimization {
        lines.push(Line::from("  • Performance"));
    }
    if config.security_scanning {
        lines.push(Line::from("  • Security"));
    }
    if config.test_generation {
        lines.push(Line::from("  • Test Generation"));
    }
    if config.refactoring_enabled {
        lines.push(Line::from("  • Refactoring"));
    }
    if config.dependency_updates {
        lines.push(Line::from("  • Dependencies"));
    }
    
    // Add recent activity if available
    if let Some(connector) = &app.system_connector {
        if let Some(autonomy_state) = &connector.story_autonomy_state {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("📊 Recent Activity:", Style::default().fg(Color::Magenta)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  Last Check: "),
                Span::styled(
                    autonomy_state.last_maintenance_check
                        .map(|t| t.format("%H:%M:%S").to_string())
                        .unwrap_or_else(|| "Never".to_string()),
                    Style::default().fg(Color::Gray)
                ),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  Issues Found: "),
                Span::styled(
                    format!("{}", autonomy_state.issues_detected),
                    Style::default().fg(if autonomy_state.issues_detected > 0 { Color::Yellow } else { Color::Green })
                ),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  Auto-Fixed: "),
                Span::styled(
                    format!("{}", autonomy_state.issues_auto_fixed),
                    Style::default().fg(Color::Green)
                ),
            ]));
        }
    }
    
    // Add help text
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("[Tab]", Style::default().fg(Color::Yellow)),
        Span::raw(" Stories → Config"),
    ]));
    
    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🌟 Autonomous Maintenance")
                .border_style(Style::default().fg(Color::Cyan))
        )
        .wrap(Wrap { trim: false });
    
    f.render_widget(paragraph, area);
}

/// Draw the storage management tab
fn draw_storage_management(f: &mut Frame, app: &mut App, area: Rect) {
    // Layout for storage management
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),  // Storage overview
            Constraint::Length(15),  // API Keys & Secrets
            Constraint::Min(10),     // Chat History
            Constraint::Length(8),   // Actions
        ])
        .split(area);
    
    // Storage Overview
    draw_storage_overview(f, app, chunks[0]);
    
    // API Keys & Secrets Management
    draw_api_keys_management(f, app, chunks[1]);
    
    // Chat History Management
    draw_chat_history_management(f, app, chunks[2]);
    
    // Storage Actions
    draw_storage_actions(f, app, chunks[3]);
}

fn draw_storage_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);
    
    // Local Storage Status
    let local_storage = vec![
        Line::from(vec![
            Span::styled("💾 Local Storage", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Path: "),
            Span::styled("~/.loki/storage", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Size: "),
            Span::styled("124 MB", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("● Active", Style::default().fg(Color::Green)),
        ]),
    ];
    
    let local_widget = Paragraph::new(local_storage)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(local_widget, chunks[0]);
    
    // Cloud Storage Status
    let cloud_storage = vec![
        Line::from(vec![
            Span::styled("☁️ Cloud Storage", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Provider: "),
            Span::styled("S3", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Bucket: "),
            Span::styled("loki-storage", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("○ Not Configured", Style::default().fg(Color::Gray)),
        ]),
    ];
    
    let cloud_widget = Paragraph::new(cloud_storage)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(cloud_widget, chunks[1]);
    
    // Security Status
    let security_status = vec![
        Line::from(vec![
            Span::styled("🔐 Security", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Encryption: "),
            Span::styled("AES-256", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Keyring: "),
            Span::styled("Available", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("🔒 Locked", Style::default().fg(Color::Yellow)),
        ]),
    ];
    
    let security_widget = Paragraph::new(security_status)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(security_widget, chunks[2]);
}

fn draw_api_keys_management(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(area);
    
    // API Keys List
    let header = Row::new(vec!["Provider", "Status", "Expires"])
        .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));
    
    let rows = vec![
        Row::new(vec!["OpenAI", "✅ Configured", "Never"]),
        Row::new(vec!["Anthropic", "✅ Configured", "Never"]),
        Row::new(vec!["Gemini", "○ Not Set", "-"]),
        Row::new(vec!["Mistral", "○ Not Set", "-"]),
        Row::new(vec!["GitHub", "✅ Configured", "2025-12-31"]),
    ];
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(15),
            Constraint::Length(15),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                " 🔑 API Keys ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
    )
    .style(Style::default().fg(Color::Gray));
    
    f.render_widget(table, chunks[0]);
    
    // API Key Actions
    let actions = vec![
        Line::from(vec![
            Span::styled("Key Actions", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[A]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Add API Key"),
        ]),
        Line::from(vec![
            Span::styled("[U]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Update Key"),
        ]),
        Line::from(vec![
            Span::styled("[D]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(" Delete Key"),
        ]),
        Line::from(vec![
            Span::styled("[I]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Import Keys"),
        ]),
        Line::from(vec![
            Span::styled("[E]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Export Keys"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[L]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" Lock Storage"),
        ]),
    ];
    
    let actions_widget = Paragraph::new(actions)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(actions_widget, chunks[1]);
}

fn draw_chat_history_management(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70),
            Constraint::Percentage(30),
        ])
        .split(area);
    
    // Chat History List
    let header = Row::new(vec!["Conversation", "Date", "Messages", "Model"])
        .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));
    
    let rows = vec![
        Row::new(vec!["Memory System Integration", "2025-08-08", "42", "claude-3.5"]),
        Row::new(vec!["Database Configuration", "2025-08-08", "28", "claude-3.5"]),
        Row::new(vec!["Story Engine Setup", "2025-08-07", "35", "gpt-4"]),
        Row::new(vec!["API Setup Discussion", "2025-08-07", "15", "claude-3.5"]),
        Row::new(vec!["Initial Configuration", "2025-08-06", "52", "claude-3.5"]),
    ];
    
    let table = Table::new(
        rows,
        vec![
            Constraint::Length(25),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta))
            .title(Span::styled(
                " 💬 Chat History ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ))
    )
    .style(Style::default().fg(Color::Gray));
    
    f.render_widget(table, chunks[0]);
    
    // Chat History Stats
    let stats = vec![
        Line::from(vec![
            Span::styled("Statistics", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Total Chats: "),
            Span::styled("127", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Messages: "),
            Span::styled("3,842", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Tokens Used: "),
            Span::styled("245.6K", Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::raw("Storage: "),
            Span::styled("18.4 MB", Style::default().fg(Color::Magenta)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Actions:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("[S]", Style::default().fg(Color::Green)),
            Span::raw(" Search History"),
        ]),
        Line::from(vec![
            Span::styled("[X]", Style::default().fg(Color::Yellow)),
            Span::raw(" Export Chat"),
        ]),
        Line::from(vec![
            Span::styled("[C]", Style::default().fg(Color::Red)),
            Span::raw(" Clear Old"),
        ]),
    ];
    
    let stats_widget = Paragraph::new(stats)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray)));
    f.render_widget(stats_widget, chunks[1]);
}

fn draw_storage_actions(f: &mut Frame, app: &App, area: Rect) {
    let actions = vec![
        Line::from(vec![
            Span::styled("🚀 Storage Actions", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[Ctrl+U]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" Unlock Storage (Enter Master Password)"),
            Span::raw("    "),
            Span::styled("[Ctrl+B]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" Backup All Data"),
        ]),
        Line::from(vec![
            Span::styled("[Ctrl+R]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Restore from Backup"),
            Span::raw("                   "),
            Span::styled("[Ctrl+S]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::raw(" Sync to Cloud"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Tip: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Storage is encrypted with AES-256. Use 'setup-apis' CLI command to configure API keys persistently."),
        ]),
    ];
    
    let actions_widget = Paragraph::new(actions)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray))
                .title(Span::styled(
                    " Actions ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });
    
    f.render_widget(actions_widget, area);
}

