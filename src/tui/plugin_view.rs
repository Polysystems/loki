use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Styled};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block,
    Borders,
    Cell,
    Gauge,
    List,
    ListItem,
    Paragraph,
    Row,
    Table,
    Tabs,
    Wrap,
};

use super::app::App;

/// Draw the plugin ecosystem view with comprehensive plugin management
/// interface
pub fn draw_plugin_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Plugin tabs
            Constraint::Min(8),    // Main content
            Constraint::Length(3), // Status/actions
        ])
        .split(area);

    draw_plugin_tabs(f, app, chunks[0]);
    draw_plugin_content(f, app, chunks[1]);
    draw_plugin_actions(f, app, chunks[2]);
}

/// Draw plugin management tabs
fn draw_plugin_tabs(f: &mut Frame, app: &App, area: Rect) {
    let tabs = vec![
        "Installed [1]",
        "Available [2]",
        "Marketplace [3]",
        "WASM Engine [4]",
        "Security [5]",
        "Performance [6]",
    ];

    let selected_tab = app.state.plugin_view.selected_tab;

    let tab_widget = Tabs::new(tabs)
        .block(Block::default().borders(Borders::ALL).title(" 🧩 Plugin Ecosystem "))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .select(selected_tab);

    f.render_widget(tab_widget, area);
}

/// Draw main plugin content based on selected tab
fn draw_plugin_content(f: &mut Frame, app: &App, area: Rect) {
    match app.state.plugin_view.selected_tab {
        0 => draw_installed_plugins(f, app, area),
        1 => draw_available_plugins(f, app, area),
        2 => draw_marketplace(f, app, area),
        3 => draw_wasm_engine(f, app, area),
        4 => draw_security_view(f, app, area),
        5 => draw_performance_view(f, app, area),
        _ => draw_installed_plugins(f, app, area),
    }
}

/// Draw installed plugins view
fn draw_installed_plugins(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Plugin list
    let plugin_items: Vec<ListItem> = app
        .state
        .plugin_view
        .installed_plugins
        .iter()
        .enumerate()
        .map(|(i, plugin)| {
            let status_icon = match plugin.state.as_str() {
                "Active" => "🟢",
                "Stopped" => "🔴",
                "Loading" => "🟡",
                "Error" => "❌",
                _ => "⚪",
            };

            let plugin_type_icon = match plugin.plugin_type.as_str() {
                "WASM" => "⚡",
                "Native" => "🔧",
                "Python" => "🐍",
                "JavaScript" => "📜",
                "Lua" => "🌙",
                "Remote" => "🌐",
                _ => "❓",
            };

            let style = if Some(i) == app.state.plugin_view.selected_plugin {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::raw(format!("{} {} ", status_icon, plugin_type_icon)),
                Span::styled(&plugin.name, Style::default().fg(Color::Cyan)),
                Span::raw(format!(" v{}", plugin.version)),
                if plugin.capabilities.is_empty() {
                    Span::raw("")
                } else {
                    Span::styled(
                        format!(" [{}]", plugin.capabilities.join(", ")),
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
                    )
                },
            ]))
            .style(style)
        })
        .collect();

    let plugins_list = List::new(plugin_items)
        .block(Block::default().borders(Borders::ALL).title(" Installed Plugins "))
        .highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(plugins_list, chunks[0]);

    // Plugin details
    if let Some(selected_idx) = app.state.plugin_view.selected_plugin {
        if let Some(plugin) = app.state.plugin_view.installed_plugins.get(selected_idx) {
            draw_plugin_details(f, plugin, chunks[1]);
        }
    } else {
        draw_plugin_overview(f, app, chunks[1]);
    }
}

/// Draw plugin details panel
fn draw_plugin_details(f: &mut Frame, plugin: &crate::tui::state::PluginInfo, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Basic info
            Constraint::Length(4), // Metrics
            Constraint::Min(4),    // Description & capabilities
        ])
        .split(area);

    // Basic information
    let info_text = vec![
        Line::from(vec![
            Span::styled("Name: ", Style::default().fg(Color::Yellow)),
            Span::raw(&plugin.name),
        ]),
        Line::from(vec![
            Span::styled("Version: ", Style::default().fg(Color::Yellow)),
            Span::raw(&plugin.version),
        ]),
        Line::from(vec![
            Span::styled("Author: ", Style::default().fg(Color::Yellow)),
            Span::raw(&plugin.author),
        ]),
        Line::from(vec![
            Span::styled("Type: ", Style::default().fg(Color::Yellow)),
            Span::raw(&plugin.plugin_type),
        ]),
        Line::from(vec![
            Span::styled("State: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                &plugin.state,
                match plugin.state.as_str() {
                    "Active" => Style::default().fg(Color::Green),
                    "Error" => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Gray),
                },
            ),
        ]),
        Line::from(vec![
            Span::styled("Load Time: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{}ms", plugin.load_time_ms)),
        ]),
    ];

    let info_paragraph = Paragraph::new(info_text)
        .block(Block::default().borders(Borders::ALL).title(" Information "))
        .wrap(Wrap { trim: true });

    f.render_widget(info_paragraph, chunks[0]);

    // Performance metrics
    let metrics_text = vec![
        Line::from(vec![
            Span::styled("Function Calls: ", Style::default().fg(Color::Cyan)),
            Span::raw(plugin.function_calls.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Memory Usage: ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{:.1} MB", plugin.memory_usage_mb)),
        ]),
        Line::from(vec![
            Span::styled("CPU Time: ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{:.2}ms", plugin.cpu_time_ms)),
        ]),
    ];

    let metrics_paragraph = Paragraph::new(metrics_text)
        .block(Block::default().borders(Borders::ALL).title(" Metrics "))
        .wrap(Wrap { trim: true });

    f.render_widget(metrics_paragraph, chunks[1]);

    // Description and capabilities
    let desc_text = vec![
        Line::from(vec![Span::styled(
            "Description:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from(Span::raw(&plugin.description)),
        Line::raw(""),
        Line::from(vec![Span::styled(
            "Capabilities:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from(Span::raw(plugin.capabilities.join(", "))),
    ];

    let desc_paragraph = Paragraph::new(desc_text)
        .block(Block::default().borders(Borders::ALL).title(" Details "))
        .wrap(Wrap { trim: true });

    f.render_widget(desc_paragraph, chunks[2]);
}

/// Draw available plugins (discovered but not installed)
fn draw_available_plugins(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    // Available plugins list
    let available_items: Vec<ListItem> = app
        .state
        .plugin_view
        .available_plugins
        .iter()
        .enumerate()
        .map(|(i, plugin)| {
            let plugin_type_icon = match plugin.plugin_type.as_str() {
                "WASM" => "⚡",
                "Native" => "🔧",
                "Python" => "🐍",
                "JavaScript" => "📜",
                "Lua" => "🌙",
                "Remote" => "🌐",
                _ => "❓",
            };

            let style = if Some(i) == app.state.plugin_view.selected_available {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::raw(format!("{} ", plugin_type_icon)),
                Span::styled(&plugin.name, Style::default().fg(Color::Cyan)),
                Span::raw(format!(" v{} - ", plugin.version)),
                Span::styled(
                    &plugin.description,
                    Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
                ),
            ]))
            .style(style)
        })
        .collect();

    let available_list = List::new(available_items)
        .block(Block::default().borders(Borders::ALL).title(" Available Plugins "))
        .highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(available_list, chunks[0]);

    // Plugin discovery stats
    let stats_text = vec![
        Line::from(vec![
            Span::styled("📁 Scanned Directories: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.scanned_directories.to_string()),
        ]),
        Line::from(vec![
            Span::styled("🔍 Available Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.available_plugins.len().to_string()),
        ]),
        Line::from(vec![
            Span::styled("⚡ WASM Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw(
                app.state
                    .plugin_view
                    .available_plugins
                    .iter()
                    .filter(|p| p.plugin_type == "WASM")
                    .count()
                    .to_string(),
            ),
        ]),
        Line::from(vec![
            Span::styled("🔧 Native Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw(
                app.state
                    .plugin_view
                    .available_plugins
                    .iter()
                    .filter(|p| p.plugin_type == "Native")
                    .count()
                    .to_string(),
            ),
        ]),
        Line::from(vec![
            Span::styled("📜 Script Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw(
                app.state
                    .plugin_view
                    .available_plugins
                    .iter()
                    .filter(|p| ["Python", "JavaScript", "Lua"].contains(&p.plugin_type.as_str()))
                    .count()
                    .to_string(),
            ),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Last Scan: ", Style::default().fg(Color::Gray)),
            Span::raw(&app.state.plugin_view.last_scan_time),
        ]),
    ];

    let stats_paragraph = Paragraph::new(stats_text)
        .block(Block::default().borders(Borders::ALL).title(" Discovery Stats "))
        .wrap(Wrap { trim: true });

    f.render_widget(stats_paragraph, chunks[1]);
}

/// Draw plugin marketplace view
fn draw_marketplace(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Search bar
            Constraint::Min(8),    // Marketplace content
        ])
        .split(area);

    // Search and filters
    let search_text = vec![Line::from(vec![
        Span::styled("🔍 Search: ", Style::default().fg(Color::Yellow)),
        Span::styled(
            &app.state.plugin_view.marketplace_search,
            Style::default().fg(Color::White).bg(Color::DarkGray),
        ),
        Span::raw("   Category: "),
        Span::styled(&app.state.plugin_view.marketplace_category, Style::default().fg(Color::Cyan)),
    ])];

    let search_paragraph = Paragraph::new(search_text)
        .block(Block::default().borders(Borders::ALL).title(" Plugin Marketplace "));

    f.render_widget(search_paragraph, chunks[0]);

    // Marketplace content
    let marketplace_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);

    // Featured/Popular plugins
    let featured_items: Vec<ListItem> = app
        .state
        .plugin_view
        .marketplace_plugins
        .iter()
        .enumerate()
        .map(|(i, plugin)| {
            let rating_stars = format!(
                "{}{}",
                "★".repeat(plugin.rating as usize),
                "☆".repeat(5 - plugin.rating as usize)
            );

            let style = if Some(i) == app.state.plugin_view.selected_marketplace {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(
                    plugin.name.clone(),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(" v{} ", plugin.version)),
                Span::styled(rating_stars, Style::default().fg(Color::Yellow)),
                Span::raw(format!(" ({})", plugin.downloads)),
                Span::raw(" - "),
                Span::styled(
                    plugin.description.clone(),
                    Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
                ),
            ]))
            .style(style)
        })
        .collect();

    let featured_list = List::new(featured_items)
        .block(Block::default().borders(Borders::ALL).title(" Featured Plugins "))
        .highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(featured_list, marketplace_chunks[0]);

    // Marketplace stats and categories
    let marketplace_info = vec![
        Line::from(vec![Span::styled(
            "📊 Marketplace Stats",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Total Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw("1,247"),
        ]),
        Line::from(vec![
            Span::styled("Categories: ", Style::default().fg(Color::Yellow)),
            Span::raw("42"),
        ]),
        Line::from(vec![
            Span::styled("Active Developers: ", Style::default().fg(Color::Yellow)),
            Span::raw("284"),
        ]),
        Line::raw(""),
        Line::from(vec![Span::styled(
            "🏷️ Popular Categories",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::raw(""),
        Line::from(vec![
            Span::raw("• "),
            Span::styled("AI/ML Tools", Style::default().fg(Color::Cyan)),
            Span::raw(" (156)"),
        ]),
        Line::from(vec![
            Span::raw("• "),
            Span::styled("Code Analysis", Style::default().fg(Color::Cyan)),
            Span::raw(" (98)"),
        ]),
        Line::from(vec![
            Span::raw("• "),
            Span::styled("Social Media", Style::default().fg(Color::Cyan)),
            Span::raw(" (67)"),
        ]),
        Line::from(vec![
            Span::raw("• "),
            Span::styled("Data Processing", Style::default().fg(Color::Cyan)),
            Span::raw(" (134)"),
        ]),
    ];

    let info_paragraph = Paragraph::new(marketplace_info)
        .block(Block::default().borders(Borders::ALL).title(" Marketplace Info "))
        .wrap(Wrap { trim: true });

    f.render_widget(info_paragraph, marketplace_chunks[1]);
}

/// Draw WebAssembly engine status and controls
fn draw_wasm_engine(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Engine status
            Constraint::Length(6), // Resource usage
            Constraint::Min(4),    // Active WASM instances
        ])
        .split(area);

    // Engine status
    let engine_status = vec![
        Line::from(vec![Span::styled(
            "🔧 WASM Engine Status",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Engine State: ", Style::default().fg(Color::Yellow)),
            Span::styled("Running", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Loaded Modules: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.wasm_loaded_modules.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Active Instances: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.wasm_active_instances.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Total Function Calls: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.wasm_function_calls.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Compilation Cache Hits: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:.1}%", app.state.plugin_view.wasm_cache_hit_rate)),
        ]),
    ];

    let status_paragraph = Paragraph::new(engine_status)
        .block(Block::default().borders(Borders::ALL).title(" WASM Engine "));

    f.render_widget(status_paragraph, chunks[0]);

    // Resource usage
    let memory_percentage = (app.state.plugin_view.wasm_memory_usage_mb
        / app.state.plugin_view.wasm_memory_limit_mb
        * 100.0)
        .min(100.0);
    let cpu_percentage = app.state.plugin_view.wasm_cpu_usage_percent;

    let memory_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Memory Usage "))
        .set_style(Style::default().fg(if memory_percentage > 80.0 {
            Color::Red
        } else if memory_percentage > 60.0 {
            Color::Yellow
        } else {
            Color::Green
        }))
        .percent(memory_percentage as u16)
        .label(format!(
            "{:.1} / {:.1} MB ({:.1}%)",
            app.state.plugin_view.wasm_memory_usage_mb,
            app.state.plugin_view.wasm_memory_limit_mb,
            memory_percentage
        ));

    let resource_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    f.render_widget(memory_gauge, resource_chunks[0]);

    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" CPU Usage "))
        .set_style(Style::default().fg(if cpu_percentage > 80.0 {
            Color::Red
        } else if cpu_percentage > 60.0 {
            Color::Yellow
        } else {
            Color::Green
        }))
        .percent(cpu_percentage as u16)
        .label(format!("{:.1}%", cpu_percentage));

    f.render_widget(cpu_gauge, resource_chunks[1]);

    // Active WASM instances table
    let header = Row::new(vec!["Plugin", "State", "Memory", "Fuel", "Calls", "Errors"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app
        .state
        .plugin_view
        .wasm_instances
        .iter()
        .map(|instance| {
            Row::new(vec![
                Cell::from(instance.plugin_name.clone()),
                Cell::from(instance.state.clone()).style(match instance.state.as_str() {
                    "Active" => Style::default().fg(Color::Green),
                    "Error" => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Gray),
                }),
                Cell::from(format!("{:.1} MB", instance.memory_mb)),
                Cell::from(format!("{:.0}%", instance.fuel_usage_percent)),
                Cell::from(instance.function_calls.to_string()),
                Cell::from(instance.error_count.to_string()).style(if instance.error_count > 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                }),
            ])
        })
        .collect();

    let instances_table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Active WASM Instances "))
    .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(instances_table, chunks[2]);
}

/// Draw security overview
fn draw_security_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Security status
    let security_status = vec![
        Line::from(vec![Span::styled(
            "🔒 Security Status",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Sandbox Mode: ", Style::default().fg(Color::Yellow)),
            Span::styled("Enabled", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Permission Checks: ", Style::default().fg(Color::Yellow)),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Code Signing: ", Style::default().fg(Color::Yellow)),
            Span::styled("Required", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Network Isolation: ", Style::default().fg(Color::Yellow)),
            Span::styled("Enforced", Style::default().fg(Color::Green)),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Security Violations: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                app.state.plugin_view.security_violations.to_string(),
                if app.state.plugin_view.security_violations > 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                },
            ),
        ]),
        Line::from(vec![
            Span::styled("Blocked Actions: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.blocked_actions.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Quarantined Plugins: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.quarantined_plugins.to_string()),
        ]),
    ];

    let security_paragraph = Paragraph::new(security_status)
        .block(Block::default().borders(Borders::ALL).title(" Security Overview "));

    f.render_widget(security_paragraph, chunks[0]);

    // Recent security events
    let security_events: Vec<ListItem> = app
        .state
        .plugin_view
        .security_events
        .iter()
        .map(|event| {
            let severity_icon = match event.severity.as_str() {
                "High" => "🔴",
                "Medium" => "🟡",
                "Low" => "🟢",
                _ => "⚪",
            };

            ListItem::new(Line::from(vec![
                Span::raw(format!("{} ", severity_icon)),
                Span::styled(&event.timestamp, Style::default().fg(Color::Gray)),
                Span::raw(" - "),
                Span::styled(&event.plugin_name, Style::default().fg(Color::Cyan)),
                Span::raw(": "),
                Span::raw(&event.event_type),
                Span::raw(" - "),
                Span::styled(&event.description, Style::default().fg(Color::Yellow)),
            ]))
        })
        .collect();

    let events_list = List::new(security_events)
        .block(Block::default().borders(Borders::ALL).title(" Recent Security Events "));

    f.render_widget(events_list, chunks[1]);
}

/// Draw performance monitoring view
fn draw_performance_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Performance summary
            Constraint::Min(8),    // Performance table
        ])
        .split(area);

    // Performance summary
    let perf_summary = vec![
        Line::from(vec![Span::styled(
            "📊 Performance Summary",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Total Function Calls: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.total_function_calls.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Average Execution Time: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:.2}ms", app.state.plugin_view.avg_execution_time_ms)),
        ]),
        Line::from(vec![
            Span::styled("Peak Memory Usage: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:.1} MB", app.state.plugin_view.peak_memory_mb)),
        ]),
        Line::from(vec![
            Span::styled("Error Rate: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{:.2}%", app.state.plugin_view.error_rate_percent),
                if app.state.plugin_view.error_rate_percent > 5.0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                },
            ),
        ]),
        Line::from(vec![
            Span::styled("Plugins Under Load: ", Style::default().fg(Color::Yellow)),
            Span::raw(app.state.plugin_view.plugins_under_load.to_string()),
        ]),
    ];

    let summary_paragraph = Paragraph::new(perf_summary)
        .block(Block::default().borders(Borders::ALL).title(" Performance Metrics "));

    f.render_widget(summary_paragraph, chunks[0]);

    // Performance table
    let header =
        Row::new(vec!["Plugin", "Calls", "Avg Time", "Memory", "CPU %", "Errors", "Status"])
            .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app
        .state
        .plugin_view
        .performance_data
        .iter()
        .map(|perf| {
            Row::new(vec![
                Cell::from(perf.plugin_name.clone()),
                Cell::from(perf.function_calls.to_string()),
                Cell::from(format!("{:.2}ms", perf.avg_time_ms)),
                Cell::from(format!("{:.1} MB", perf.memory_mb)),
                Cell::from(format!("{:.1}%", perf.cpu_percent)),
                Cell::from(perf.error_count.to_string()).style(if perf.error_count > 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                }),
                Cell::from(perf.status.clone()).style(match perf.status.as_str() {
                    "Optimal" => Style::default().fg(Color::Green),
                    "Warning" => Style::default().fg(Color::Yellow),
                    "Critical" => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Gray),
                }),
            ])
        })
        .collect();

    let perf_table = Table::new(
        rows,
        [
            Constraint::Percentage(20),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Plugin Performance Details "))
    .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(perf_table, chunks[1]);
}

/// Draw plugin action buttons and status
fn draw_plugin_actions(f: &mut Frame, _app: &App, area: Rect) {
    let actions_text = vec![Line::from(vec![
        Span::styled("Actions: ", Style::default().fg(Color::Yellow)),
        Span::styled("[L]", Style::default().fg(Color::Green)),
        Span::raw("oad "),
        Span::styled("[U]", Style::default().fg(Color::Red)),
        Span::raw("nload "),
        Span::styled("[R]", Style::default().fg(Color::Cyan)),
        Span::raw("eload "),
        Span::styled("[I]", Style::default().fg(Color::Blue)),
        Span::raw("nstall "),
        Span::styled("[S]", Style::default().fg(Color::Magenta)),
        Span::raw("can "),
        Span::styled("[C]", Style::default().fg(Color::Yellow)),
        Span::raw("onfigure"),
    ])];

    let actions_paragraph = Paragraph::new(actions_text)
        .block(Block::default().borders(Borders::ALL).title(" Plugin Actions "))
        .alignment(Alignment::Center);

    f.render_widget(actions_paragraph, area);
}

/// Draw plugin ecosystem overview when no specific plugin is selected
fn draw_plugin_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Plugin ecosystem stats
            Constraint::Length(6), // WASM engine status
            Constraint::Min(0),    // Security & performance summary
        ])
        .split(area);

    // Plugin ecosystem stats
    draw_ecosystem_stats(f, app, chunks[0]);

    // WASM engine status
    draw_wasm_engine_status(f, app, chunks[1]);

    // Security and performance summary
    draw_security_performance_summary(f, app, chunks[2]);
}

/// Draw plugin ecosystem statistics
fn draw_ecosystem_stats(f: &mut Frame, app: &App, area: Rect) {
    let stats_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Installed plugins
            Constraint::Percentage(25), // Available plugins
            Constraint::Percentage(25), // Function calls
            Constraint::Percentage(25), // Error rate
        ])
        .split(area);

    let plugin_state = &app.state.plugin_view;

    // Installed plugins count
    let installed_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📦 Installed "))
        .set_style(Style::default().fg(Color::Green))
        .percent(((plugin_state.installed_plugins.len() as f64 / 10.0) * 100.0).min(100.0) as u16)
        .label(format!("{} plugins", plugin_state.installed_plugins.len()));
    f.render_widget(installed_gauge, stats_layout[0]);

    // Available plugins count
    let available_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🔍 Available "))
        .set_style(Style::default().fg(Color::Blue))
        .percent(((plugin_state.available_plugins.len() as f64 / 20.0) * 100.0).min(100.0) as u16)
        .label(format!("{} plugins", plugin_state.available_plugins.len()));
    f.render_widget(available_gauge, stats_layout[1]);

    // Total function calls
    let calls_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚡ Function Calls "))
        .set_style(Style::default().fg(Color::Yellow))
        .percent(((plugin_state.total_function_calls as f64 / 10000.0) * 100.0).min(100.0) as u16)
        .label(format!("{}", plugin_state.total_function_calls));
    f.render_widget(calls_gauge, stats_layout[2]);

    // Error rate
    let error_color = if plugin_state.error_rate_percent < 1.0 {
        Color::Green
    } else if plugin_state.error_rate_percent < 5.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let error_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ❌ Error Rate "))
        .set_style(Style::default().fg(error_color))
        .percent((plugin_state.error_rate_percent * 10.0).min(100.0) as u16)
        .label(format!("{:.1}%", plugin_state.error_rate_percent));
    f.render_widget(error_gauge, stats_layout[3]);
}

/// Draw WASM engine status
fn draw_wasm_engine_status(f: &mut Frame, app: &App, area: Rect) {
    let wasm_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33), // Loaded modules
            Constraint::Percentage(33), // Active instances
            Constraint::Percentage(34), // Memory usage
        ])
        .split(area);

    let plugin_state = &app.state.plugin_view;

    // Loaded modules
    let modules_para = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Modules: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.wasm_loaded_modules),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Cache Hit: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}%", plugin_state.wasm_cache_hit_rate * 100.0),
                Style::default().fg(Color::Green),
            ),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🧩 WASM Modules "));
    f.render_widget(modules_para, wasm_layout[0]);

    // Active instances
    let instances_para = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Active: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.wasm_active_instances),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Calls: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.wasm_function_calls),
                Style::default().fg(Color::Blue),
            ),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" ⚙️ Instances "));
    f.render_widget(instances_para, wasm_layout[1]);

    // Memory usage
    let memory_usage_percent =
        (plugin_state.wasm_memory_usage_mb / plugin_state.wasm_memory_limit_mb * 100.0).min(100.0);
    let memory_para = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Used: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}MB", plugin_state.wasm_memory_usage_mb),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Limit: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}MB", plugin_state.wasm_memory_limit_mb),
                Style::default().fg(Color::Gray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Usage: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}%", memory_usage_percent),
                Style::default().fg(if memory_usage_percent > 80.0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 💾 Memory "));
    f.render_widget(memory_para, wasm_layout[2]);
}

/// Draw security and performance summary
fn draw_security_performance_summary(f: &mut Frame, app: &App, area: Rect) {
    let summary_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Security status
            Constraint::Percentage(50), // Performance metrics
        ])
        .split(area);

    let plugin_state = &app.state.plugin_view;

    // Security status
    let security_text = vec![
        Line::from(vec![Span::styled(
            "🛡️ Security Status",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("Violations: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.security_violations),
                Style::default().fg(if plugin_state.security_violations > 0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Blocked Actions: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.blocked_actions),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Quarantined: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", plugin_state.quarantined_plugins),
                Style::default().fg(if plugin_state.quarantined_plugins > 0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
    ];

    let security_para = Paragraph::new(security_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(security_para, summary_layout[0]);

    // Performance metrics
    let performance_text = vec![
        Line::from(vec![Span::styled(
            "⚡ Performance Metrics",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("Avg Execution: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2}ms", plugin_state.avg_execution_time_ms),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Peak Memory: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}MB", plugin_state.peak_memory_mb),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Under Load: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{} plugins", plugin_state.plugins_under_load),
                Style::default().fg(if plugin_state.plugins_under_load > 3 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
    ];

    let performance_para = Paragraph::new(performance_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(performance_para, summary_layout[1]);
}
