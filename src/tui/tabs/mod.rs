// Old chat module is being replaced by modular system in src/tui/chat/
// pub mod chat;
// pub use chat::*;
pub mod social;
pub use social::*;
pub mod settings;
pub mod utilities;
pub mod stories;
pub mod cognitive;
pub use cognitive::*;
pub mod memory;
pub use memory::*;
pub mod keybindings;
pub use keybindings::*;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, Line, Modifier, Span, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap, Gauge, Sparkline};
pub use settings::*;
pub use utilities::*;

use chrono::Utc;
use crate::tui::App;
use crate::tui::ui::centered_rect;

pub fn draw_tab_home(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Draw dashboard sub-tabs header
    draw_home_dashboard_tabs(f, app, chunks[0]);

    // Draw content based on current dashboard sub-tab
    match app.state.home_dashboard_tabs.current_index {
        0 => draw_welcome_screen(f, app, chunks[1]),
        1 => draw_monitoring_management(f, app, chunks[1]),
        2 => draw_keybindings(f, app, chunks[1]),
        _ => draw_welcome_screen(f, app, chunks[1]),
    }
}

fn draw_home_content(f: &mut Frame, area: Rect) {
    let content_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30), // Top spacer
            Constraint::Length(8),      // LOKI ASCII art
            Constraint::Length(4),      // Navigation hints
            Constraint::Percentage(60), // Bottom spacer
        ])
        .split(area);

    draw_loki_ascii(f, content_chunks[1]);
    draw_navigation_hints(f, content_chunks[2]);
}

fn draw_loki_ascii(f: &mut Frame, area: Rect) {
    let loki_ascii = vec![
        Line::from("██╗      ██████╗ ██╗  ██╗██╗"),
        Line::from("██║     ██╔═══██╗██║ ██╔╝██║"),
        Line::from("██║     ██║   ██║█████╔╝ ██║"),
        Line::from("██║     ██║   ██║██╔═██╗ ██║"),
        Line::from("███████╗╚██████╔╝██║  ██╗██║"),
        Line::from("╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝"),
    ];

    let ascii_widget = Paragraph::new(loki_ascii)
        .style(Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);

    f.render_widget(ascii_widget, area);
}

fn draw_navigation_hints(f: &mut Frame, area: Rect) {
    let hints_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // First line
            Constraint::Length(1), // Second line
            Constraint::Length(2), // Spacer
        ])
        .split(area);

    // First hint line
    let tab_hint = Line::from(vec![
        Span::styled("Tab", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("/"),
        Span::styled("Shift+Tab", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" to navigate between tabs"),
    ]);

    let tab_widget = Paragraph::new(tab_hint)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

    f.render_widget(tab_widget, hints_chunks[0]);

    // Second hint line
    let keybind_hint = Line::from(vec![
        Span::styled("F1", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" to view keybind map"),
    ]);

    let keybind_widget = Paragraph::new(keybind_hint)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

    f.render_widget(keybind_widget, hints_chunks[1]);

    // Second hint line
    let subtab_hint = Line::from(vec![
        Span::styled("Ctrl+J", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("/"),
        Span::styled("Ctrl+K", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" to navigate between sub-tabs"),
    ]);

    let keybind_widget = Paragraph::new(subtab_hint)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

    f.render_widget(keybind_widget, hints_chunks[2]);
}

/// Draw the home dashboard sub-tabs header
fn draw_home_dashboard_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = app.state.home_dashboard_tabs.tabs
        .iter()
        .map(|tab| Line::from(tab.name.clone()))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("🏠 Loki Dashboard"))
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default().add_modifier(Modifier::BOLD).bg(Color::Blue).fg(Color::White),
        )
        .select(app.state.home_dashboard_tabs.current_index);

    f.render_widget(tabs, area);
}

/// Draw the animated welcome screen with LOKI ASCII art and floating Nordic runes
fn draw_welcome_screen(f: &mut Frame, _app: &App, area: Rect) {
    // Get current time for animations
    let time_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f32();
    
    // Draw massive amount of floating Nordic runes across entire screen
    draw_floating_runes_full_screen(f, area, time_secs);
    
    // Calculate center area for massive LOKI ASCII art
    let center_area = centered_rect(80, 60, area);
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(25), // Top spacer
            Constraint::Length(12),     // Massive LOKI ASCII art
            Constraint::Percentage(63), // Bottom spacer
        ])
        .split(center_area);
    
    // Draw massive enhanced LOKI ASCII art with dynamic color effects
    draw_massive_loki_ascii(f, chunks[1], time_secs);
}

/// Draw massive amount of floating Nordic runes across the entire screen with free movement
fn draw_floating_runes_full_screen(f: &mut Frame, area: Rect, time_secs: f32) {
    if area.height < 3 || area.width < 10 { return; }
    
    // Complete Elder Futhark runes alphabet for maximum coverage
    let runes = [
        "ᚠ", "ᚢ", "ᚦ", "ᚨ", "ᚱ", "ᚲ", "ᚷ", "ᚹ", "ᚺ", "ᚾ", "ᛁ", "ᛃ", 
        "ᛈ", "ᛉ", "ᛊ", "ᛏ", "ᛒ", "ᛖ", "ᛗ", "ᛚ", "ᛜ", "ᛞ", "ᛟ", "ᚪ",
        "ᚫ", "ᚣ", "ᛡ", "ᛠ", "ᛤ", "ᛥ", "ᛦ", "ᛧ", "ᛨ", "ᛩ", "ᛪ", "᛫"
    ];
    
    let mut grid = vec![vec![' '; area.width as usize]; area.height as usize];
    let mut colors = vec![vec![Color::Black; area.width as usize]; area.height as usize];
    
    // Create 30+ floating runes with physics-based movement
    let num_runes = 35;
    for i in 0..num_runes {
        let rune_idx = i % runes.len();
        let rune = runes[rune_idx];
        
        // Unique movement parameters for each rune
        let speed_x = 0.3 + (i as f32) * 0.07;
        let speed_y = 0.2 + (i as f32) * 0.05;
        let phase_x = (i as f32) * 3.14159 / 5.0; // Better distributed starting positions
        let phase_y = (i as f32) * 3.14159 / 7.0;
        
        // Simple but varied movement patterns
        let t = time_secs;
        
        // Base position for each rune (spread across the screen)
        let base_x = (i as f32) / (num_runes as f32);
        let base_y = ((i * 7) % num_runes) as f32 / (num_runes as f32);
        
        // X movement: add oscillation to base position
        let x_oscillation = (t * speed_x + phase_x).sin() * 0.3;
        let x_normalized = (base_x + x_oscillation).max(0.0).min(0.99);
        
        // Y movement: different pattern for interesting paths
        let y_oscillation = (t * speed_y + phase_y).cos() * 0.3;
        let y_normalized = (base_y + y_oscillation).max(0.0).min(0.99);
        
        // Convert to grid coordinates with proper bounds
        let x_pos = (x_normalized * (area.width as f32)) as usize;
        let y_pos = (y_normalized * (area.height as f32)) as usize;
        
        // Ensure we're within bounds
        let x_pos = x_pos.min(area.width as usize - 1);
        let y_pos = y_pos.min(area.height as usize - 1);
        
        if y_pos < grid.len() && x_pos < grid[0].len() {
            // Dynamic color based on position and time
            let hue = (t * 50.0 + (i as f32) * 10.0 + x_pos as f32 + y_pos as f32) % 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
            
            grid[y_pos][x_pos] = rune.chars().next().unwrap_or(' ');
            colors[y_pos][x_pos] = Color::Rgb(r, g, b);
        }
    }
    
    // Convert grid to lines with colors
    let mut lines = Vec::new();
    for (y, row) in grid.iter().enumerate() {
        let mut spans = Vec::new();
        
        for (x, &ch) in row.iter().enumerate() {
            let color = colors[y][x];
            // Create a span for each character to preserve positioning
            if ch == ' ' {
                spans.push(Span::raw(" "));
            } else {
                spans.push(Span::styled(
                    ch.to_string(), 
                    Style::default().fg(color).add_modifier(Modifier::BOLD)
                ));
            }
        }
        
        lines.push(Line::from(spans));
    }
    
    let runes_widget = Paragraph::new(lines)
        .alignment(Alignment::Left);
    
    f.render_widget(runes_widget, area);
}

/// Convert HSV to RGB color values
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        5 => (c, 0.0, x),
        _ => (c, 0.0, x),
    };
    
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8
    )
}

/// Draw massive LOKI ASCII art with dynamic rainbow color effects
fn draw_massive_loki_ascii(f: &mut Frame, area: Rect, time_secs: f32) {
    // Massive LOKI ASCII art
    let loki_lines = vec![
        "██╗      ██████╗ ██╗  ██╗██╗",
        "██║     ██╔═══██╗██║ ██╔╝██║",
        "██║     ██║   ██║█████╔╝ ██║",
        "██║     ██║   ██║██╔═██╗ ██║",
        "███████╗╚██████╔╝██║  ██╗██║",
        "╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝"
    ];
    
    let mut styled_lines = Vec::new();
    
    for (line_idx, line) in loki_lines.iter().enumerate() {
        let mut spans = Vec::new();
        
        // Create a rainbow wave effect across each character
        for (char_idx, ch) in line.chars().enumerate() {
            // Calculate color based on position and time
            let wave_offset = char_idx as f32 * 0.1 + line_idx as f32 * 0.2;
            let hue = ((time_secs * 100.0 + wave_offset * 30.0) % 360.0).abs();
            
            // Add vertical wave effect
            let vertical_wave = (time_secs * 3.0 + line_idx as f32 * 0.5).sin() * 20.0;
            let adjusted_hue = (hue + vertical_wave) % 360.0;
            
            // Convert HSV to RGB for rainbow effect
            let saturation = 0.9 + (time_secs * 2.0 + char_idx as f32 * 0.1).sin() * 0.1;
            let brightness = 0.8 + (time_secs * 1.5 + char_idx as f32 * 0.05).cos() * 0.2;
            let (r, g, b) = hsv_to_rgb(adjusted_hue, saturation, brightness);
            
            let color = Color::Rgb(r, g, b);
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            ));
        }
        
        styled_lines.push(Line::from(spans));
    }
    
    let ascii_widget = Paragraph::new(styled_lines)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::NONE));
    
    f.render_widget(ascii_widget, area);
}


/// Draw welcome system status
fn draw_welcome_system_status(f: &mut Frame, app: &App, area: Rect) {
    // Get system status
    let (agents_status, memory_status, tools_status) = if let Some(ref system_connector) = app.system_connector {
        if let Ok(metrics) = system_connector.get_system_metrics() {
            (
                format!("{} agents", metrics.active_agents),
                format!("{:.1}% memory", metrics.memory_usage),
                format!("{} tools", metrics.tool_executions)
            )
        } else {
            ("0 agents".to_string(), "0.0% memory".to_string(), "0 tools".to_string())
        }
    } else {
        ("disconnected".to_string(), "disconnected".to_string(), "disconnected".to_string())
    };

    let status_line = Line::from(vec![
        Span::styled("⚡ ", Style::default().fg(Color::Yellow)),
        Span::styled(agents_status, Style::default().fg(Color::Green)),
        Span::styled(" • ", Style::default().fg(Color::DarkGray)),
        Span::styled("🧠 ", Style::default().fg(Color::Magenta)),
        Span::styled(memory_status, Style::default().fg(Color::Yellow)),
        Span::styled(" • ", Style::default().fg(Color::DarkGray)),
        Span::styled("🔧 ", Style::default().fg(Color::Cyan)),
        Span::styled(tools_status, Style::default().fg(Color::Blue)),
    ]);

    let status_widget = Paragraph::new(status_line)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::NONE));

    f.render_widget(status_widget, area);
}

// ===== Monitoring Functions (moved from utilities) =====

/// Draw monitoring management interface with enhanced visuals
fn draw_monitoring_management(f: &mut Frame, app: &App, area: Rect) {
    // Try to get real metrics from the cache
    let cache = app.state.utilities_manager.cached_metrics.read().unwrap();
    
    // Create main layout with header, controls hint, and content areas
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(2),  // Controls hint
            Constraint::Min(0),     // Main content
        ])
        .split(area);
    
    // Draw header with last update time
    let mut header_spans = vec![
        Span::styled(
            "📊 System Monitoring & Analytics",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        ),
    ];
    
    if let Some(last_update) = cache.last_update {
        let elapsed = Utc::now() - last_update;
        let staleness_color = if elapsed.num_seconds() > 10 {
            Color::Red
        } else if elapsed.num_seconds() > 5 {
            Color::Yellow
        } else {
            Color::Green
        };
        
        header_spans.push(Span::raw("  |  "));
        header_spans.push(Span::styled(
            format!("Last Update: {}s ago", elapsed.num_seconds()),
            Style::default().fg(staleness_color),
        ));
    }
    
    let header = Paragraph::new(Line::from(header_spans))
        .block(Block::default().borders(Borders::ALL).title("System Monitor"))
        .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);
    
    // Draw controls hint
    let controls = vec![
        Span::styled("[F5]", Style::default().fg(Color::Yellow)),
        Span::raw(" Refresh  "),
        Span::styled("[↑↓]", Style::default().fg(Color::Yellow)),
        Span::raw(" Navigate  "),
        Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
        Span::raw(" Details  "),
        Span::styled("[E]", Style::default().fg(Color::Yellow)),
        Span::raw(" Export  "),
        Span::styled("[C]", Style::default().fg(Color::Yellow)),
        Span::raw(" Clear History"),
    ];
    let controls_widget = Paragraph::new(Line::from(controls))
        .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
        .alignment(Alignment::Center);
    f.render_widget(controls_widget, chunks[1]);
    
    // Main content area
    if let Some(metrics) = &cache.system_metrics {
        // Create 4-panel layout
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[2]);
        
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_chunks[0]);
        
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_chunks[1]);
        
        // Panel 1: System Overview with CPU/Memory gauges
        draw_monitoring_system_overview(f, left_chunks[0], metrics, &cache);
        
        // Panel 2: Performance graphs
        draw_monitoring_performance_graphs(f, left_chunks[1], &cache);
        
        // Panel 3: Resources (Disk, Network, GPU)
        draw_monitoring_resources(f, right_chunks[0], metrics);
        
        // Panel 4: Alerts and warnings
        draw_monitoring_alerts(f, right_chunks[1], metrics);
        
    } else {
        // Loading state
        let loading = vec![
            Line::from(""),
            Line::from("⏳ System metrics loading..."),
            Line::from(""),
            Line::from("Please wait while real-time metrics are collected."),
        ];
        let paragraph = Paragraph::new(loading)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        f.render_widget(paragraph, chunks[2]);
    }
}

/// Draw system overview panel with gauges
fn draw_monitoring_system_overview(f: &mut Frame, area: Rect, metrics: &crate::monitoring::real_time::SystemMetrics, _cache: &UtilitiesCache) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title and info
            Constraint::Length(3),  // CPU gauge
            Constraint::Length(3),  // Memory gauge
            Constraint::Min(0),     // System info
        ])
        .split(area);
    
    // Title
    let title = Paragraph::new("🖥️ System Overview")
        .block(Block::default().borders(Borders::ALL).title("Overview"))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(title, chunks[0]);
    
    // CPU Gauge
    let cpu_color = if metrics.cpu.usage_percent > 80.0 {
        Color::Red
    } else if metrics.cpu.usage_percent > 60.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            "CPU: {} cores @ {}MHz",
            metrics.cpu.core_count,
            metrics.cpu.frequency_mhz
        )))
        .gauge_style(Style::default().fg(cpu_color).bg(Color::Black))
        .percent(metrics.cpu.usage_percent as u16)
        .label(format!("{:.1}%", metrics.cpu.usage_percent));
    f.render_widget(cpu_gauge, chunks[1]);
    
    // Memory Gauge
    let memory_color = if metrics.memory.usage_percent > 90.0 {
        Color::Red
    } else if metrics.memory.usage_percent > 75.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    
    let memory_gb_used = metrics.memory.used_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let memory_gb_total = metrics.memory.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    
    let memory_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Memory"))
        .gauge_style(Style::default().fg(memory_color).bg(Color::Black))
        .percent(metrics.memory.usage_percent as u16)
        .label(format!("{:.1}GB / {:.1}GB", memory_gb_used, memory_gb_total));
    f.render_widget(memory_gauge, chunks[2]);
    
    // System info
    let info_text = vec![
        Line::from(vec![
            Span::styled("Host: ", Style::default().fg(Color::Yellow)),
            Span::raw(&metrics.system.hostname),
        ]),
        Line::from(vec![
            Span::styled("OS: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{} {}", metrics.system.os_name, metrics.system.os_version)),
        ]),
        Line::from(vec![
            Span::styled("Uptime: ", Style::default().fg(Color::Yellow)),
            Span::raw(format_uptime(metrics.system.uptime)),
        ]),
    ];
    
    let info = Paragraph::new(info_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(info, chunks[3]);
}

/// Draw performance graphs panel
fn draw_monitoring_performance_graphs(f: &mut Frame, area: Rect, cache: &crate::tui::tabs::utilities::UtilitiesCache) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // CPU history
            Constraint::Percentage(50),  // Memory history
        ])
        .split(area);
    
    // CPU usage sparkline
    if let Some(history) = &cache.cpu_history {
        if !history.is_empty() {
            let cpu_data: Vec<u64> = history.iter()
                .map(|&v| (v * 100.0) as u64)
                .collect();
            
            let cpu_sparkline = Sparkline::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(format!("CPU Usage History ({}s)", history.len())))
                .data(&cpu_data)
                .style(Style::default().fg(Color::Cyan))
                .max(100);
            f.render_widget(cpu_sparkline, chunks[0]);
        }
    } else {
        let placeholder = Paragraph::new("CPU history data not available")
            .block(Block::default().borders(Borders::ALL).title("CPU Usage History"))
            .alignment(Alignment::Center);
        f.render_widget(placeholder, chunks[0]);
    }
    
    // Memory usage sparkline
    if let Some(history) = &cache.memory_history {
        if !history.is_empty() {
            let memory_data: Vec<u64> = history.iter()
                .map(|&v| (v * 100.0) as u64)
                .collect();
            
            let memory_sparkline = Sparkline::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(format!("Memory Usage History ({}s)", history.len())))
                .data(&memory_data)
                .style(Style::default().fg(Color::Green))
                .max(100);
            f.render_widget(memory_sparkline, chunks[1]);
        }
    } else {
        let placeholder = Paragraph::new("Memory history data not available")
            .block(Block::default().borders(Borders::ALL).title("Memory Usage History"))
            .alignment(Alignment::Center);
        f.render_widget(placeholder, chunks[1]);
    }
}

/// Draw resources panel
fn draw_monitoring_resources(f: &mut Frame, area: Rect, metrics: &crate::monitoring::real_time::SystemMetrics) {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("💾 Resources", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ];
    
    // Disk metrics
    lines.push(Line::from(vec![
        Span::styled("Disk: ", Style::default().fg(Color::Cyan)),
        Span::raw(format!(
            "{:.1}GB / {:.1}GB ({:.1}%)",
            metrics.disk.used_space_bytes as f64 / 1e9,
            metrics.disk.total_space_bytes as f64 / 1e9,
            metrics.disk.usage_percent
        )),
    ]));
    
    lines.push(Line::from(vec![
        Span::styled("I/O: ", Style::default().fg(Color::Cyan)),
        Span::raw(format!(
            "↓ {}/s ↑ {}/s",
            format_bytes(metrics.disk.io_read_bytes_per_sec),
            format_bytes(metrics.disk.io_write_bytes_per_sec)
        )),
    ]));
    
    lines.push(Line::from(""));
    
    // Network metrics
    lines.push(Line::from(vec![
        Span::styled("Network: ", Style::default().fg(Color::Blue)),
        Span::raw(format!(
            "↓ {}/s ↑ {}/s",
            format_bytes(metrics.network.bytes_received_per_sec),
            format_bytes(metrics.network.bytes_sent_per_sec)
        )),
    ]));
    
    lines.push(Line::from(""));
    
    // GPU metrics if available
    if let Some(gpu) = &metrics.gpu {
        for device in &gpu.devices {
            lines.push(Line::from(vec![
                Span::styled("GPU: ", Style::default().fg(Color::Magenta)),
                Span::raw(&device.name),
            ]));
            
            if let Some(util) = device.utilization_percent {
                lines.push(Line::from(vec![
                    Span::raw("  Util: "),
                    Span::styled(
                        format!("{:.1}%", util),
                        if util > 90.0 {
                            Style::default().fg(Color::Red)
                        } else if util > 70.0 {
                            Style::default().fg(Color::Yellow)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                    ),
                ]));
            }
            
            if device.memory_total_bytes > 0 {
                let gpu_mem_gb_used = device.memory_used_bytes as f64 / 1e9;
                let gpu_mem_gb_total = device.memory_total_bytes as f64 / 1e9;
                lines.push(Line::from(vec![
                    Span::raw("  Mem: "),
                    Span::raw(format!("{:.1}GB / {:.1}GB", gpu_mem_gb_used, gpu_mem_gb_total)),
                ]));
            }
        }
    }
    
    // Process metrics
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Process: ", Style::default().fg(Color::Green)),
        Span::raw(format!(
            "PID {} | CPU {:.1}% | Mem {:.1}MB",
            metrics.process.pid,
            metrics.process.cpu_usage_percent,
            metrics.process.memory_usage_bytes as f64 / 1e6
        )),
    ]));
    
    let resources = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Resources"))
        .wrap(Wrap { trim: true });
    f.render_widget(resources, area);
}

/// Draw alerts panel
fn draw_monitoring_alerts(f: &mut Frame, area: Rect, metrics: &crate::monitoring::real_time::SystemMetrics) {
    let mut alerts = vec![
        Line::from(vec![
            Span::styled("🚨 System Alerts", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ];
    
    let mut has_alerts = false;
    
    // CPU alert
    if metrics.cpu.usage_percent > 80.0 {
        alerts.push(Line::from(vec![
            Span::styled("⚠️  ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("High CPU usage: {:.1}%", metrics.cpu.usage_percent),
                Style::default().fg(Color::Red),
            ),
        ]));
        has_alerts = true;
    }
    
    // Memory alert
    if metrics.memory.usage_percent > 90.0 {
        alerts.push(Line::from(vec![
            Span::styled("⚠️  ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("Critical memory usage: {:.1}%", metrics.memory.usage_percent),
                Style::default().fg(Color::Red),
            ),
        ]));
        has_alerts = true;
    } else if metrics.memory.usage_percent > 80.0 {
        alerts.push(Line::from(vec![
            Span::styled("⚡ ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("High memory usage: {:.1}%", metrics.memory.usage_percent),
                Style::default().fg(Color::Yellow),
            ),
        ]));
        has_alerts = true;
    }
    
    // Disk alert
    if metrics.disk.usage_percent > 90.0 {
        alerts.push(Line::from(vec![
            Span::styled("⚠️  ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("Low disk space: {:.1}% used", metrics.disk.usage_percent),
                Style::default().fg(Color::Red),
            ),
        ]));
        has_alerts = true;
    }
    
    // Network anomaly detection (simple threshold)
    let network_total = metrics.network.bytes_received_per_sec + metrics.network.bytes_sent_per_sec;
    if network_total > 100_000_000 { // 100 MB/s
        alerts.push(Line::from(vec![
            Span::styled("⚡ ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("High network activity: {}/s", format_bytes(network_total)),
                Style::default().fg(Color::Yellow),
            ),
        ]));
        has_alerts = true;
    }
    
    if !has_alerts {
        alerts.push(Line::from(vec![
            Span::styled("✅ ", Style::default().fg(Color::Green)),
            Span::styled("All systems operating normally", Style::default().fg(Color::Green)),
        ]));
    }
    
    let alerts_widget = Paragraph::new(alerts)
        .block(Block::default().borders(Borders::ALL).title("Alerts"))
        .wrap(Wrap { trim: true });
    f.render_widget(alerts_widget, area);
}

/// Format bytes to human readable format
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    format!("{:.1}{}", size, UNITS[unit_idx])
}

/// Format uptime duration to human readable format
fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}
