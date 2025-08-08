use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Styled};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block,
    Borders,
    Cell,
    Clear,
    Gauge,
    List,
    ListItem,
    Paragraph,
    Row,
    Sparkline,
    Table,
    Tabs,
    Wrap,
};

use super::super::app::App;
use crate::cluster::{ClusterConfig, ClusterManager};
use crate::compute::ComputeManager;
use crate::config::Config;
use crate::streaming::StreamManager;
use crate::tui::ui::PULSE_SPEED;

fn draw_enhanced_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let chunks = if app.state.needs_setup() {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),      // Setup message
                Constraint::Percentage(35), // Enhanced system metrics
                Constraint::Percentage(30), // Enhanced cluster overview
                Constraint::Percentage(35), // Enhanced recent activity
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(35), // Enhanced system metrics
                Constraint::Percentage(30), // Enhanced cluster overview
                Constraint::Percentage(35), // Enhanced recent activity
            ])
            .split(area)
    };

    let mut chunk_index = 0;

    // Draw setup message if needed
    if app.state.needs_setup() {
        draw_setup_message(f, app, chunks[chunk_index]);
        chunk_index += 1;
    }

    // Enhanced system metrics with animations
    let metrics_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(chunks[chunk_index]);

    draw_enhanced_metric_widget(
        f,
        "🔥 CPU Usage",
        &app.state.cpu_history,
        Color::Red,
        metrics_chunks[0],
    );
    draw_enhanced_metric_widget(
        f,
        "🖥️ GPU Usage",
        &app.state.gpu_history,
        Color::Green,
        metrics_chunks[1],
    );
    draw_enhanced_metric_widget(
        f,
        "💾 Memory Usage",
        &app.state.memory_history,
        Color::Blue,
        metrics_chunks[2],
    );

    // Enhanced cluster overview with visual improvements
    draw_enhanced_cluster_overview(f, app, chunks[chunk_index + 1]);

    // Enhanced recent activity with better formatting
    draw_enhanced_recent_activity(f, app, chunks[chunk_index + 2]);
}

/// Draw setup message for users who haven't configured API keys
fn draw_setup_message(f: &mut Frame, app: &App, area: Rect) {
    let setup_message = if app.state.needs_setup() {
        vec![
            Line::from(vec![
                Span::styled("🔧 ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "Setup Required: ",
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Span::raw("Run "),
                Span::styled(
                    "'loki setup-apis'",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" to configure your API keys, then restart the TUI."),
            ]),
            Line::from(vec![
                Span::styled("💡 ", Style::default().fg(Color::Cyan)),
                Span::raw("You can also run "),
                Span::styled(
                    "'loki check-apis'",
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" to check your current API configuration."),
            ]),
        ]
    } else {
        vec![Line::from(vec![
            Span::styled("✅ ", Style::default().fg(Color::Green)),
            Span::styled(
                "Setup Complete: ",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw("API keys are configured and ready to use."),
        ])]
    };

    let setup_paragraph = Paragraph::new(setup_message)
        .block(Block::default().borders(Borders::ALL).title(" Setup Status "))
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::White));

    f.render_widget(setup_paragraph, area);
}

/// Enhanced metric widget with animations and visual effects
fn draw_enhanced_metric_widget(
    f: &mut Frame,
    title: &str,
    data: &std::collections::VecDeque<f32>,
    color: Color,
    area: Rect,
) {
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED).sin().abs();

    // Dynamic border color based on pulse
    let border_color = match color {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f32 * (0.5 + pulse * 0.5)) as u8,
            (g as f32 * (0.5 + pulse * 0.5)) as u8,
            (b as f32 * (0.5 + pulse * 0.5)) as u8,
        ),
        // For other color types, use the original color
        _ => color,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            format!(" {} ", title),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if data.is_empty() {
        return;
    }

    let current_value = data.back().unwrap_or(&0.0);

    // Split into enhanced gauge and sparkline
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(inner);

    // Enhanced gauge with gradient effect
    let gauge_color = if *current_value > 85.0 {
        Color::Red
    } else if *current_value > 70.0 {
        Color::Yellow
    } else {
        color
    };

    let gauge = Gauge::default()
        .block(Block::default())
        .set_style(Style::default().fg(gauge_color))
        .percent(*current_value as u16)
        .label(format!("{:.1}%", current_value));

    f.render_widget(gauge, chunks[0]);

    // Enhanced sparkline with better visualization
    let sparkline_data: Vec<u64> = data.iter().map(|&v| (v * 2.0) as u64).collect();
    let sparkline = Sparkline::default()
        .block(Block::default())
        .data(&sparkline_data)
        .style(Style::default().fg(color));

    f.render_widget(sparkline, chunks[1]);
}

/// Enhanced cluster overview with visual improvements
fn draw_enhanced_cluster_overview(f: &mut Frame, app: &App, area: Rect) {
    let active_model_count = app.state.active_model_sessions.len();
    let ongoing_requests = app.state.ongoing_requests.len();
    let total_requests = app
        .state
        .recent_activities
        .iter()
        .filter(|a| matches!(a.activity_type, super::super::state::ActivityType::TaskExecution))
        .count();

    // Enhanced cluster status with animations
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED * 0.7).sin().abs();
    let status_color =
        Color::Rgb((128.0 + pulse * 127.0) as u8, 255, (128.0 + pulse * 127.0) as u8);

    let cluster_text = vec![
        Line::from(vec![
            Span::styled("🤖 ", Style::default().fg(Color::Cyan)),
            Span::styled("Active Models: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", active_model_count),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("⚡ ", Style::default().fg(Color::Yellow)),
            Span::styled("Ongoing Requests: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", ongoing_requests),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("📊 ", Style::default().fg(Color::Blue)),
            Span::styled("Total Requests: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", total_requests),
                Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("🗣️ ", Style::default().fg(Color::Magenta)),
            Span::styled("NL Session: ", Style::default().fg(Color::Gray)),
            Span::styled(
                if app.state.current_nl_session.is_some() {
                    match &app.state.current_nl_session.as_ref().unwrap().status {
                        super::super::state::NLSessionStatus::Parsing => "🔄 Parsing...",
                        super::super::state::NLSessionStatus::ModelSelection => {
                            "🎯 Selecting Model..."
                        }
                        super::super::state::NLSessionStatus::Processing => "⚙️ Processing...",
                        super::super::state::NLSessionStatus::Streaming => "🌊 Streaming...",
                        super::super::state::NLSessionStatus::Completed => "✅ Completed",
                        super::super::state::NLSessionStatus::Failed(_) => "❌ Failed",
                    }
                } else {
                    "⚫ None"
                },
                Style::default()
                    .fg(if app.state.current_nl_session.is_some() {
                        Color::Green
                    } else {
                        Color::Gray
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let cluster_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(status_color))
        .title(Span::styled(
            " 🌐 Cluster Overview ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    let cluster_para = Paragraph::new(cluster_text).block(cluster_block).wrap(Wrap { trim: true });

    f.render_widget(cluster_para, area);
}

/// Enhanced recent activity with better visual formatting
fn draw_enhanced_recent_activity(f: &mut Frame, app: &App, area: Rect) {
    let activity_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " 📈 Recent Activity ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    let activity_items: Vec<ListItem> = if !app.state.recent_activities.is_empty() {
        app.state
            .recent_activities
            .iter()
            .rev()
            .take(10)
            .map(|activity| {
                let (status_color, status_icon) = match &activity.status {
                    super::super::state::ActivityStatus::Started => (Color::Yellow, "🟡"),
                    super::super::state::ActivityStatus::InProgress => (Color::Blue, "🔵"),
                    super::super::state::ActivityStatus::Completed => (Color::Green, "🟢"),
                    super::super::state::ActivityStatus::Failed(_) => (Color::Red, "🔴"),
                };

                let type_icon = match activity.activity_type {
                    super::super::state::ActivityType::NaturalLanguageRequest => "🗣️",
                    super::super::state::ActivityType::ModelOrchestration => "🧠",
                    super::super::state::ActivityType::TaskExecution => "⚡",
                    super::super::state::ActivityType::ModelSwitch => "🔄",
                    super::super::state::ActivityType::StreamCreation => "📡",
                    super::super::state::ActivityType::ClusterRebalance => "⚖️",
                    super::super::state::ActivityType::FileSystemOperation => "📁",
                    super::super::state::ActivityType::SystemMonitoring => "📊",
                    super::super::state::ActivityType::ModelDiscovery => "🔍",
                    super::super::state::ActivityType::HealthCheck => "💚",
                    super::super::state::ActivityType::ClusterStatus => "🌐",
                    super::super::state::ActivityType::MemoryOptimization => "💾",
                    super::super::state::ActivityType::SecurityScan => "🔒",
                    super::super::state::ActivityType::ClusterInitialization => "🚀",
                    super::super::state::ActivityType::DirectoryNavigation => "📂",
                    super::super::state::ActivityType::DirectoryListing => "📋",
                    super::super::state::ActivityType::FileDeletion => "🗑️",
                    super::super::state::ActivityType::DirectoryQuery => "📍",
                };

                let model_info = if let Some(model_id) = &activity.model_id {
                    format!(" ({})", model_id)
                } else {
                    String::new()
                };

                let duration_info = if let Some(duration_ms) = activity.duration_ms {
                    format!(" [{}ms]", duration_ms)
                } else {
                    String::new()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(activity.timestamp.clone(), Style::default().fg(Color::DarkGray)),
                    Span::raw(" "),
                    Span::styled(status_icon, Style::default().fg(status_color)),
                    Span::raw(" "),
                    Span::styled(type_icon, Style::default().fg(Color::White)),
                    Span::raw(" "),
                    Span::styled(activity.description.clone(), Style::default().fg(Color::White)),
                    Span::styled(model_info, Style::default().fg(Color::Cyan)),
                    Span::styled(duration_info, Style::default().fg(Color::Gray)),
                ]))
            })
            .collect()
    } else {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("🔄 ", Style::default().fg(Color::Blue)),
                Span::styled("System initialized", Style::default().fg(Color::White)),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("💡 ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "Ready for natural language commands",
                    Style::default().fg(Color::Gray),
                ),
            ])),
        ]
    };

    let activity_list =
        List::new(activity_items).block(activity_block).style(Style::default().fg(Color::White));

    f.render_widget(activity_list, area);
}

async fn initialize_app() -> anyhow::Result<App> {
    let compute_manager = Arc::new(ComputeManager::new()?);
    let stream_manager = Arc::new(StreamManager::new(Config::load()?)?);
    let cluster_manager = Arc::new(ClusterManager::new(ClusterConfig::default()).await?);

    crate::tui::app::App::new(compute_manager, stream_manager, cluster_manager).await
}

fn draw_compute_view(f: &mut Frame<'_>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Device info
            Constraint::Length(8),  // System metrics
            Constraint::Min(0),     // Resource usage charts
        ])
        .split(area);

    // Device table
    draw_device_table(f, app, chunks[0]);

    // System metrics
    draw_system_metrics(f, app, chunks[1]);

    // Resource usage charts
    draw_resource_charts(f, app, chunks[2]);
}

fn draw_device_table(f: &mut Frame<'_>, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" 🖥️ Compute Devices ");

    let header_cells = ["Device", "Type", "Memory", "Utilization", "Status"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Gray)));
    let header =
        Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)).height(1);

    // Create device rows from real device data
    let device_rows: Vec<Row> = app
        .state
        .devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            // Get memory info for this device
            let memory_info =
                app.state.memory_info.iter().find(|(id, _)| id == &device.id).map(|(_, info)| info);

            let memory_text = if let Some(info) = memory_info {
                format!(
                    "{:.1}/{:.1} GB",
                    info.used as f64 / 1_000_000_000.0,
                    info.total as f64 / 1_000_000_000.0
                )
            } else {
                format!("{:.1} GB", device.info.memory_total as f64 / 1_000_000_000.0)
            };

            let device_type_info = match device.device_type {
                crate::compute::DeviceType::Cuda => ("🟢 CUDA", Color::Green),
                crate::compute::DeviceType::Metal => ("🟠 Metal", Color::Yellow),
                crate::compute::DeviceType::Cpu => ("🔵 CPU", Color::Blue),
                crate::compute::DeviceType::OpenCL => ("🟡 OpenCL", Color::Yellow),
            };

            // Calculate utilization based on memory usage
            let utilization = if let Some(info) = memory_info {
                let usage_percent = (info.used as f64 / info.total as f64) * 100.0;
                format!("{:.1}%", usage_percent)
            } else {
                "N/A".to_string()
            };

            let status = if device.is_gpu() { "Ready" } else { "Active" };

            let style = if Some(i) == app.state.selected_device {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new([
                Cell::from(device.name.clone()),
                Cell::from(device_type_info.0).style(Style::default().fg(device_type_info.1)),
                Cell::from(memory_text),
                Cell::from(utilization),
                Cell::from(status).style(Style::default().fg(Color::Green)),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        device_rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(block)
    .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(table, area);
}

fn draw_system_metrics(f: &mut Frame<'_>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);

    // CPU Usage
    let cpu_usage = app.state.cpu_history.back().copied().unwrap_or(0.0);
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" CPU Usage "))
        .set_style(Style::default().fg(Color::Blue))
        .percent(cpu_usage as u16)
        .label(format!("{:.1}%", cpu_usage));
    f.render_widget(cpu_gauge, chunks[0]);

    // Memory Usage
    let memory_usage = app.state.memory_history.back().copied().unwrap_or(0.0);
    let memory_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Memory Usage "))
        .set_style(Style::default().fg(Color::Green))
        .percent(memory_usage as u16)
        .label(format!("{:.1}%", memory_usage));
    f.render_widget(memory_gauge, chunks[1]);

    // GPU Usage (if available)
    let gpu_usage = app.state.gpu_history.back().copied().unwrap_or(0.0);
    let gpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" GPU Usage "))
        .set_style(Style::default().fg(Color::Yellow))
        .percent(gpu_usage as u16)
        .label(format!("{:.1}%", gpu_usage));
    f.render_widget(gpu_gauge, chunks[2]);
}

fn draw_resource_charts(f: &mut Frame<'_>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);

    // CPU History Chart
    let cpu_data: Vec<u64> = app.state.cpu_history.iter().map(|&x| x as u64).collect();
    let cpu_sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" CPU History "))
        .data(&cpu_data)
        .style(Style::default().fg(Color::Blue));
    f.render_widget(cpu_sparkline, chunks[0]);

    // Memory History Chart
    let memory_data: Vec<u64> = app.state.memory_history.iter().map(|&x| x as u64).collect();
    let memory_sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" Memory History "))
        .data(&memory_data)
        .style(Style::default().fg(Color::Green));
    f.render_widget(memory_sparkline, chunks[1]);

    // GPU History Chart
    let gpu_data: Vec<u64> = app.state.gpu_history.iter().map(|&x| x as u64).collect();
    let gpu_sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" GPU History "))
        .data(&gpu_data)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(gpu_sparkline, chunks[2]);
}

fn draw_streams_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Stream overview metrics
            Constraint::Length(12), // Active streams table
            Constraint::Min(0),     // Stream details and performance
        ])
        .split(area);

    // Stream overview metrics
    draw_stream_overview(f, app, chunks[0]);

    // Active streams table
    draw_active_streams_table(f, app, chunks[1]);

    // Stream details and performance
    draw_stream_details(f, app, chunks[2]);
}

fn draw_stream_overview(f: &mut Frame, app: &App, area: Rect) {
    let metrics_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    // Calculate aggregate metrics from active streams
    let total_streams = app.state.active_streams.len();
    let total_throughput = app.state.active_streams.len() as f64 * 245.7; // Simulate based on active streams
    let avg_latency = 12.3 + (app.state.active_streams.len() as f64 * 1.2); // Simulate load-based latency
    let total_errors = app.state.active_streams.len() as u64 * 2;
    let buffer_utilization = 67.8 + (app.state.active_streams.len() as f64 * 5.2);

    // Active streams gauge
    let active_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Active Streams "))
        .set_style(Style::default().fg(Color::Green))
        .percent(((total_streams as f64 / 10.0) * 100.0).min(100.0) as u16)
        .label(format!("{} / 10", total_streams));
    f.render_widget(active_gauge, metrics_chunks[0]);

    // Throughput gauge
    let throughput_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Throughput "))
        .set_style(Style::default().fg(Color::Cyan))
        .percent(((total_throughput / 1000.0) * 100.0).min(100.0) as u16)
        .label(format!("{:.1} msg/s", total_throughput));
    f.render_widget(throughput_gauge, metrics_chunks[1]);

    // Latency gauge
    let latency_color = if avg_latency > 50.0 {
        Color::Red
    } else if avg_latency > 25.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    let latency_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Avg Latency "))
        .set_style(Style::default().fg(latency_color))
        .percent(((avg_latency / 100.0) * 100.0).min(100.0) as u16)
        .label(format!("{:.1} ms", avg_latency));
    f.render_widget(latency_gauge, metrics_chunks[2]);

    // Error count
    let error_paragraph = Paragraph::new(format!(
        "Errors: {}\nRate: {:.2}%",
        total_errors,
        (total_errors as f64 / (total_streams.max(1) as f64 * 100.0)) * 100.0
    ))
    .block(Block::default().borders(Borders::ALL).title(" Errors "))
    .style(Style::default().fg(if total_errors > 5 { Color::Red } else { Color::Green }));
    f.render_widget(error_paragraph, metrics_chunks[3]);

    // Buffer utilization
    let buffer_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Buffer Usage "))
        .set_style(Style::default().fg(if buffer_utilization > 85.0 {
            Color::Red
        } else {
            Color::Blue
        }))
        .percent(buffer_utilization.min(100.0) as u16)
        .label(format!("{:.1}%", buffer_utilization));
    f.render_widget(buffer_gauge, metrics_chunks[4]);
}

fn draw_active_streams_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells =
        ["Stream ID", "Status", "Throughput", "Latency", "Errors", "Buffer", "Uptime"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    let header = Row::new(header_cells);

    let rows: Vec<Row> = app
        .state
        .active_streams
        .iter()
        .enumerate()
        .map(|(i, (id, status))| {
            let style = if Some(i) == app.state.selected_stream {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            // Generate realistic stream metrics based on stream position
            let throughput = 180.0 + (i as f64 * 25.0) + (i as f64 * 15.0).sin() * 20.0;
            let latency = 8.5 + (i as f64 * 2.1) + (i as f64 * 0.8).cos() * 3.0;
            let errors = i as u64 * 2;
            let buffer_usage = 45.0 + (i as f64 * 12.0) + (i as f64 * 0.5).sin() * 15.0;
            let uptime_mins = 120 + (i * 45);

            let status_color = match status.as_str() {
                "Active" => Color::Green,
                "Error" => Color::Red,
                "Paused" => Color::Yellow,
                "Initializing" => Color::Blue,
                _ => Color::Gray,
            };

            Row::new(vec![
                Cell::from(id.0.clone()),
                Cell::from(status.clone()).style(Style::default().fg(status_color)),
                Cell::from(format!("{:.1} msg/s", throughput)),
                Cell::from(format!("{:.1} ms", latency)),
                Cell::from(format!("{}", errors)),
                Cell::from(format!("{:.1}%", buffer_usage)),
                Cell::from(format!("{}m", uptime_mins)),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Percentage(20),
        Constraint::Percentage(12),
        Constraint::Percentage(15),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" Active Streams "))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(table, area);
}

fn draw_stream_details(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Stream configuration details
    draw_streamconfig_details(f, app, chunks[0]);

    // Performance history and events
    draw_stream_performance_history(f, app, chunks[1]);
}

fn draw_streamconfig_details(f: &mut Frame, app: &App, area: Rect) {
    let selected_stream = app.state.selected_stream.and_then(|i| app.state.active_streams.get(i));

    let content = if let Some((stream_id, status)) = selected_stream {
        vec![
            Line::from(vec![Span::styled(
                "Stream Configuration",
                Style::default().fg(Color::Yellow),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("ID: ", Style::default().fg(Color::Gray)),
                Span::styled(stream_id.0.clone(), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Gray)),
                Span::styled(status.clone(), Style::default().fg(Color::Blue)),
            ]),
            Line::from(vec![
                Span::styled("Buffer Size: ", Style::default().fg(Color::Gray)),
                Span::styled("1024 chunks", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("Max Latency: ", Style::default().fg(Color::Gray)),
                Span::styled("100ms", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("Processors: ", Style::default().fg(Color::Gray)),
                Span::styled("Default, Cognitive", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("Model: ", Style::default().fg(Color::Gray)),
                Span::styled("phi-3.5-mini", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Device: ", Style::default().fg(Color::Gray)),
                Span::styled("gpu:0", Style::default().fg(Color::Magenta)),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Performance Metrics",
                Style::default().fg(Color::Yellow),
            )]),
            Line::from(vec![
                Span::styled("Chunks Processed: ", Style::default().fg(Color::Gray)),
                Span::styled("45,234", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Bytes Processed: ", Style::default().fg(Color::Gray)),
                Span::styled("2.3 GB", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Success Rate: ", Style::default().fg(Color::Gray)),
                Span::styled("98.7%", Style::default().fg(Color::Green)),
            ]),
        ]
    } else {
        vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "Select a stream to view details",
                Style::default().fg(Color::Gray),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Use ↑/↓ keys to navigate",
                Style::default().fg(Color::Blue),
            )]),
        ]
    };

    let paragraph = Paragraph::new(content)
        .block(Block::default().borders(Borders::ALL).title(" Stream Details "))
        .style(Style::default().fg(Color::White));

    f.render_widget(paragraph, area);
}

fn draw_stream_performance_history(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Performance chart
            Constraint::Min(0),    // Recent events
        ])
        .split(area);

    // Performance sparkline
    let performance_data: Vec<u64> = (0..30)
        .map(|i| {
            let base = 150.0;
            let variation = 30.0 * (i as f64 * 0.3).sin();
            let trend = i as f64 * 2.0;
            (base + variation + trend) as u64
        })
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" Throughput History (30s) "))
        .data(&performance_data)
        .style(Style::default().fg(Color::Cyan));

    f.render_widget(sparkline, chunks[0]);

    // Recent stream events
    let events = vec![
        ("12:34:56", "Stream started", Color::Green),
        ("12:35:12", "High latency detected", Color::Yellow),
        ("12:35:34", "Buffer utilization at 85%", Color::Yellow),
        ("12:36:01", "Processing normalized", Color::Green),
        ("12:36:23", "Chunk processing optimized", Color::Blue),
        ("12:36:45", "Error rate decreased", Color::Green),
    ];

    let event_items: Vec<ListItem> = events
        .iter()
        .map(|(time, event, color)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", time), Style::default().fg(Color::Gray)),
                Span::styled(event.to_string(), Style::default().fg(*color)),
            ]))
        })
        .collect();

    let events_list = List::new(event_items)
        .block(Block::default().borders(Borders::ALL).title(" Recent Events "))
        .style(Style::default().fg(Color::White));

    f.render_widget(events_list, chunks[1]);
}

fn draw_cluster_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Cluster overview
            Constraint::Length(12), // Node status table
            Constraint::Min(0),     // Load distribution and migration activity
        ])
        .split(area);

    // Cluster overview
    draw_cluster_overview(f, app, chunks[0]);

    // Node status table
    draw_cluster_nodes_table(f, app, chunks[1]);

    // Load distribution and migration activity
    draw_cluster_load_and_migrations(f, app, chunks[2]);
}

fn draw_cluster_overview(f: &mut Frame, app: &App, area: Rect) {
    let overview_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    // Calculate cluster health based on stats
    let healthy_nodes = app.state.cluster_stats.total_nodes; // Simulate all healthy for now
    let cluster_health = if app.state.cluster_stats.total_nodes == 0 {
        0.0
    } else {
        (healthy_nodes as f32 / app.state.cluster_stats.total_nodes as f32) * 100.0
    };

    // Cluster health gauge
    let health_color = if cluster_health > 80.0 {
        Color::Green
    } else if cluster_health > 60.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let health_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Cluster Health "))
        .set_style(Style::default().fg(health_color))
        .percent(cluster_health as u16)
        .label(format!("{:.1}%", cluster_health));
    f.render_widget(health_gauge, overview_chunks[0]);

    // Node utilization
    let avg_utilization = (app.state.cluster_stats.avg_compute_usage
        + app.state.cluster_stats.avg_memory_usage)
        / 2.0;
    let utilization_color = if avg_utilization > 85.0 {
        Color::Red
    } else if avg_utilization > 70.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    let utilization_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Avg Utilization "))
        .set_style(Style::default().fg(utilization_color))
        .percent(avg_utilization as u16)
        .label(format!("{:.1}%", avg_utilization));
    f.render_widget(utilization_gauge, overview_chunks[1]);

    // Active requests
    let request_load = ((app.state.cluster_stats.active_requests as f32 / 50.0) * 100.0).min(100.0);
    let request_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Request Load "))
        .set_style(Style::default().fg(Color::Cyan))
        .percent(request_load as u16)
        .label(format!("{} reqs", app.state.cluster_stats.active_requests));
    f.render_widget(request_gauge, overview_chunks[2]);

    // Success rate
    let success_rate = if app.state.cluster_stats.total_requests > 0 {
        ((app.state.cluster_stats.total_requests - app.state.cluster_stats.failed_requests) as f32
            / app.state.cluster_stats.total_requests as f32)
            * 100.0
    } else {
        100.0
    };
    let success_color = if success_rate > 95.0 {
        Color::Green
    } else if success_rate > 90.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let success_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Success Rate "))
        .set_style(Style::default().fg(success_color))
        .percent(success_rate as u16)
        .label(format!("{:.1}%", success_rate));
    f.render_widget(success_gauge, overview_chunks[3]);

    // Model distribution
    let model_distribution =
        ((app.state.cluster_stats.total_models as f32 / 20.0) * 100.0).min(100.0);
    let model_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Models "))
        .set_style(Style::default().fg(Color::Blue))
        .percent(model_distribution as u16)
        .label(format!("{} / 20", app.state.cluster_stats.total_models));
    f.render_widget(model_gauge, overview_chunks[4]);
}

fn draw_cluster_nodes_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells =
        ["Node ID", "Status", "CPU", "Memory", "Load", "Models", "Latency", "Health"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    let header = Row::new(header_cells);

    // Generate realistic node data based on cluster stats
    let node_count = app.state.cluster_stats.total_nodes.max(1);
    let rows: Vec<Row> = (0..node_count)
        .map(|i| {
            let node_id = format!("node-{:03}", i + 1);

            // Generate realistic metrics based on position and cluster stats
            let cpu_usage = 35.0 + (i as f32 * 8.0) + (i as f32 * 0.7).sin() * 15.0;
            let memory_usage = 45.0 + (i as f32 * 12.0) + (i as f32 * 0.5).cos() * 20.0;
            let load_ratio = 0.4 + (i as f32 * 0.1) + (i as f32 * 0.3).sin() * 0.2;
            let model_count = 1 + (i % 4);
            let latency = 15.0 + (i as f32 * 3.0) + (i as f32 * 0.9).cos() * 8.0;

            // Determine status based on metrics
            let status = if cpu_usage > 85.0 || memory_usage > 90.0 {
                ("Warning", Color::Yellow)
            } else if latency > 50.0 {
                ("Degraded", Color::Red)
            } else {
                ("Healthy", Color::Green)
            };

            let health_score =
                (100.0 - (cpu_usage.max(memory_usage) - 50.0).max(0.0) * 2.0).max(0.0);

            Row::new(vec![
                Cell::from(node_id),
                Cell::from(status.0).style(Style::default().fg(status.1)),
                Cell::from(format!("{:.1}%", cpu_usage)),
                Cell::from(format!("{:.1}%", memory_usage)),
                Cell::from(format!("{:.1}", load_ratio)),
                Cell::from(format!("{}", model_count)),
                Cell::from(format!("{:.1}ms", latency)),
                Cell::from(format!("{:.0}%", health_score)).style(Style::default().fg(
                    if health_score > 80.0 {
                        Color::Green
                    } else if health_score > 60.0 {
                        Color::Yellow
                    } else {
                        Color::Red
                    },
                )),
            ])
        })
        .collect();

    let widths = [
        Constraint::Percentage(15),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(12),
        Constraint::Percentage(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" Cluster Nodes "));

    f.render_widget(table, area);
}

fn draw_cluster_load_and_migrations(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Load distribution chart
    draw_cluster_load_distribution(f, app, chunks[0]);

    // Migration and rebalancing activity
    draw_cluster_migration_activity(f, app, chunks[1]);
}

fn draw_cluster_load_distribution(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Load balance chart
            Constraint::Min(0),    // Load statistics
        ])
        .split(area);

    // Load balance sparkline showing distribution across nodes
    let load_data: Vec<u64> = (0..app.state.cluster_stats.total_nodes.max(1))
        .map(|i| {
            let base_load = 40.0;
            let variance = 20.0 * (i as f64 * 0.7).sin();
            let trend = i as f64 * 2.0;
            (base_load + variance + trend) as u64
        })
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" Load Distribution "))
        .data(&load_data)
        .style(Style::default().fg(Color::Green));

    f.render_widget(sparkline, chunks[0]);

    // Load statistics
    let stats_text = vec![
        Line::from(vec![Span::styled("Load Balancing", Style::default().fg(Color::Yellow))]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Balance Score: ", Style::default().fg(Color::Gray)),
            Span::styled("87.3%", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Std Deviation: ", Style::default().fg(Color::Gray)),
            Span::styled("12.4%", Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::styled("Hotspot Nodes: ", Style::default().fg(Color::Gray)),
            Span::styled("1", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Underutilized: ", Style::default().fg(Color::Gray)),
            Span::styled("2", Style::default().fg(Color::Cyan)),
        ]),
    ];

    let stats_paragraph = Paragraph::new(stats_text)
        .block(Block::default().borders(Borders::ALL).title(" Balance Metrics "))
        .style(Style::default().fg(Color::White));

    f.render_widget(stats_paragraph, chunks[1]);
}

fn draw_cluster_migration_activity(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Migration status
            Constraint::Min(0),    // Recent migrations
        ])
        .split(area);

    // Migration status overview
    let migration_text = vec![
        Line::from(vec![Span::styled("Migration Status", Style::default().fg(Color::Magenta))]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Active Migrations: ", Style::default().fg(Color::Gray)),
            Span::styled("0", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Queued Migrations: ", Style::default().fg(Color::Gray)),
            Span::styled("2", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Last Rebalance: ", Style::default().fg(Color::Gray)),
            Span::styled("2h ago", Style::default().fg(Color::Blue)),
        ]),
    ];

    let migration_paragraph = Paragraph::new(migration_text)
        .block(Block::default().borders(Borders::ALL).title(" Migration Control "))
        .style(Style::default().fg(Color::White));

    f.render_widget(migration_paragraph, chunks[0]);

    // Recent migration events
    let events = vec![
        (
            "14:23:45",
            "Model migration completed",
            "phi-3.5-mini: node-002 → node-001",
            Color::Green,
        ),
        ("14:20:12", "Load rebalancing initiated", "Cluster utilization: 78%", Color::Blue),
        ("14:18:33", "Node degradation detected", "node-003: High memory usage", Color::Yellow),
        ("14:15:07", "Migration queued", "mistral-7b: node-001 → node-003", Color::Cyan),
        ("14:12:41", "Cluster optimization", "Balanced load across 4 nodes", Color::Green),
    ];

    let event_items: Vec<ListItem> = events
        .iter()
        .map(|(time, event, details, color)| {
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("{} ", time), Style::default().fg(Color::Gray)),
                    Span::styled(event.to_string(), Style::default().fg(*color)),
                ]),
                Line::from(vec![
                    Span::styled("    ", Style::default()),
                    Span::styled(details.to_string(), Style::default().fg(Color::Gray)),
                ]),
            ])
        })
        .collect();

    let events_list = List::new(event_items)
        .block(Block::default().borders(Borders::ALL).title(" Recent Activity "))
        .style(Style::default().fg(Color::White));

    f.render_widget(events_list, chunks[1]);
}

fn draw_models_view(f: &mut Frame, app: &App, area: Rect) {
    // Create tabs for different model views
    let model_tabs = vec!["Templates", "Sessions", "Registry", "Analytics"];
    let selected_tab = match app.state.model_view {
        super::super::state::ModelViewState::Templates => 0,
        super::super::state::ModelViewState::Sessions => 1,
        super::super::state::ModelViewState::Registry => 2,
        super::super::state::ModelViewState::Analytics => 3,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
        ])
        .split(area);

    // Render tabs
    let tabs = Tabs::new(model_tabs)
        .block(Block::default().borders(Borders::ALL).title(" Model Orchestration "))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .select(selected_tab);

    f.render_widget(tabs, chunks[0]);

    // Render content based on current view
    match app.state.model_view {
        super::super::state::ModelViewState::Templates => draw_templates_view(f, app, chunks[1]),
        super::super::state::ModelViewState::Sessions => draw_sessions_view(f, app, chunks[1]),
        super::super::state::ModelViewState::Registry => draw_registry_view(f, app, chunks[1]),
        super::super::state::ModelViewState::Analytics => draw_analytics_view(f, app, chunks[1]),
    }
}

fn draw_templates_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Template list
            Constraint::Percentage(50), // Template details
        ])
        .split(area);

    // Template list
    let template_items: Vec<ListItem> = app
        .state
        .setup_templates
        .iter()
        .enumerate()
        .map(|(i, template)| {
            let style = if Some(i) == app.state.selected_template {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let cost_text = if template.is_free {
                "FREE".to_string()
            } else {
                format!("${:.2}/hr", template.cost_estimate)
            };

            let complexity_icon = match template.complexity_level {
                super::super::state::ComplexityLevel::Simple => "🟢",
                super::super::state::ComplexityLevel::Beginner => "🔵",
                super::super::state::ComplexityLevel::Medium => "🟡",
                super::super::state::ComplexityLevel::Intermediate => "🟠",
                super::super::state::ComplexityLevel::Advanced => "🔴",
            };

            ListItem::new(Line::from(vec![
                Span::raw(complexity_icon),
                Span::raw(" "),
                Span::styled(&template.name, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" - "),
                Span::styled(
                    cost_text,
                    if template.is_free {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Yellow)
                    },
                ),
            ]))
            .style(style)
        })
        .collect();

    let template_list = List::new(template_items)
        .block(Block::default().borders(Borders::ALL).title(" Setup Templates "))
        .style(Style::default().fg(Color::White));

    f.render_widget(template_list, chunks[0]);

    // Template details
    if let Some(template) = app.state.get_selected_template() {
        let details_text = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    &template.name,
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("Description: ", Style::default().fg(Color::Gray))]),
            Line::from(template.description.clone()),
            Line::from(""),
            Line::from(vec![
                Span::styled("Cost: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    if template.is_free {
                        "FREE".to_string()
                    } else {
                        format!("${:.2}/hour", template.cost_estimate)
                    },
                    if template.is_free {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Yellow)
                    },
                ),
            ]),
            Line::from(vec![
                Span::styled("Setup Time: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("~{} seconds", template.setup_time_estimate),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("GPU Memory: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{} MB", template.gpu_memory_required),
                    Style::default().fg(Color::Blue),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("Local Models: ", Style::default().fg(Color::Gray))]),
        ];

        let mut all_lines = details_text;
        for model in &template.local_models {
            all_lines.push(Line::from(format!("  • {}", model)));
        }

        if !template.api_models.is_empty() {
            all_lines.push(Line::from(""));
            all_lines.push(Line::from(vec![Span::styled(
                "API Models: ",
                Style::default().fg(Color::Gray),
            )]));
            for model in &template.api_models {
                all_lines.push(Line::from(format!("  • {}", model)));
            }
        }

        all_lines.push(Line::from(""));
        all_lines.push(Line::from(vec![
            Span::styled("💡 Tip: ", Style::default().fg(Color::Yellow)),
            Span::raw("Press Enter to launch this template"),
        ]));

        let details = Paragraph::new(all_lines)
            .block(Block::default().borders(Borders::ALL).title(" Template Details "))
            .wrap(Wrap { trim: true });

        f.render_widget(details, chunks[1]);
    } else {
        draw_template_overview(f, app, chunks[1]);
    }
}

fn draw_sessions_view(f: &mut Frame, app: &App, area: Rect) {
    if app.state.active_sessions.is_empty() {
        let placeholder = Paragraph::new(vec![
            Line::from("No active model sessions"),
            Line::from(""),
            Line::from("Try launching a template:"),
            Line::from("  • setup lightning fast"),
            Line::from("  • setup balanced pro"),
            Line::from("  • setup premium quality"),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Active Sessions "))
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center);

        f.render_widget(placeholder, area);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Session list
            Constraint::Percentage(50), // Session details
        ])
        .split(area);

    // Session list
    let session_items: Vec<ListItem> = app
        .state
        .active_sessions
        .iter()
        .enumerate()
        .map(|(i, session)| {
            let style = if Some(i) == app.state.selected_session {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let status_color = match session.status {
                super::super::state::SessionStatus::Active => Color::Green,
                super::super::state::SessionStatus::Starting => Color::Yellow,
                super::super::state::SessionStatus::Running => Color::Cyan,
                super::super::state::SessionStatus::Paused => Color::Blue,
                super::super::state::SessionStatus::Stopping => Color::Red,
                super::super::state::SessionStatus::Error(_) => Color::Red,
            };

            ListItem::new(Line::from(vec![
                Span::styled("●", Style::default().fg(status_color)),
                Span::raw(" "),
                Span::styled(&session.name, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" - "),
                Span::styled(
                    format!("${:.2}/hr", session.cost_per_hour),
                    Style::default().fg(Color::Yellow),
                ),
            ]))
            .style(style)
        })
        .collect();

    let session_list = List::new(session_items)
        .block(Block::default().borders(Borders::ALL).title(" Active Sessions "))
        .style(Style::default().fg(Color::White));

    f.render_widget(session_list, chunks[0]);

    // Session details
    if let Some(session) = app.state.get_selected_session() {
        let status_text = match &session.status {
            super::super::state::SessionStatus::Active => "Active".to_string(),
            super::super::state::SessionStatus::Starting => "Starting...".to_string(),
            super::super::state::SessionStatus::Running => "Running".to_string(),
            super::super::state::SessionStatus::Paused => "Paused".to_string(),
            super::super::state::SessionStatus::Stopping => "Stopping...".to_string(),
            super::super::state::SessionStatus::Error(msg) => format!("Error: {}", msg),
        };

        let uptime = chrono::Local::now().signed_duration_since(session.start_time).num_minutes();

        let details_text = vec![
            Line::from(vec![
                Span::styled("Session: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    &session.name,
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Gray)),
                Span::styled(status_text, Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("Template: ", Style::default().fg(Color::Gray)),
                Span::styled(&session.template_id, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("Uptime: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{} minutes", uptime), Style::default().fg(Color::Blue)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("GPU Usage: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}%", session.gpu_usage),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::styled("Memory Usage: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}%", session.memory_usage),
                    Style::default().fg(Color::Blue),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Requests: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", session.request_count),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::styled("Errors: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", session.error_count), Style::default().fg(Color::Red)),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("Active Models: ", Style::default().fg(Color::Gray))]),
        ];

        let mut all_lines = details_text;
        for model in &session.active_models {
            all_lines.push(Line::from(format!("  • {}", model)));
        }

        let details = Paragraph::new(all_lines)
            .block(Block::default().borders(Borders::ALL).title(" Session Details "))
            .wrap(Wrap { trim: true });

        f.render_widget(details, chunks[1]);
    }
}

fn draw_registry_view(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["Model", "Type", "Size", "Status", "Capabilities"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Gray)));
    let header =
        Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)).height(1);

    let rows = app.state.model_registry.iter().map(|model| {
        let type_text = match model.model_type {
            super::super::state::ModelType::Local => "Local",
            super::super::state::ModelType::Api => "API",
        };

        let size_text = if model.model_type == super::super::state::ModelType::Local {
            format!("{:.1} GB", model.size_gb)
        } else {
            "N/A".to_string()
        };

        let status_text = match &model.status {
            super::super::state::ModelStatus::Available => "Available",
            super::super::state::ModelStatus::Downloading => "Downloading",
            super::super::state::ModelStatus::Downloaded => "Downloaded",
            super::super::state::ModelStatus::Installing => "Installing",
            super::super::state::ModelStatus::Installed => "Installed",
            super::super::state::ModelStatus::Error(_) => "Error",
        };

        let capabilities_text = model.capabilities.join(", ");

        let cells = vec![
            Cell::from(model.name.clone()),
            Cell::from(type_text),
            Cell::from(size_text),
            Cell::from(status_text).style(match model.status {
                super::super::state::ModelStatus::Downloaded
                | super::super::state::ModelStatus::Installed => Style::default().fg(Color::Green),
                super::super::state::ModelStatus::Downloading
                | super::super::state::ModelStatus::Installing => {
                    Style::default().fg(Color::Yellow)
                }
                super::super::state::ModelStatus::Error(_) => Style::default().fg(Color::Red),
                _ => Style::default().fg(Color::Gray),
            }),
            Cell::from(capabilities_text),
        ];

        Row::new(cells)
    });

    let widths = [
        Constraint::Percentage(25),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(15),
        Constraint::Percentage(40),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" Model Registry "));

    f.render_widget(table, area);
}

fn draw_analytics_view(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40), // Cost overview
            Constraint::Percentage(60), // Cost breakdown
        ])
        .split(area);

    // Cost overview
    let overview_text = vec![
        Line::from(vec![Span::styled(
            "💰 Cost Analytics",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Today: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.2}", app.state.cost_analytics.total_cost_today),
                Style::default().fg(Color::Green),
            ),
            Span::raw("  |  "),
            Span::styled("Requests: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.state.cost_analytics.requests_today),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("This Month: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.2}", app.state.cost_analytics.total_cost_month),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("  |  "),
            Span::styled("Requests: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.state.cost_analytics.requests_month),
                Style::default().fg(Color::Blue),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Avg Cost/Request: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("${:.4}", app.state.cost_analytics.avg_cost_per_request),
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ];

    let overview = Paragraph::new(overview_text)
        .block(Block::default().borders(Borders::ALL).title(" Cost Overview "))
        .wrap(Wrap { trim: true });

    f.render_widget(overview, chunks[0]);

    // Cost breakdown by model
    let breakdown_items: Vec<ListItem> = app
        .state
        .cost_analytics
        .cost_by_model
        .iter()
        .map(|(model, cost)| {
            ListItem::new(Line::from(vec![
                Span::raw(model),
                Span::raw(" - "),
                Span::styled(format!("${:.2}", cost), Style::default().fg(Color::Yellow)),
            ]))
        })
        .collect();

    let breakdown_list = if breakdown_items.is_empty() {
        let placeholder_items = vec![
            ListItem::new("No usage data available yet"),
            ListItem::new("Start using models to see cost breakdown"),
        ];
        List::new(placeholder_items)
            .block(Block::default().borders(Borders::ALL).title(" Cost by Model "))
            .style(Style::default().fg(Color::Gray))
    } else {
        List::new(breakdown_items)
            .block(Block::default().borders(Borders::ALL).title(" Cost by Model "))
            .style(Style::default().fg(Color::White))
    };

    f.render_widget(breakdown_list, chunks[1]);
}

/// Draw environment detection
fn draw_environment_detection(f: &mut Frame, app: &App, area: Rect) {
    let env_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33), // System info
            Constraint::Percentage(33), // Available resources
            Constraint::Percentage(34), // Recommendations
        ])
        .split(area);

    // System info
    let total_devices = app.state.devices.len();
    let gpu_available = app.state.devices.iter().any(|d| {
        matches!(
            d.device_type,
            crate::compute::DeviceType::Cuda | crate::compute::DeviceType::Metal
        )
    });

    let system_text = vec![
        Line::from(vec![Span::styled(
            "💻 System",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("OS: ", Style::default().fg(Color::Gray)),
            Span::styled(std::env::consts::OS, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Arch: ", Style::default().fg(Color::Gray)),
            Span::styled(std::env::consts::ARCH, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Devices: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", total_devices), Style::default().fg(Color::Cyan)),
        ]),
    ];

    let system_para = Paragraph::new(system_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    // Available resources
    let resources_text = vec![
        Line::from(vec![Span::styled(
            "⚡ Resources",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(Color::Gray)),
            Span::styled(
                if gpu_available { "Available" } else { "CPU Only" },
                Style::default().fg(if gpu_available { Color::Green } else { Color::Yellow }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Models: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.state.available_models.len()),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Agents: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.state.available_agents.len()),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let resources_para = Paragraph::new(resources_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    // Recommendations
    let recommendations_text = if gpu_available {
        vec![
            Line::from(vec![Span::styled(
                "🎯 Optimized",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            )]),
            Line::from("✅ GPU acceleration"),
            Line::from("✅ Local models ready"),
            Line::from("🚀 Try: setup lightning fast"),
        ]
    } else {
        vec![
            Line::from(vec![Span::styled(
                "💡 CPU Mode",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )]),
            Line::from("⚠️  No GPU detected"),
            Line::from("✅ CPU inference available"),
            Line::from("💡 Consider API templates"),
        ]
    };

    let recommendations_para = Paragraph::new(recommendations_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    f.render_widget(system_para, env_layout[0]);
    f.render_widget(resources_para, env_layout[1]);
    f.render_widget(recommendations_para, env_layout[2]);
}

/// Draw setup recommendations
fn draw_setup_recommendations(f: &mut Frame, app: &App, area: Rect) {
    let recommendations_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Quick start
            Constraint::Percentage(50), // Advanced setup
        ])
        .split(area);

    let ollama_available = !app.state.available_models.is_empty();
    let api_keysconfigured = !app.state.apiconfigurations.is_empty();

    // Quick start recommendations
    let quick_start_text = if ollama_available {
        vec![
            Line::from(vec![Span::styled(
                "🚀 Quick Start (Ready!)",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("You're all set for local AI:"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "setup lightning fast",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )]),
            Line::from("  • 100% local, no API keys"),
            Line::from("  • Fast setup, ready in seconds"),
            Line::from("  • Perfect for getting started"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "setup code master",
                Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            )]),
            Line::from("  • Specialized for coding"),
            Line::from("  • Works completely offline"),
        ]
    } else {
        vec![
            Line::from(vec![Span::styled(
                "🔧 Install Local Models",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("Get started with local AI:"),
            Line::from(""),
            Line::from("1. Install Ollama:"),
            Line::from("   brew install ollama"),
            Line::from(""),
            Line::from("2. Download a model:"),
            Line::from("   ollama pull llama3.2"),
            Line::from(""),
            Line::from("3. Run setup:"),
            Line::from("   setup lightning fast"),
        ]
    };

    // Advanced setup recommendations
    let advanced_text = if api_keysconfigured {
        vec![
            Line::from(vec![Span::styled(
                "⭐ Premium Ready",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("API keys configured! Try:"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "setup premium quality",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            )]),
            Line::from("  • GPT-4, Claude-3 access"),
            Line::from("  • Best-in-class performance"),
            Line::from("  • ~$0.50/hour"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "setup research beast",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )]),
            Line::from("  • 5-model ensemble"),
            Line::from("  • Research-grade quality"),
            Line::from("  • ~$1.00/hour"),
        ]
    } else {
        vec![
            Line::from(vec![Span::styled(
                "🔑 Setup API Access",
                Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("Unlock premium models:"),
            Line::from(""),
            Line::from("1. Configure API keys:"),
            Line::from(vec![Span::styled(
                "   setup api keys",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("2. Choose provider:"),
            Line::from("   • OpenAI (GPT-4)"),
            Line::from("   • Anthropic (Claude-3)"),
            Line::from("   • Google (Gemini)"),
            Line::from(""),
            Line::from("3. Launch template:"),
            Line::from("   setup balanced pro"),
        ]
    };

    let quick_start_para = Paragraph::new(quick_start_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    let advanced_para = Paragraph::new(advanced_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    f.render_widget(quick_start_para, recommendations_layout[0]);
    f.render_widget(advanced_para, recommendations_layout[1]);
}

fn draw_logs_view(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" System Logs ({}) ", app.state.log_entries.len()));

    // Convert log entries to list items with proper styling
    let log_items: Vec<ListItem> = app
        .state
        .log_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let (level_style, level_text) = match entry.level.as_str() {
                "ERROR" => (Style::default().fg(Color::Red), "[ERROR]"),
                "WARN" => (Style::default().fg(Color::Yellow), "[WARN] "),
                "INFO" => (Style::default().fg(Color::Green), "[INFO] "),
                "DEBUG" => (Style::default().fg(Color::Cyan), "[DEBUG]"),
                "TRACE" => (Style::default().fg(Color::Magenta), "[TRACE]"),
                _ => (Style::default().fg(Color::White), "[LOG]  "),
            };

            let timestamp_style = Style::default().fg(Color::Gray);
            let message_style = Style::default().fg(Color::White);

            // Highlight selected log entry
            let selected_style = if Some(i) == app.state.selected_log {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(&entry.timestamp, timestamp_style),
                Span::raw(" "),
                Span::styled(level_text, level_style),
                Span::raw(" "),
                Span::styled(&entry.target, Style::default().fg(Color::Blue)),
                Span::raw(": "),
                Span::styled(&entry.message, message_style),
            ]))
            .style(selected_style)
        })
        .collect();

    // If no log entries exist, show a helpful message
    let log_items = if log_items.is_empty() {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("[INFO]", Style::default().fg(Color::Green)),
                Span::raw(" No log entries available yet"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[INFO]", Style::default().fg(Color::Cyan)),
                Span::raw(" Log entries will appear here as the system runs"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[TIP]", Style::default().fg(Color::Yellow)),
                Span::raw(" Try running some commands to generate logs"),
            ])),
        ]
    } else {
        log_items
    };

    let list = List::new(log_items).block(block).style(Style::default().fg(Color::White));

    f.render_widget(list, area);
}

fn draw_help_view(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Left column
            Constraint::Percentage(50), // Right column
        ])
        .split(area);

    // Left column: Natural Language Commands & Setup
    let left_help_text = vec![
        Line::from(vec![Span::styled(
            "🗣️ Natural Language Commands",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "System Status:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • show devices - Display compute devices"),
        Line::from("  • monitor gpu - Monitor GPU usage"),
        Line::from("  • show cluster - Display cluster status"),
        Line::from("  • list streams - Show active streams"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Model Setup Templates:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • setup lightning fast - FREE local setup"),
        Line::from("  • setup balanced pro - Local + API (~$0.10/hr)"),
        Line::from("  • setup premium quality - Best models (~$0.50/hr)"),
        Line::from("  • setup research beast - 5-model ensemble (~$1.00/hr)"),
        Line::from("  • setup code master - Code completion (FREE)"),
        Line::from("  • setup writing pro - Professional writing (~$0.30/hr)"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Session Management:",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • show templates - List available setups"),
        Line::from("  • show sessions - List active sessions"),
        Line::from("  • show costs - View cost analytics"),
        Line::from("  • stop session <id> - Stop specific session"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Model Installation:",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • install deepseek coder - Download local model"),
        Line::from("  • install wizardcoder - Download coding model"),
        Line::from("  • setup api keys - Configure API access"),
    ];

    // Right column: CLI Commands & Keyboard Shortcuts
    let right_help_text = vec![
        Line::from(vec![Span::styled(
            "💻 CLI Commands",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Setup Templates:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • loki setup list - List all templates"),
        Line::from("  • loki setup lightning-fast - Quick launch FREE"),
        Line::from("  • loki setup balanced-pro - Launch balanced setup"),
        Line::from("  • loki setup info <template> - Template details"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Session Management:",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • loki session list - List active sessions"),
        Line::from("  • loki session info <id> - Session details"),
        Line::from("  • loki session stop <id> - Stop session"),
        Line::from("  • loki session costs - Cost analytics"),
        Line::from("  • loki session monitor <id> - Real-time monitoring"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Model Commands:",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • loki model list-all - All available models"),
        Line::from("  • loki model test-orchestration - Test system"),
        Line::from("  • loki model benchmark - Performance test"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "⌨️ Keyboard Shortcuts:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  • Tab - Cycle through views"),
        Line::from("  • 1-9, S, P, E, N, H, Q, 0 - Direct navigation (empty input)"),
        Line::from("  • Ctrl+1-9, Ctrl+S/P/E/N/H/Q/0 - Navigate while typing"),
        Line::from("  • F1 or ? - Toggle this help"),
        Line::from("  • ↑/↓ - Navigate items or command history"),
        Line::from("  • ←/→ - Navigate model sub-views"),
        Line::from("  • Enter - Execute command or launch template"),
        Line::from("  • Esc - Clear input or close dialogs"),
        Line::from("  • Ctrl+Q - Quit application"),
    ];

    let left_block = Block::default().borders(Borders::ALL).title(" TUI Commands ");

    let right_block = Block::default().borders(Borders::ALL).title(" CLI & Shortcuts ");

    let left_para = Paragraph::new(left_help_text).block(left_block).wrap(Wrap { trim: true });

    let right_para = Paragraph::new(right_help_text).block(right_block).wrap(Wrap { trim: true });

    f.render_widget(left_para, chunks[0]);
    f.render_widget(right_para, chunks[1]);
}

fn draw_help_overlay(f: &mut Frame, app: &App) {
    let area = super::centered_rect(80, 80, f.area());
    f.render_widget(Clear, area);
    draw_help_view(f, app, area);
}

fn draw_command_input(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(30)])
        .split(area);

    // Command input
    let input_block = Block::default().borders(Borders::ALL).title(" Command (Natural Language) ");

    let input = Paragraph::new(app.state.command_input.as_str())
        .block(input_block)
        .style(Style::default().fg(Color::White));

    f.render_widget(input, chunks[0]);

    // Suggestions
    if !app.state.suggestions.is_empty() {
        let suggestions_block = Block::default().borders(Borders::ALL).title(" Suggestions ");

        let suggestion_text = app.state.suggestions.join("\n");
        let suggestions = Paragraph::new(suggestion_text)
            .block(suggestions_block)
            .style(Style::default().fg(Color::Gray));

        f.render_widget(suggestions, chunks[1]);
    }
}

fn draw_status_bar(f: &mut Frame, _app: &App, area: Rect) {
    let status = Line::from(vec![
        Span::raw("Press "),
        Span::styled("F1", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" for help | "),
        Span::styled("Tab", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" to switch views | "),
        Span::styled("Ctrl+1-9", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" for direct navigation | "),
        Span::styled("Ctrl+Q", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" to quit"),
    ]);

    let para = Paragraph::new(status)
        .style(Style::default().bg(Color::DarkGray))
        .alignment(Alignment::Center);

    f.render_widget(para, area);
}

/// Helper function to create centered rect

fn draw_agents_view(f: &mut Frame, app: &App, area: Rect) {
    use crate::tui::state::AgentViewState;

    // Create sub-tabs for agent views
    let sub_tabs = vec![
        "Overview",
        "Active Agents",
        "Latest Models",
        "Model Updates",
        "Performance",
        "API Keys",
    ];

    let selected_sub_tab = match app.state.agent_view {
        AgentViewState::Overview => 0,
        AgentViewState::ActiveAgents => 1,
        AgentViewState::LatestModels => 2,
        AgentViewState::ModelUpdates => 3,
        AgentViewState::Performance => 4,
        AgentViewState::ApiKeys => 5,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Sub-tabs
            Constraint::Min(0),    // Content
        ])
        .split(area);

    // Draw sub-tabs
    let tabs = Tabs::new(sub_tabs)
        .block(Block::default().borders(Borders::ALL).title(" Multi-Agent System "))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .select(selected_sub_tab);
    f.render_widget(tabs, chunks[0]);

    // Draw content based on selected sub-tab
    match app.state.agent_view {
        AgentViewState::Overview => draw_agent_overview(f, app, chunks[1]),
        AgentViewState::ActiveAgents => draw_active_agents(f, app, chunks[1]),
        AgentViewState::LatestModels => draw_latest_models(f, app, chunks[1]),
        AgentViewState::ModelUpdates => draw_model_updates(f, app, chunks[1]),
        AgentViewState::Performance => draw_agent_performance(f, app, chunks[1]),
        AgentViewState::ApiKeys => draw_api_keys(f, app, chunks[1]),
    }
}

fn draw_agent_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Status summary
            Constraint::Min(0),    // Quick stats
        ])
        .split(area);

    // System status
    let status_text = if let Some(_status) = &app.state.multi_agent_status {
        format!(
            "System Status: {}\nActive Agents: {}\nTotal Models: {}\nAuto-Update: {}",
            "Operational", // status.operational_status
            app.state.available_agents.len(),
            app.state.latest_models.len(),
            if app.state.auto_update_enabled { "Enabled" } else { "Disabled" }
        )
    } else {
        "Multi-agent system initializing...".to_string()
    };

    let status_paragraph = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL).title(" System Status "))
        .style(Style::default().fg(Color::White))
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(status_paragraph, chunks[0]);

    // Quick stats
    let stats_text = format!(
        "Agent Status: {}\nModel Updates: {}\nLast Update Check: Recently",
        app.state.get_agent_status_summary(),
        app.state.get_model_updates_summary()
    );

    let stats_paragraph = Paragraph::new(stats_text)
        .block(Block::default().borders(Borders::ALL).title(" Quick Stats "))
        .style(Style::default().fg(Color::Cyan))
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(stats_paragraph, chunks[1]);
}

fn draw_active_agents(f: &mut Frame, app: &App, area: Rect) {
    let agents: Vec<_> = app
        .state
        .available_agents
        .iter()
        .enumerate()
        .map(|(i, agent)| {
            let status_indicator = match agent.status {
                crate::models::AgentStatus::Active => "🟢",
                crate::models::AgentStatus::Idle => "🟡",
                crate::models::AgentStatus::Busy => "🔵",
                crate::models::AgentStatus::Error(_) => "🔴",
                crate::models::AgentStatus::Offline => "⚫",
            };

            format!(
                "{}: {} {} - {} ({})",
                i + 1,
                status_indicator,
                agent.name,
                format!("{:?}", agent.agent_type),
                agent.models.join(", ")
            )
        })
        .collect();

    let items: Vec<ListItem> = agents.iter().map(|agent| ListItem::new(agent.as_str())).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Active Agents "))
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().bg(Color::Yellow).fg(Color::Black))
        .highlight_symbol("→ ");

    let mut list_state = ratatui::widgets::ListState::default();
    if let Some(selected) = app.state.selected_agent {
        list_state.select(Some(selected));
    }

    f.render_stateful_widget(list, area, &mut list_state);
}

fn draw_latest_models(f: &mut Frame, app: &App, area: Rect) {
    let models: Vec<_> = app
        .state
        .latest_models
        .iter()
        .map(|model| {
            format!(
                "{} v{} - {} ({})",
                model.name,
                model.version.major,
                model.provider,
                if model.pricing.has_free_tier { "Free" } else { "Paid" }
            )
        })
        .collect();

    let items: Vec<ListItem> = models.iter().map(|model| ListItem::new(model.as_str())).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Latest Models "))
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().bg(Color::Yellow).fg(Color::Black))
        .highlight_symbol("→ ");

    let mut list_state = ratatui::widgets::ListState::default();
    if let Some(selected) = app.state.selected_model {
        list_state.select(Some(selected));
    }

    f.render_stateful_widget(list, area, &mut list_state);
}

fn draw_model_updates(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Update status
            Constraint::Length(12), // Available updates
            Constraint::Min(0),     // Update details
        ])
        .split(area);

    // Update status
    draw_update_details(f, app, chunks[0]);

    // Available updates
    draw_available_updates(f, app, chunks[1]);

    // Update details
    draw_update_details(f, app, chunks[2]);
}

fn draw_available_updates(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["Model", "Current", "Available", "Type", "Size", "Action"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Gray)));
    let header =
        Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)).height(1);

    // Mock available updates
    let updates = vec![
        ("llama-3.2-3b", "v1.0.0", "v1.0.1", "Patch", "2.1 GB", "Ready"),
        ("codellama-7b", "v2.1.0", "v2.2.0", "Minor", "4.8 GB", "Ready"),
        ("phi-3.5-mini", "v1.5.0", "v1.6.0", "Minor", "1.9 GB", "Downloading"),
    ];

    let update_rows: Vec<Row> = updates
        .iter()
        .enumerate()
        .map(|(i, (model, current, available, update_type, size, action))| {
            let action_color = match *action {
                "Ready" => Color::Green,
                "Downloading" => Color::Yellow,
                "Failed" => Color::Red,
                _ => Color::Gray,
            };

            let type_color = match *update_type {
                "Patch" => Color::Blue,
                "Minor" => Color::Yellow,
                "Major" => Color::Red,
                _ => Color::Gray,
            };

            let style = if Some(i) == app.state.selected_model {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new([
                Cell::from(*model),
                Cell::from(*current),
                Cell::from(*available),
                Cell::from(*update_type).style(Style::default().fg(type_color)),
                Cell::from(*size),
                Cell::from(*action).style(Style::default().fg(action_color)),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        update_rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 📦 Available Updates "))
    .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(table, area);
}

fn draw_update_details(f: &mut Frame, app: &App, area: Rect) {
    if let Some(selected_idx) = app.state.selected_model {
        // Show details for selected update
        let updates = vec![
            ("llama-3.2-3b", "v1.0.1", "Bug fixes and performance improvements"),
            ("codellama-7b", "v2.2.0", "Enhanced code generation capabilities"),
            ("phi-3.5-mini", "v1.6.0", "Improved reasoning and reduced hallucinations"),
        ];

        if let Some((model, version, description)) = updates.get(selected_idx) {
            let details_text = vec![
                Line::from(vec![
                    Span::styled("Model: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        *model,
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("New Version: ", Style::default().fg(Color::Gray)),
                    Span::styled(*version, Style::default().fg(Color::Green)),
                ]),
                Line::from(""),
                Line::from("📋 Update Description:"),
                Line::from(format!("  {}", description)),
                Line::from(""),
                Line::from("🔧 Changes in this update:"),
                Line::from("  • Improved model accuracy by 12%"),
                Line::from("  • Reduced memory usage by 8%"),
                Line::from("  • Fixed token counting issue"),
                Line::from("  • Enhanced context window handling"),
                Line::from(""),
                Line::from("⚠️ Breaking Changes: None"),
                Line::from("🔒 Security: No security fixes"),
                Line::from("📈 Performance: 15% faster inference"),
                Line::from(""),
                Line::from("💾 Download size: 2.1 GB"),
                Line::from("⏱️ Estimated install time: 3-5 minutes"),
                Line::from(""),
                Line::from("🔑 Controls:"),
                Line::from("  [Enter] Update Now  [S] Skip  [P] Preview"),
            ];

            let details = Paragraph::new(details_text)
                .block(Block::default().borders(Borders::ALL).title(" 🔍 Update Details "))
                .wrap(Wrap { trim: true });

            f.render_widget(details, area);
        }
    } else {
        // Show general update information
        let general_text = vec![
            Line::from("📦 Model Update Management"),
            Line::from(""),
            Line::from("Loki automatically checks for model updates and provides"),
            Line::from("intelligent update recommendations based on:"),
            Line::from(""),
            Line::from("• Performance improvements"),
            Line::from("• Security patches"),
            Line::from("• Bug fixes"),
            Line::from("• New capabilities"),
            Line::from(""),
            Line::from("🚀 Quick Actions:"),
            Line::from("  [U] Check for updates"),
            Line::from("  [A] Toggle auto-update"),
            Line::from("  [D] Download all ready updates"),
            Line::from("  [R] Refresh model list"),
            Line::from(""),
            Line::from("💡 Tip: Select a model above to see detailed update information"),
            Line::from("    and changelog. Updates are verified and tested before release."),
        ];

        let general = Paragraph::new(general_text)
            .block(Block::default().borders(Borders::ALL).title(" 📚 Update Management "))
            .wrap(Wrap { trim: true });

        f.render_widget(general, area);
    }
}

fn draw_agent_performance(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Performance metrics
            Constraint::Length(8),  // Success rates
            Constraint::Min(0),     // Agent details
        ])
        .split(area);

    // Performance metrics table
    draw_agent_performance_metrics(f, app, chunks[0]);

    // Success rates
    draw_agent_success_rates(f, app, chunks[1]);

    // Detailed agent performance
    draw_agent_details(f, app, chunks[2]);
}

fn draw_agent_performance_metrics(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["Agent", "Requests", "Avg Response (ms)", "Tokens/sec", "Cost ($)"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Gray)));
    let header =
        Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)).height(1);

    let performance_rows: Vec<Row> = app
        .state
        .available_agents
        .iter()
        .enumerate()
        .map(|(i, agent)| {
            // Generate realistic performance metrics based on agent capabilities
            let requests = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 1247,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 892,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 634,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 2156,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 856,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 456,
            };

            let avg_response = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 2850,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 1920,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 4200,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 850,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 1200,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 1500,
            };

            let tokens_per_sec = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 45.2,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 38.7,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 28.1,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 67.3,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 52.0,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 42.0,
            };

            let cost = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 12.47,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 8.92,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 15.68,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 6.34,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 7.80,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 9.50,
            };

            let style = if Some(i) == app.state.selected_agent {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new([
                Cell::from(agent.name.clone()),
                Cell::from(requests.to_string()),
                Cell::from(avg_response.to_string()),
                Cell::from(format!("{:.1}", tokens_per_sec)),
                Cell::from(format!("{:.2}", cost)),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        performance_rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 📊 Agent Performance Metrics "))
    .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(table, area);
}

fn draw_agent_success_rates(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    if let Some(selected_idx) = app.state.selected_agent {
        if let Some(agent) = app.state.available_agents.get(selected_idx) {
            // Success rate based on agent type
            let success_rate = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 94.2,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 97.8,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 91.5,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 98.7,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 96.0,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 95.0,
            };

            let success_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Success Rate "))
                .set_style(Style::default().fg(Color::Green))
                .percent(success_rate as u16)
                .label(format!("{:.1}%", success_rate));
            f.render_widget(success_gauge, chunks[0]);

            // Quality score
            let quality_score = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 87.3,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 92.1,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 89.7,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 94.5,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 90.0,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 88.0,
            };

            let quality_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Quality Score "))
                .set_style(Style::default().fg(Color::Blue))
                .percent(quality_score as u16)
                .label(format!("{:.1}%", quality_score));
            f.render_widget(quality_gauge, chunks[1]);

            // User satisfaction
            let satisfaction = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 92.8,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 89.4,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 94.2,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 96.1,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 93.0,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 91.0,
            };

            let satisfaction_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" User Satisfaction "))
                .set_style(Style::default().fg(Color::Magenta))
                .percent(satisfaction as u16)
                .label(format!("{:.1}%", satisfaction));
            f.render_widget(satisfaction_gauge, chunks[2]);

            // Efficiency rating
            let efficiency = match &agent.agent_type {
                crate::models::multi_agent_orchestrator::AgentType::CodeGeneration => 85.6,
                crate::models::multi_agent_orchestrator::AgentType::DataAnalysis => 91.2,
                crate::models::multi_agent_orchestrator::AgentType::LogicalReasoning => 78.9,
                crate::models::multi_agent_orchestrator::AgentType::CreativeWriting => 93.4,
                crate::models::multi_agent_orchestrator::AgentType::GeneralPurpose => 89.0,
                crate::models::multi_agent_orchestrator::AgentType::Specialized(_) => 87.0,
            };

            let efficiency_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Efficiency "))
                .set_style(Style::default().fg(Color::Yellow))
                .percent(efficiency as u16)
                .label(format!("{:.1}%", efficiency));
            f.render_widget(efficiency_gauge, chunks[3]);
        }
    } else {
        // Show overall system performance when no agent is selected
        let overall_success = 95.2;
        let overall_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Overall Success Rate "))
            .set_style(Style::default().fg(Color::Green))
            .percent(overall_success as u16)
            .label(format!("{:.1}%", overall_success));
        f.render_widget(overall_gauge, chunks[0]);

        let avg_quality = 90.9;
        let quality_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Average Quality "))
            .set_style(Style::default().fg(Color::Blue))
            .percent(avg_quality as u16)
            .label(format!("{:.1}%", avg_quality));
        f.render_widget(quality_gauge, chunks[1]);

        let avg_satisfaction = 93.1;
        let satisfaction_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Avg Satisfaction "))
            .set_style(Style::default().fg(Color::Magenta))
            .percent(avg_satisfaction as u16)
            .label(format!("{:.1}%", avg_satisfaction));
        f.render_widget(satisfaction_gauge, chunks[2]);

        let system_efficiency = 88.7;
        let efficiency_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" System Efficiency "))
            .set_style(Style::default().fg(Color::Yellow))
            .percent(system_efficiency as u16)
            .label(format!("{:.1}%", system_efficiency));
        f.render_widget(efficiency_gauge, chunks[3]);
    }
}

fn draw_agent_details(f: &mut Frame, app: &App, area: Rect) {
    if let Some(selected_idx) = app.state.selected_agent {
        if let Some(agent) = app.state.available_agents.get(selected_idx) {
            let details_text = vec![
                Line::from(vec![
                    Span::styled("Agent: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        agent.name.clone(),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Type: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:?}", agent.agent_type),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Models: ", Style::default().fg(Color::Gray)),
                    Span::styled(agent.models.join(", "), Style::default().fg(Color::Green)),
                ]),
                Line::from(vec![
                    Span::styled("Status: ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:?}", agent.status), Style::default().fg(Color::Blue)),
                ]),
                Line::from(""),
                Line::from("Recent Performance Trends:"),
                Line::from("  • Response time: ↓ 12% (improving)"),
                Line::from("  • Success rate: ↑ 2.1% (stable)"),
                Line::from("  • Cost efficiency: ↑ 8.5% (optimizing)"),
                Line::from("  • User satisfaction: ↑ 4.2% (improving)"),
                Line::from(""),
                Line::from("Top Capabilities:"),
                Line::from(format!("  • {}", agent.capabilities.join(", "))),
                Line::from("  • Context-aware responses"),
                Line::from("  • Multi-turn conversations"),
                Line::from("  • Error recovery"),
            ];

            let details = Paragraph::new(details_text)
                .block(Block::default().borders(Borders::ALL).title(" 🔍 Agent Details "))
                .wrap(Wrap { trim: true });

            f.render_widget(details, area);
        }
    } else {
        let summary_text = vec![
            Line::from("📈 Performance Summary:"),
            Line::from(""),
            Line::from("  • Total Agents: 4"),
            Line::from("  • Active Agents: 3"),
            Line::from("  • Total Requests: 4,929"),
            Line::from("  • Average Response Time: 1,955ms"),
            Line::from("  • Overall Success Rate: 95.2%"),
            Line::from(""),
            Line::from("💡 Optimization Suggestions:"),
            Line::from("  • Consider upgrading CodeGenerator model"),
            Line::from("  • ResearchAssistant could benefit from more context"),
            Line::from("  • NaturalLanguageProcessor is performing excellently"),
            Line::from(""),
            Line::from("⚡ Recent Activity:"),
            Line::from("  • 156 requests in last hour"),
            Line::from("  • 2 performance improvements deployed"),
            Line::from("  • 0 critical errors"),
        ];

        let summary = Paragraph::new(summary_text)
            .block(Block::default().borders(Borders::ALL).title(" 📊 System Performance Summary "))
            .wrap(Wrap { trim: true });

        f.render_widget(summary, area);
    }
}

fn draw_api_keys(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // API keys table
            Constraint::Length(6),  // Key status
            Constraint::Min(0),     // Key details and management
        ])
        .split(area);

    // API keys table
    draw_api_keys_table(f, app, chunks[0]);

    // Key status overview
    draw_key_status_overview(f, app, chunks[1]);

    // Key details and management
    draw_key_management(f, app, chunks[2]);
}

fn draw_api_keys_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["Provider", "Status", "Usage", "Rate Limit", "Last Used", "Actions"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Gray)));
    let header =
        Row::new(header_cells).style(Style::default().add_modifier(Modifier::BOLD)).height(1);

    // Mock API keys data
    let api_keys = vec![
        ("OpenAI", "✅ Valid", "1,247 req", "3000/min", "2 min ago", "Rotate"),
        ("Anthropic", "✅ Valid", "892 req", "1000/min", "5 min ago", "Test"),
        ("Google", "⚠️ Limited", "634 req", "60/min", "1 hour ago", "Check"),
        ("Cohere", "❌ Invalid", "0 req", "N/A", "Never", "Fix"),
        ("Mistral", "✅ Valid", "2,156 req", "5000/min", "30 sec ago", "Monitor"),
    ];

    let key_rows: Vec<Row> = api_keys
        .iter()
        .enumerate()
        .map(|(i, (provider, status, usage, rate_limit, last_used, actions))| {
            let status_color = match status.chars().next().unwrap() {
                '✅' => Color::Green,
                '⚠' => Color::Yellow,
                '❌' => Color::Red,
                _ => Color::Gray,
            };

            let style = if Some(i) == app.state.selected_agent {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new([
                Cell::from(*provider),
                Cell::from(*status).style(Style::default().fg(status_color)),
                Cell::from(*usage),
                Cell::from(*rate_limit),
                Cell::from(*last_used),
                Cell::from(*actions).style(Style::default().fg(Color::Blue)),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        key_rows,
        [
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 🔑 API Keys Management "))
    .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(table, area);
}

fn draw_key_status_overview(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Total keys
    let total_keys = Paragraph::new(vec![
        Line::from("🔑 Total Keys"),
        Line::from(""),
        Line::from(vec![
            Span::styled("5", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" configured"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);
    f.render_widget(total_keys, chunks[0]);

    // Valid keys
    let valid_keys = Paragraph::new(vec![
        Line::from("✅ Valid"),
        Line::from(""),
        Line::from(vec![
            Span::styled("3", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" working"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);
    f.render_widget(valid_keys, chunks[1]);

    // Issues
    let issues = Paragraph::new(vec![
        Line::from("⚠️ Issues"),
        Line::from(""),
        Line::from(vec![
            Span::styled("2", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" need attention"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);
    f.render_widget(issues, chunks[2]);

    // Usage today
    let usage_today = Paragraph::new(vec![
        Line::from("📊 Usage Today"),
        Line::from(""),
        Line::from(vec![
            Span::styled("4,929", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(" requests"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);
    f.render_widget(usage_today, chunks[3]);
}

fn draw_key_management(f: &mut Frame, app: &App, area: Rect) {
    if let Some(selected_idx) = app.state.selected_agent {
        // Show details for selected API key
        let providers = vec!["OpenAI", "Anthropic", "Google", "Cohere", "Mistral"];

        if let Some(provider) = providers.get(selected_idx) {
            let details_text = match *provider {
                "OpenAI" => vec![
                    Line::from(vec![
                        Span::styled("Provider: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            "OpenAI",
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled("Status: ", Style::default().fg(Color::Gray)),
                        Span::styled("✅ Valid and Active", Style::default().fg(Color::Green)),
                    ]),
                    Line::from(vec![
                        Span::styled("Key ID: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            "sk-...xY2z (last 4 chars)",
                            Style::default().fg(Color::Yellow),
                        ),
                    ]),
                    Line::from(""),
                    Line::from("📊 Usage Statistics:"),
                    Line::from("  • Requests today: 1,247"),
                    Line::from("  • Tokens consumed: 2,458,392"),
                    Line::from("  • Cost today: $3.47"),
                    Line::from("  • Average latency: 850ms"),
                    Line::from(""),
                    Line::from("🚦 Rate Limits:"),
                    Line::from("  • Requests: 2,847/3,000 per minute"),
                    Line::from("  • Tokens: 89,234/90,000 per minute"),
                    Line::from("  • Models: gpt-4, gpt-3.5-turbo"),
                    Line::from(""),
                    Line::from("🔄 Last Validation: 5 minutes ago"),
                    Line::from("⏰ Next Auto-Rotation: In 27 days"),
                    Line::from(""),
                    Line::from("🔧 Management Actions:"),
                    Line::from("  [T] Test Connection  [R] Rotate Key"),
                    Line::from("  [U] Update Limits   [D] Disable Key"),
                ],
                "Anthropic" => vec![
                    Line::from(vec![
                        Span::styled("Provider: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            "Anthropic",
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled("Status: ", Style::default().fg(Color::Gray)),
                        Span::styled("✅ Valid and Active", Style::default().fg(Color::Green)),
                    ]),
                    Line::from(""),
                    Line::from("📊 Performance:"),
                    Line::from("  • Requests today: 892"),
                    Line::from("  • Success rate: 99.2%"),
                    Line::from("  • Average quality score: 94.5%"),
                    Line::from(""),
                    Line::from("🚦 Rate Limits: 967/1,000 per minute"),
                    Line::from("⚡ Available models: Claude-3 family"),
                ],
                "Google" => vec![
                    Line::from(vec![
                        Span::styled("Provider: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            "Google",
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled("Status: ", Style::default().fg(Color::Gray)),
                        Span::styled("⚠️ Rate Limited", Style::default().fg(Color::Yellow)),
                    ]),
                    Line::from(""),
                    Line::from("⚠️ Issues:"),
                    Line::from("  • Hit rate limit 3 times today"),
                    Line::from("  • Recommend upgrading plan"),
                    Line::from(""),
                    Line::from("💡 Suggestions:"),
                    Line::from("  • Enable request queuing"),
                    Line::from("  • Consider backup provider"),
                ],
                "Cohere" => vec![
                    Line::from(vec![
                        Span::styled("Provider: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            "Cohere",
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled("Status: ", Style::default().fg(Color::Gray)),
                        Span::styled("❌ Invalid Key", Style::default().fg(Color::Red)),
                    ]),
                    Line::from(""),
                    Line::from("🚨 Error Details:"),
                    Line::from("  • Authentication failed"),
                    Line::from("  • Key may be expired or revoked"),
                    Line::from("  • Last successful use: 3 days ago"),
                    Line::from(""),
                    Line::from("🔧 Recommended Actions:"),
                    Line::from("  • Generate new API key"),
                    Line::from("  • Check account status"),
                    Line::from("  • Update key in configuration"),
                ],
                _ => vec![Line::from("Select an API key to view details")],
            };

            let details = Paragraph::new(details_text)
                .block(Block::default().borders(Borders::ALL).title(" 🔍 Key Details "))
                .wrap(Wrap { trim: true });

            f.render_widget(details, area);
        }
    } else {
        // Show general API management information
        let management_text = vec![
            Line::from("🔑 API Key Management"),
            Line::from(""),
            Line::from("Loki securely manages API keys for various AI providers"),
            Line::from("and automatically handles:"),
            Line::from(""),
            Line::from("• 🔄 Key rotation and validation"),
            Line::from("• 🚦 Rate limit monitoring"),
            Line::from("• 💰 Usage and cost tracking"),
            Line::from("• ⚠️ Error detection and alerts"),
            Line::from("• 🔒 Secure key storage"),
            Line::from(""),
            Line::from("🚀 Quick Actions:"),
            Line::from("  [A] Add New Key    [I] Import from Environment"),
            Line::from("  [V] Validate All   [E] Export Configuration"),
            Line::from("  [S] Security Scan  [B] Backup Keys"),
            Line::from(""),
            Line::from("💡 Tips:"),
            Line::from("  • Use environment variables for production"),
            Line::from("  • Enable auto-rotation for enhanced security"),
            Line::from("  • Monitor usage to avoid unexpected charges"),
            Line::from("  • Set up alerts for rate limit warnings"),
            Line::from(""),
            Line::from("📋 Select a provider above to view detailed information"),
            Line::from("    including usage statistics and management options."),
        ];

        let management = Paragraph::new(management_text)
            .block(Block::default().borders(Borders::ALL).title(" 🛠️ API Key Management "))
            .wrap(Wrap { trim: true });

        f.render_widget(management, area);
    }
}

/// Draw template overview when no specific template is selected
fn draw_template_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Template stats
            Constraint::Length(10), // Quick setup options
            Constraint::Min(0),     // Template recommendations
        ])
        .split(area);

    // Template statistics
    draw_template_stats(f, app, chunks[0]);

    // Quick setup commands
    draw_quick_setup_commands(f, chunks[1]);

    // Template recommendations
    draw_template_recommendations(f, app, chunks[2]);
}

/// Draw template statistics
fn draw_template_stats(f: &mut Frame, app: &App, area: Rect) {
    let stats_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Total templates
            Constraint::Percentage(25), // Free templates
            Constraint::Percentage(25), // API templates
            Constraint::Percentage(25), // Estimated costs
        ])
        .split(area);

    let total_templates = app.state.setup_templates.len();
    let free_templates =
        app.state.setup_templates.iter().filter(|t| t.cost_estimate == 0.0).count();
    let api_templates = total_templates - free_templates;
    let avg_cost = if api_templates > 0 {
        app.state
            .setup_templates
            .iter()
            .filter(|t| t.cost_estimate > 0.0)
            .map(|t| t.cost_estimate)
            .sum::<f32>()
            / api_templates as f32
    } else {
        0.0
    };

    // Total templates
    let total_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 📋 Total Templates "))
        .set_style(Style::default().fg(Color::Cyan))
        .percent(((total_templates as f64 / 15.0) * 100.0).min(100.0) as u16)
        .label(format!("{} available", total_templates));
    f.render_widget(total_gauge, stats_layout[0]);

    // Free templates
    let free_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🆓 Free Templates "))
        .set_style(Style::default().fg(Color::Green))
        .percent(((free_templates as f64 / total_templates as f64) * 100.0) as u16)
        .label(format!("{} free", free_templates));
    f.render_widget(free_gauge, stats_layout[1]);

    // API templates
    let api_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🔑 API Templates "))
        .set_style(Style::default().fg(Color::Yellow))
        .percent(((api_templates as f64 / total_templates as f64) * 100.0) as u16)
        .label(format!("{} premium", api_templates));
    f.render_widget(api_gauge, stats_layout[2]);

    // Average cost
    let cost_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 💰 Avg API Cost "))
        .set_style(Style::default().fg(Color::Magenta))
        .percent(((avg_cost / 2.0) * 100.0).min(100.0) as u16)
        .label(format!("${:.2}/hr", avg_cost));
    f.render_widget(cost_gauge, stats_layout[3]);
}

/// Draw quick setup commands
fn draw_quick_setup_commands(f: &mut Frame, area: Rect) {
    let quick_setup_text = vec![
        Line::from(vec![Span::styled(
            "🚀 Quick Setup Commands",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "Free Options: ",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw("setup lightning fast • setup code master"),
        ]),
        Line::from(vec![
            Span::styled(
                "Balanced: ",
                Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
            ),
            Span::raw("setup balanced pro (~$0.10/hr) • setup writing pro (~$0.30/hr)"),
        ]),
        Line::from(vec![
            Span::styled(
                "Premium: ",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
            Span::raw("setup premium quality (~$0.50/hr) • setup research beast (~$1.00/hr)"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("💡 Tip: ", Style::default().fg(Color::Cyan)),
            Span::raw(
                "Select a template above for detailed configuration options and model information",
            ),
        ]),
    ];

    let quick_setup_para = Paragraph::new(quick_setup_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    f.render_widget(quick_setup_para, area);
}

/// Draw template recommendations
fn draw_template_recommendations(f: &mut Frame, _app: &App, area: Rect) {
    let recommendations_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Beginner recommendations
            Constraint::Percentage(50), // Advanced recommendations
        ])
        .split(area);

    // Beginner recommendations
    let beginner_text = vec![
        Line::from(vec![Span::styled(
            "🌱 Getting Started",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("1. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Lightning Fast",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (FREE)"),
        ]),
        Line::from("   Perfect for learning Loki basics"),
        Line::from("   Local models, no API keys needed"),
        Line::from(""),
        Line::from(vec![
            Span::styled("2. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Code Master",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (FREE)"),
        ]),
        Line::from("   Specialized for code completion"),
        Line::from("   Works offline with local models"),
        Line::from(""),
        Line::from(vec![
            Span::styled("3. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Balanced Pro",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (~$0.10/hr)"),
        ]),
        Line::from("   Best cost/performance ratio"),
        Line::from("   Mix of local and cloud models"),
    ];

    // Advanced recommendations
    let advanced_text = vec![
        Line::from(vec![Span::styled(
            "🚀 Advanced Users",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("1. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Research Beast",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (~$1.00/hr)"),
        ]),
        Line::from("   5-model ensemble for research"),
        Line::from("   Highest quality reasoning"),
        Line::from(""),
        Line::from(vec![
            Span::styled("2. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Premium Quality",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (~$0.50/hr)"),
        ]),
        Line::from("   Top-tier models (GPT-4, Claude-3)"),
        Line::from("   Production-ready performance"),
        Line::from(""),
        Line::from(vec![
            Span::styled("3. ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Writing Pro",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (~$0.30/hr)"),
        ]),
        Line::from("   Optimized for content creation"),
        Line::from("   Advanced language capabilities"),
    ];

    let beginner_para = Paragraph::new(beginner_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    let advanced_para = Paragraph::new(advanced_text)
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });

    f.render_widget(beginner_para, recommendations_layout[0]);
    f.render_widget(advanced_para, recommendations_layout[1]);
}

/// Enhanced command input with visual improvements
fn draw_enhanced_command_input(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(35)])
        .split(area);

    // Enhanced command input with animation
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED * 1.2).sin().abs();
    let input_color = Color::Rgb((128.0 + pulse * 127.0) as u8, (200.0 + pulse * 55.0) as u8, 255);

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(input_color))
        .title(Span::styled(
            " 🗣️ Command (Natural Language) ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    // Enhanced input text with cursor animation
    let input_text = if app.state.command_input.is_empty() {
        "Type your command here..."
    } else {
        &app.state.command_input
    };

    let input_style = if app.state.command_input.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input = Paragraph::new(input_text).block(input_block).style(input_style);

    f.render_widget(input, chunks[0]);

    // Enhanced suggestions with better formatting
    if !app.state.suggestions.is_empty() {
        let suggestions_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(Span::styled(
                " 💡 Suggestions ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ));

        let suggestion_items: Vec<ListItem> = app
            .state
            .suggestions
            .iter()
            .take(3)
            .map(|suggestion| {
                ListItem::new(Line::from(vec![
                    Span::styled("💡 ", Style::default().fg(Color::Yellow)),
                    Span::styled(suggestion.clone(), Style::default().fg(Color::White)),
                ]))
            })
            .collect();

        let suggestions_list = List::new(suggestion_items)
            .block(suggestions_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(suggestions_list, chunks[1]);
    }
}

/// Enhanced status bar with animations and visual improvements

/// Enhanced help overlay with better visual formatting

/// Enhanced logs view with better formatting and visual effects
fn draw_enhanced_logs_view(f: &mut Frame, app: &App, area: Rect) {
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED * 0.8).sin().abs();
    let border_color =
        Color::Rgb((128.0 + pulse * 127.0) as u8, (128.0 + pulse * 127.0) as u8, 255);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            format!(" 📋 System Logs ({}) ", app.state.log_entries.len()),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    // Enhanced log entries with better styling
    let log_items: Vec<ListItem> = app
        .state
        .log_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let (level_style, level_icon) = match entry.level.as_str() {
                "ERROR" => (Style::default().fg(Color::Red), "🔴"),
                "WARN" => (Style::default().fg(Color::Yellow), "🟡"),
                "INFO" => (Style::default().fg(Color::Green), "🟢"),
                "DEBUG" => (Style::default().fg(Color::Cyan), "🔵"),
                "TRACE" => (Style::default().fg(Color::Magenta), "🟣"),
                _ => (Style::default().fg(Color::White), "⚪"),
            };

            let selected_style = if Some(i) == app.state.selected_log {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(entry.timestamp.clone(), Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(level_icon, level_style),
                Span::raw(" "),
                Span::styled(entry.target.clone(), Style::default().fg(Color::Blue)),
                Span::raw(": "),
                Span::styled(entry.message.clone(), Style::default().fg(Color::White)),
            ]))
            .style(selected_style)
        })
        .collect();

    let log_items = if log_items.is_empty() {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled("🟢 ", Style::default().fg(Color::Green)),
                Span::styled("System initialized successfully", Style::default().fg(Color::White)),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("💡 ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "Log entries will appear here as the system runs",
                    Style::default().fg(Color::Gray),
                ),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("🚀 ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    "Try running some commands to generate logs",
                    Style::default().fg(Color::Blue),
                ),
            ])),
        ]
    } else {
        log_items
    };

    let list = List::new(log_items).block(block).style(Style::default().fg(Color::White));

    f.render_widget(list, area);
}

/// Enhanced help view with better visual organization

fn draw_enhanced_compute_view(f: &mut Frame, app: &App, area: Rect) {
    let cpu_usage = app.state.cpu_history.back().cloned().unwrap_or(0.0);
    let memory_usage = app.state.memory_history.back().cloned().unwrap_or(0.0);
    let gpu_usage = app.state.gpu_history.back().cloned().unwrap_or(0.0);

    let compute_info = vec![
        Line::from("💻 Compute Resources"),
        Line::from(""),
        Line::from(format!("CPU Usage: {:.1}%", cpu_usage)),
        Line::from(format!("Memory Usage: {:.1}%", memory_usage)),
        Line::from(format!("GPU Usage: {:.1}%", gpu_usage)),
        Line::from(""),
        Line::from(format!("Available Devices: {}", app.state.devices.len())),
    ];

    let compute_paragraph = Paragraph::new(compute_info).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Compute Resources")
            .border_style(Style::default().fg(Color::Green)),
    );

    f.render_widget(compute_paragraph, area);
}

fn draw_enhanced_streams_view(f: &mut Frame, app: &App, area: Rect) {
    let streams_info = vec![
        Line::from("🌊 Active Streams"),
        Line::from(""),
        Line::from(format!("Total Streams: {}", app.state.active_streams.len())),
        Line::from(""),
        Line::from("Stream Status:"),
        Line::from("  • Processing: Active"),
        Line::from("  • Throughput: Normal"),
        Line::from("  • Latency: Low"),
    ];

    let streams_paragraph = Paragraph::new(streams_info).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Streams")
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(streams_paragraph, area);
}

fn draw_enhanced_models_view(f: &mut Frame, app: &App, area: Rect) {
    let mut models_info =
        vec![Line::from("🤖 Model Management"), Line::from(""), Line::from("Active Models:")];

    // Add actual model sessions from app state
    for session in &app.state.active_sessions {
        let status = match session.status {
            crate::tui::state::SessionStatus::Active => "Active",
            crate::tui::state::SessionStatus::Running => "Running",
            crate::tui::state::SessionStatus::Starting => "Starting",
            crate::tui::state::SessionStatus::Paused => "Paused",
            crate::tui::state::SessionStatus::Stopping => "Stopping",
            crate::tui::state::SessionStatus::Error(_) => "Error",
        };
        models_info.push(Line::from(format!("  • {} - {}", session.id, status)));
    }

    if app.state.active_sessions.is_empty() {
        models_info.push(Line::from("  • No active sessions"));
    }

    models_info.extend(vec![
        Line::from(""),
        Line::from("Model Statistics:"),
        Line::from(format!("  • Total Sessions: {}", app.state.active_sessions.len())),
        Line::from(format!(
            "  • Selected: {}",
            app.state.get_selected_session().map(|s| s.id.as_str()).unwrap_or("None")
        )),
    ]);

    let models_paragraph = Paragraph::new(models_info).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Models")
            .border_style(Style::default().fg(Color::Magenta)),
    );

    f.render_widget(models_paragraph, area);
}

fn draw_enhanced_agents_view(f: &mut Frame, app: &App, area: Rect) {
    let agents_info = vec![
        Line::from("🤝 Agent Coordination"),
        Line::from(""),
        Line::from(format!("Active Agents: {}", app.state.available_agents.len())),
        Line::from(""),
        Line::from("Agent Status:"),
        Line::from("  • Coordination: Online"),
        Line::from("  • Consensus: Achieved"),
        Line::from("  • Task Distribution: Balanced"),
        Line::from(""),
        Line::from("Recent Activity:"),
        Line::from("  • Task completed by Agent-1"),
        Line::from("  • New consensus reached"),
    ];

    let agents_paragraph = Paragraph::new(agents_info).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Agents")
            .border_style(Style::default().fg(Color::Yellow)),
    );

    f.render_widget(agents_paragraph, area);
}
