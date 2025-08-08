use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Styled};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    BarChart,
    Block,
    Borders,
    Cell,
    Gauge,
    Paragraph,
    Row,
    Sparkline,
    Table,
    Tabs,
    Wrap,
};

use super::app::App;
use super::state::AgentSpecializationViewState;

/// Draw the agent specialization and routing view
pub fn draw_agent_specialization_view(f: &mut Frame, app: &App, area: Rect) {
    let sub_tabs = vec![
        "Overview",
        "Specializations",
        "Routing",
        "Performance",
        "Load Balance",
        "Opportunities",
    ];

    let selected_sub_tab = match app.state.agent_specialization_view {
        AgentSpecializationViewState::Overview => 0,
        AgentSpecializationViewState::Specializations => 1,
        AgentSpecializationViewState::Routing => 2,
        AgentSpecializationViewState::Performance => 3,
        AgentSpecializationViewState::LoadBalancing => 4,
        AgentSpecializationViewState::Opportunities => 5,
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
        .block(Block::default().borders(Borders::ALL).title(" 🎯 Agent Specialization & Routing "))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .select(selected_sub_tab);
    f.render_widget(tabs, chunks[0]);

    // Draw content based on selected sub-tab
    match app.state.agent_specialization_view {
        AgentSpecializationViewState::Overview => draw_specialization_overview(f, app, chunks[1]),
        AgentSpecializationViewState::Specializations => {
            draw_agent_specializations(f, app, chunks[1])
        }
        AgentSpecializationViewState::Routing => draw_routing_system(f, app, chunks[1]),
        AgentSpecializationViewState::Performance => draw_routing_performance(f, app, chunks[1]),
        AgentSpecializationViewState::LoadBalancing => draw_load_balancing(f, app, chunks[1]),
        AgentSpecializationViewState::Opportunities => {
            draw_specialization_opportunities(f, app, chunks[1])
        }
    }
}

/// Draw specialization system overview
fn draw_specialization_overview(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // System metrics
            Constraint::Length(12), // Agent specialization chart
            Constraint::Min(0),     // Quick insights and actions
        ])
        .split(area);

    // System metrics
    draw_system_metrics(f, chunks[0]);

    // Agent specialization distribution
    draw_specialization_distribution(f, chunks[1]);

    // System insights and actions
    draw_system_insights(f, chunks[2]);
}

/// Draw system metrics overview
fn draw_system_metrics(f: &mut Frame, area: Rect) {
    let metrics_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // Total agents
            Constraint::Percentage(20), // Specialized agents
            Constraint::Percentage(20), // Routing accuracy
            Constraint::Percentage(20), // System efficiency
            Constraint::Percentage(20), // Load balance
        ])
        .split(area);

    // Mock data - in real implementation, get from specialization system
    let total_agents = "24";
    let specialized_agents = "18";
    let routing_accuracy = 94;
    let system_efficiency = "87%";
    let load_balance = 78;

    // Total agents card
    let agents_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            total_agents,
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Total Agents"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🤖 Agents "))
    .alignment(Alignment::Center);
    f.render_widget(agents_card, metrics_layout[0]);

    // Specialized agents card
    let specialized_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            specialized_agents,
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Specialized"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🎯 Specialized "))
    .alignment(Alignment::Center);
    f.render_widget(specialized_card, metrics_layout[1]);

    // Routing accuracy gauge
    let accuracy_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🎯 Accuracy "))
        .set_style(Style::default().fg(if routing_accuracy > 90 {
            Color::Green
        } else if routing_accuracy > 80 {
            Color::Yellow
        } else {
            Color::Red
        }))
        .percent(routing_accuracy)
        .label(format!("{}%", routing_accuracy));
    f.render_widget(accuracy_gauge, metrics_layout[2]);

    // System efficiency card
    let efficiency_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            system_efficiency,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Efficiency"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" ⚡ Efficiency "))
    .alignment(Alignment::Center);
    f.render_widget(efficiency_card, metrics_layout[3]);

    // Load balance gauge
    let balance_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚖️ Balance "))
        .set_style(Style::default().fg(if load_balance > 80 {
            Color::Green
        } else if load_balance > 60 {
            Color::Yellow
        } else {
            Color::Red
        }))
        .percent(load_balance)
        .label(format!("{}%", load_balance));
    f.render_widget(balance_gauge, metrics_layout[4]);
}

/// Draw specialization distribution chart
fn draw_specialization_distribution(f: &mut Frame, area: Rect) {
    // Mock data for specialization distribution
    let specialization_data = vec![
        ("Code", 8),
        ("Data", 6),
        ("NLP", 5),
        ("Writing", 4),
        ("Docs", 3),
        ("Math", 2),
        ("Research", 4),
        ("Education", 3),
    ];

    let chart = BarChart::default()
        .block(
            Block::default().borders(Borders::ALL).title(" 📊 Agent Specialization Distribution "),
        )
        .data(&specialization_data)
        .bar_width(4)
        .bar_style(Style::default().fg(Color::Magenta))
        .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));

    f.render_widget(chart, area);
}

/// Draw system insights and recommended actions
fn draw_system_insights(f: &mut Frame, area: Rect) {
    let insights_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // System insights
            Constraint::Percentage(50), // Recommended actions
        ])
        .split(area);

    // System insights
    let insights_text = vec![
        Line::from("🔍 System Insights"),
        Line::from(""),
        Line::from("Specialization Status:"),
        Line::from("• 18/24 agents have developed specializations"),
        Line::from("• Code generation is most popular (8 agents)"),
        Line::from("• Data analysis shows highest performance gains"),
        Line::from("• 3 agents are cross-specialized"),
        Line::from(""),
        Line::from("Routing Intelligence:"),
        Line::from("• 94% routing accuracy (↑2% this week)"),
        Line::from("• Average routing time: 12ms"),
        Line::from("• Load variance: 0.23 (target: <0.3)"),
        Line::from("• Predictive routing saves 15% response time"),
        Line::from(""),
        Line::from("Performance Trends:"),
        Line::from("• Overall quality score: 0.89 (↑0.03)"),
        Line::from("• Specialized agents: 23% faster"),
        Line::from("• Cost efficiency: +18% vs random routing"),
        Line::from("• User satisfaction: 91% (↑5%)"),
    ];

    let insights_paragraph = Paragraph::new(insights_text)
        .block(Block::default().borders(Borders::ALL).title(" 💡 Intelligence "))
        .wrap(Wrap { trim: true });
    f.render_widget(insights_paragraph, insights_layout[0]);

    // Recommended actions
    let actions_text = vec![
        Line::from("🚀 Recommended Actions"),
        Line::from(""),
        Line::from("High Priority:"),
        Line::from(""),
        Line::from("🎯 [Auto] Develop Math specialization"),
        Line::from("   Only 2 agents, high demand detected"),
        Line::from(""),
        Line::from("⚖️ [Manual] Rebalance Code agents"),
        Line::from("   Load variance: 0.35 (above target)"),
        Line::from(""),
        Line::from("🔄 [Auto] Enable cross-training"),
        Line::from("   Agent-15 shows NLP+Writing potential"),
        Line::from(""),
        Line::from("Medium Priority:"),
        Line::from(""),
        Line::from("📊 Optimize routing for cost efficiency"),
        Line::from("🔧 Tune specialization thresholds"),
        Line::from("📈 Implement advanced learning paths"),
        Line::from(""),
        Line::from("Quick Actions:"),
        Line::from("[O] Apply All Optimizations"),
        Line::from("[S] Start Specialization Training"),
        Line::from("[R] Rebalance Loads"),
        Line::from("[A] Analyze Performance"),
    ];

    let actions_paragraph = Paragraph::new(actions_text)
        .block(Block::default().borders(Borders::ALL).title(" 🎯 Actions "))
        .wrap(Wrap { trim: true });
    f.render_widget(actions_paragraph, insights_layout[1]);
}

/// Draw agent specializations table
fn draw_agent_specializations(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(70), // Specializations table
            Constraint::Percentage(30), // Specialization details
        ])
        .split(area);

    // Agent specializations table
    draw_specializations_table(f, app, chunks[0]);

    // Specialization management
    draw_specialization_management(f, app, chunks[1]);
}

/// Draw specializations table
fn draw_specializations_table(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Agent"),
        Cell::from("Primary"),
        Cell::from("Secondary"),
        Cell::from("Proficiency"),
        Cell::from("Performance"),
        Cell::from("Trend"),
        Cell::from("Tasks"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("Agent-01"),
            Cell::from("Code Generation"),
            Cell::from("Technical Docs"),
            Cell::from("🟢 Expert"),
            Cell::from("⭐⭐⭐⭐⭐"),
            Cell::from("📈 ↑"),
            Cell::from("147"),
        ]),
        Row::new(vec![
            Cell::from("Agent-02"),
            Cell::from("Data Analysis"),
            Cell::from("-"),
            Cell::from("🟡 Advanced"),
            Cell::from("⭐⭐⭐⭐"),
            Cell::from("📈 ↑"),
            Cell::from("89"),
        ]),
        Row::new(vec![
            Cell::from("Agent-03"),
            Cell::from("NLP"),
            Cell::from("Translation"),
            Cell::from("🟢 Expert"),
            Cell::from("⭐⭐⭐⭐⭐"),
            Cell::from("📊 →"),
            Cell::from("203"),
        ]),
        Row::new(vec![
            Cell::from("Agent-04"),
            Cell::from("Creative Writing"),
            Cell::from("Education"),
            Cell::from("🟡 Advanced"),
            Cell::from("⭐⭐⭐⭐"),
            Cell::from("📈 ↑"),
            Cell::from("156"),
        ]),
        Row::new(vec![
            Cell::from("Agent-05"),
            Cell::from("Mathematics"),
            Cell::from("Problem Solving"),
            Cell::from("🔴 Intermediate"),
            Cell::from("⭐⭐⭐"),
            Cell::from("📉 ↓"),
            Cell::from("67"),
        ]),
        Row::new(vec![
            Cell::from("Agent-06"),
            Cell::from("Research"),
            Cell::from("Data Analysis"),
            Cell::from("🟢 Expert"),
            Cell::from("⭐⭐⭐⭐⭐"),
            Cell::from("📈 ↑"),
            Cell::from("124"),
        ]),
        Row::new(vec![
            Cell::from("Agent-07"),
            Cell::from("Code Generation"),
            Cell::from("-"),
            Cell::from("🟡 Advanced"),
            Cell::from("⭐⭐⭐⭐"),
            Cell::from("📊 →"),
            Cell::from("98"),
        ]),
        Row::new(vec![
            Cell::from("Agent-08"),
            Cell::from("Technical Docs"),
            Cell::from("Education"),
            Cell::from("🟡 Advanced"),
            Cell::from("⭐⭐⭐⭐"),
            Cell::from("📈 ↑"),
            Cell::from("112"),
        ]),
    ];

    let specializations_table = Table::new(
        rows,
        [
            Constraint::Percentage(12),
            Constraint::Percentage(20),
            Constraint::Percentage(18),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
            Constraint::Percentage(10),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 🎯 Agent Specializations "));

    f.render_widget(specializations_table, area);
}

/// Draw specialization management controls
fn draw_specialization_management(f: &mut Frame, _app: &App, area: Rect) {
    let management_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Training controls
            Constraint::Percentage(50), // Performance tuning
        ])
        .split(area);

    // Training controls
    let training_text = vec![
        Line::from("🎓 Specialization Training"),
        Line::from(""),
        Line::from("Selected: Agent-05 (Mathematics)"),
        Line::from(""),
        Line::from("Training Options:"),
        Line::from("[1] 🎯 Intensive Training (2 weeks)"),
        Line::from("    Target: Expert level"),
        Line::from("    Required tasks: 50"),
        Line::from(""),
        Line::from("[2] 🌱 Gradual Development (6 weeks)"),
        Line::from("    Target: Advanced+ level"),
        Line::from("    Required tasks: 30"),
        Line::from(""),
        Line::from("[3] 🤝 Cross-Training"),
        Line::from("    Add secondary: Problem Solving"),
        Line::from("    Synergy bonus: +15%"),
        Line::from(""),
        Line::from("Progress: ████░░░░░░ 40%"),
        Line::from("ETA: 3.2 weeks"),
    ];

    let training_paragraph = Paragraph::new(training_text)
        .block(Block::default().borders(Borders::ALL).title(" 🎓 Training "))
        .wrap(Wrap { trim: true });
    f.render_widget(training_paragraph, management_layout[0]);

    // Performance tuning
    let tuning_text = vec![
        Line::from("⚙️ Performance Tuning"),
        Line::from(""),
        Line::from("Current Metrics:"),
        Line::from("Success Rate: 89% (target: 95%)"),
        Line::from("Avg Quality: 0.84 (target: 0.90)"),
        Line::from("Response Time: 1.8s (target: 1.5s)"),
        Line::from(""),
        Line::from("Optimization Options:"),
        Line::from(""),
        Line::from("🔧 Adjust routing weights"),
        Line::from("   Current: 0.75 → Suggested: 0.85"),
        Line::from(""),
        Line::from("🎯 Refine task matching"),
        Line::from("   Enable fuzzy matching"),
        Line::from(""),
        Line::from("📊 Update learning rate"),
        Line::from("   Current: 0.1 → Suggested: 0.15"),
        Line::from(""),
        Line::from("[T] Apply Tuning  [R] Reset  [A] Auto-tune"),
    ];

    let tuning_paragraph = Paragraph::new(tuning_text)
        .block(Block::default().borders(Borders::ALL).title(" ⚙️ Tuning "))
        .wrap(Wrap { trim: true });
    f.render_widget(tuning_paragraph, management_layout[1]);
}

/// Draw routing system interface
fn draw_routing_system(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),     // Routing overview
            Constraint::Percentage(60), // Routing decisions table
            Constraint::Percentage(40), // Routing strategies
        ])
        .split(area);

    // Routing overview
    draw_routing_overview(f, chunks[0]);

    // Recent routing decisions
    draw_routing_decisions(f, app, chunks[1]);

    // Routing strategies
    draw_routing_strategies(f, app, chunks[2]);
}

/// Draw routing overview metrics
fn draw_routing_overview(f: &mut Frame, area: Rect) {
    let overview_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Routing accuracy
            Constraint::Percentage(25), // Avg routing time
            Constraint::Percentage(25), // Active strategies
            Constraint::Percentage(25), // Queue status
        ])
        .split(area);

    // Routing accuracy card
    let accuracy_card = Paragraph::new(vec![
        Line::from("Last 1000 routes"),
        Line::from(""),
        Line::from(Span::styled(
            "94.2%",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("✅ 942 correct"),
        Line::from("❌ 58 suboptimal"),
        Line::from(""),
        Line::from("Target: 95%"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🎯 Accuracy "))
    .alignment(Alignment::Center);
    f.render_widget(accuracy_card, overview_layout[0]);

    // Average routing time card
    let timing_card = Paragraph::new(vec![
        Line::from("Decision Speed"),
        Line::from(""),
        Line::from(Span::styled(
            "12ms",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("📊 P50: 8ms"),
        Line::from("📊 P95: 28ms"),
        Line::from("📊 P99: 45ms"),
        Line::from(""),
        Line::from("Target: <15ms"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" ⚡ Speed "))
    .alignment(Alignment::Center);
    f.render_widget(timing_card, overview_layout[1]);

    // Active strategies card
    let strategies_card = Paragraph::new(vec![
        Line::from("Strategy Mix"),
        Line::from(""),
        Line::from(Span::styled(
            "5",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("🎯 Specialization: 40%"),
        Line::from("⚖️ Load Balance: 25%"),
        Line::from("⚡ Performance: 20%"),
        Line::from("💰 Cost: 15%"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🔄 Strategies "))
    .alignment(Alignment::Center);
    f.render_widget(strategies_card, overview_layout[2]);

    // Queue status card
    let queue_card = Paragraph::new(vec![
        Line::from("Current Queue"),
        Line::from(""),
        Line::from(Span::styled(
            "23",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("⏳ Pending routes"),
        Line::from("🔄 Avg wait: 1.2s"),
        Line::from("📈 Peak: 67 (2PM)"),
        Line::from("📉 Low: 2 (3AM)"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 📋 Queue "))
    .alignment(Alignment::Center);
    f.render_widget(queue_card, overview_layout[3]);
}

/// Draw recent routing decisions
fn draw_routing_decisions(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Time"),
        Cell::from("Task Type"),
        Cell::from("Selected Agent"),
        Cell::from("Reason"),
        Cell::from("Confidence"),
        Cell::from("Actual Performance"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("14:23:45"),
            Cell::from("Code Review"),
            Cell::from("Agent-01"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("96%"),
            Cell::from("✅ Excellent"),
        ]),
        Row::new(vec![
            Cell::from("14:23:42"),
            Cell::from("Data Analysis"),
            Cell::from("Agent-02"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("91%"),
            Cell::from("✅ Good"),
        ]),
        Row::new(vec![
            Cell::from("14:23:38"),
            Cell::from("Translation"),
            Cell::from("Agent-03"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("89%"),
            Cell::from("✅ Excellent"),
        ]),
        Row::new(vec![
            Cell::from("14:23:35"),
            Cell::from("Code Generation"),
            Cell::from("Agent-07"),
            Cell::from("⚖️ Load Balancing"),
            Cell::from("78%"),
            Cell::from("⚠️ Acceptable"),
        ]),
        Row::new(vec![
            Cell::from("14:23:31"),
            Cell::from("Math Problem"),
            Cell::from("Agent-05"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("67%"),
            Cell::from("❌ Poor"),
        ]),
        Row::new(vec![
            Cell::from("14:23:28"),
            Cell::from("Research"),
            Cell::from("Agent-06"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("94%"),
            Cell::from("✅ Excellent"),
        ]),
        Row::new(vec![
            Cell::from("14:23:24"),
            Cell::from("Documentation"),
            Cell::from("Agent-08"),
            Cell::from("🎯 Specialization Match"),
            Cell::from("86%"),
            Cell::from("✅ Good"),
        ]),
    ];

    let decisions_table = Table::new(
        rows,
        [
            Constraint::Percentage(12),
            Constraint::Percentage(18),
            Constraint::Percentage(15),
            Constraint::Percentage(25),
            Constraint::Percentage(12),
            Constraint::Percentage(18),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 🎯 Recent Routing Decisions "));

    f.render_widget(decisions_table, area);
}

/// Draw routing strategies configuration
fn draw_routing_strategies(f: &mut Frame, _app: &App, area: Rect) {
    let strategies_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Strategy weights
            Constraint::Percentage(50), // Strategy performance
        ])
        .split(area);

    // Strategy weights
    let weights_text = vec![
        Line::from("⚖️ Strategy Weights"),
        Line::from(""),
        Line::from("🎯 Specialization-Based: 40%"),
        Line::from("   [████████░░] Enabled"),
        Line::from("   Effectiveness: 94%"),
        Line::from(""),
        Line::from("⚖️ Load-Balanced: 25%"),
        Line::from("   [██████░░░░] Enabled"),
        Line::from("   Effectiveness: 78%"),
        Line::from(""),
        Line::from("⚡ Performance-Based: 20%"),
        Line::from("   [████████░░] Enabled"),
        Line::from("   Effectiveness: 87%"),
        Line::from(""),
        Line::from("💰 Cost-Optimized: 15%"),
        Line::from("   [██████░░░░] Enabled"),
        Line::from("   Effectiveness: 71%"),
        Line::from(""),
        Line::from("[A] Auto-adjust  [M] Manual  [R] Reset"),
    ];

    let weights_paragraph = Paragraph::new(weights_text)
        .block(Block::default().borders(Borders::ALL).title(" ⚖️ Weights "))
        .wrap(Wrap { trim: true });
    f.render_widget(weights_paragraph, strategies_layout[0]);

    // Strategy performance
    let performance_text = vec![
        Line::from("📊 Strategy Performance"),
        Line::from(""),
        Line::from("Last 24 Hours:"),
        Line::from(""),
        Line::from("🎯 Specialization-Based:"),
        Line::from("   Routes: 456 | Success: 97%"),
        Line::from("   Avg Quality: 0.91"),
        Line::from(""),
        Line::from("⚖️ Load-Balanced:"),
        Line::from("   Routes: 289 | Success: 89%"),
        Line::from("   Load Variance: 0.21"),
        Line::from(""),
        Line::from("⚡ Performance-Based:"),
        Line::from("   Routes: 234 | Success: 92%"),
        Line::from("   Avg Time: 1.4s"),
        Line::from(""),
        Line::from("💰 Cost-Optimized:"),
        Line::from("   Routes: 167 | Success: 85%"),
        Line::from("   Cost Savings: 18%"),
        Line::from(""),
        Line::from("Overall: 94.2% success rate"),
    ];

    let performance_paragraph = Paragraph::new(performance_text)
        .block(Block::default().borders(Borders::ALL).title(" 📊 Performance "))
        .wrap(Wrap { trim: true });
    f.render_widget(performance_paragraph, strategies_layout[1]);
}

/// Draw routing performance metrics
fn draw_routing_performance(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),      // Performance overview
            Constraint::Percentage(60), // Performance charts
            Constraint::Percentage(40), // Performance analysis
        ])
        .split(area);

    // Performance overview
    draw_performance_overview(f, chunks[0]);

    // Performance charts
    draw_performance_charts(f, app, chunks[1]);

    // Performance analysis
    draw_performance_analysis(f, app, chunks[2]);
}

/// Draw performance overview metrics
fn draw_performance_overview(f: &mut Frame, area: Rect) {
    let overview_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // Success rate
            Constraint::Percentage(20), // Response time
            Constraint::Percentage(20), // Quality score
            Constraint::Percentage(20), // Throughput
            Constraint::Percentage(20), // Efficiency
        ])
        .split(area);

    // Success rate card
    let success_rate = 94;
    let success_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ✅ Success "))
        .set_style(Style::default().fg(Color::Green))
        .percent(success_rate)
        .label(format!("{}%", success_rate));
    f.render_widget(success_gauge, overview_layout[0]);

    // Response time card
    let response_time_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "1.4s",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Avg Response"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" ⚡ Speed "))
    .alignment(Alignment::Center);
    f.render_widget(response_time_card, overview_layout[1]);

    // Quality score card
    let quality_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "0.89",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Quality Score"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🎯 Quality "))
    .alignment(Alignment::Center);
    f.render_widget(quality_card, overview_layout[2]);

    // Throughput card
    let throughput_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "847",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Tasks/Hour"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🚀 Throughput "))
    .alignment(Alignment::Center);
    f.render_widget(throughput_card, overview_layout[3]);

    // Efficiency card
    let efficiency_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "87%",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Efficiency"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" ⚙️ Efficiency "))
    .alignment(Alignment::Center);
    f.render_widget(efficiency_card, overview_layout[4]);
}

/// Draw performance charts
fn draw_performance_charts(f: &mut Frame, _app: &App, area: Rect) {
    let charts_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Success rate trend
            Constraint::Percentage(50), // Response time trend
        ])
        .split(area);

    // Success rate trend data
    let success_data =
        vec![89, 91, 88, 92, 90, 94, 93, 95, 92, 94, 96, 93, 95, 94, 97, 95, 94, 96, 95, 94];

    // Convert to sparkline data
    let success_sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" 📈 Success Rate Trend (20 days) "))
        .data(&success_data)
        .style(Style::default().fg(Color::Green));

    f.render_widget(success_sparkline, charts_layout[0]);

    // Response time trend data (in deciseconds for display)
    let response_data =
        vec![18, 16, 19, 15, 17, 14, 15, 13, 16, 14, 13, 15, 14, 14, 12, 13, 14, 13, 13, 14];

    let response_sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚡ Response Time Trend (20 days) "))
        .data(&response_data)
        .style(Style::default().fg(Color::Blue));

    f.render_widget(response_sparkline, charts_layout[1]);
}

/// Draw performance analysis
fn draw_performance_analysis(f: &mut Frame, _app: &App, area: Rect) {
    let analysis_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Performance insights
            Constraint::Percentage(50), // Optimization recommendations
        ])
        .split(area);

    // Performance insights
    let insights_text = vec![
        Line::from("🔍 Performance Insights"),
        Line::from(""),
        Line::from("Best Performing Agents:"),
        Line::from("• Agent-01: 97% success, 1.1s avg"),
        Line::from("• Agent-03: 96% success, 1.2s avg"),
        Line::from("• Agent-06: 95% success, 1.3s avg"),
        Line::from(""),
        Line::from("Performance Patterns:"),
        Line::from("• Specialized agents: +23% performance"),
        Line::from("• Peak hours: 14:00-16:00 (slower)"),
        Line::from("• Cross-specialized: +15% quality"),
        Line::from("• Math tasks: 12% below target"),
        Line::from(""),
        Line::from("Quality Distribution:"),
        Line::from("• >0.9: 67% of tasks"),
        Line::from("• 0.8-0.9: 28% of tasks"),
        Line::from("• <0.8: 5% of tasks"),
        Line::from(""),
        Line::from("Bottlenecks:"),
        Line::from("• Agent-05 mathematics specialization"),
        Line::from("• Load imbalance during peak hours"),
    ];

    let insights_paragraph = Paragraph::new(insights_text)
        .block(Block::default().borders(Borders::ALL).title(" 🔍 Insights "))
        .wrap(Wrap { trim: true });
    f.render_widget(insights_paragraph, analysis_layout[0]);

    // Optimization recommendations
    let recommendations_text = vec![
        Line::from("💡 Optimization Recommendations"),
        Line::from(""),
        Line::from("Immediate Actions:"),
        Line::from(""),
        Line::from("🎯 Intensive training for Agent-05"),
        Line::from("   Target: Math specialization"),
        Line::from("   Expected: +15% success rate"),
        Line::from(""),
        Line::from("⚖️ Implement peak-hour load balancing"),
        Line::from("   Use secondary specializations"),
        Line::from("   Expected: -0.3s response time"),
        Line::from(""),
        Line::from("🔄 Enable predictive pre-routing"),
        Line::from("   Queue tasks during low load"),
        Line::from("   Expected: +8% throughput"),
        Line::from(""),
        Line::from("Strategic Improvements:"),
        Line::from(""),
        Line::from("📊 Develop quality prediction models"),
        Line::from("🤝 Cross-train high performers"),
        Line::from("🔧 Optimize routing algorithms"),
        Line::from("📈 Implement A/B testing"),
        Line::from(""),
        Line::from("[I] Implement All  [T] Test Changes"),
    ];

    let recommendations_paragraph = Paragraph::new(recommendations_text)
        .block(Block::default().borders(Borders::ALL).title(" 💡 Recommendations "))
        .wrap(Wrap { trim: true });
    f.render_widget(recommendations_paragraph, analysis_layout[1]);
}

/// Draw load balancing interface
fn draw_load_balancing(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),      // Load overview
            Constraint::Percentage(60), // Agent loads
            Constraint::Percentage(40), // Balancing controls
        ])
        .split(area);

    // Load overview
    draw_load_overview(f, chunks[0]);

    // Agent loads table
    draw_agent_loads(f, app, chunks[1]);

    // Balancing controls
    draw_balancing_controls(f, app, chunks[2]);
}

/// Draw load balancing overview
fn draw_load_overview(f: &mut Frame, area: Rect) {
    let overview_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Load variance
            Constraint::Percentage(25), // Most loaded
            Constraint::Percentage(25), // Least loaded
            Constraint::Percentage(25), // Balance score
        ])
        .split(area);

    // Load variance card
    let variance_card = Paragraph::new(vec![
        Line::from("Load Distribution"),
        Line::from(""),
        Line::from(Span::styled(
            "0.23",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Variance"),
        Line::from("Target: <0.30"),
        Line::from("Status: ✅ Good"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 📊 Variance "))
    .alignment(Alignment::Center);
    f.render_widget(variance_card, overview_layout[0]);

    // Most loaded agent card
    let most_loaded_card = Paragraph::new(vec![
        Line::from("Highest Load"),
        Line::from(""),
        Line::from(Span::styled(
            "Agent-01",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Load: 8.7/10"),
        Line::from("Tasks: 23"),
        Line::from("Queue: 5"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🔴 Most Loaded "))
    .alignment(Alignment::Center);
    f.render_widget(most_loaded_card, overview_layout[1]);

    // Least loaded agent card
    let least_loaded_card = Paragraph::new(vec![
        Line::from("Lowest Load"),
        Line::from(""),
        Line::from(Span::styled(
            "Agent-05",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Load: 2.1/10"),
        Line::from("Tasks: 3"),
        Line::from("Queue: 0"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🟢 Least Loaded "))
    .alignment(Alignment::Center);
    f.render_widget(least_loaded_card, overview_layout[2]);

    // Balance score
    let balance_score = 78;
    let balance_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" ⚖️ Balance "))
        .set_style(Style::default().fg(if balance_score > 80 {
            Color::Green
        } else if balance_score > 60 {
            Color::Yellow
        } else {
            Color::Red
        }))
        .percent(balance_score)
        .label(format!("{}%", balance_score));
    f.render_widget(balance_gauge, overview_layout[3]);
}

/// Draw agent loads table
fn draw_agent_loads(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Agent"),
        Cell::from("Active Tasks"),
        Cell::from("Queue Depth"),
        Cell::from("Load Score"),
        Cell::from("Capacity"),
        Cell::from("Response Time"),
        Cell::from("Status"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("Agent-01"),
            Cell::from("23"),
            Cell::from("5"),
            Cell::from("8.7/10"),
            Cell::from("87%"),
            Cell::from("2.1s"),
            Cell::from("🔴 High"),
        ]),
        Row::new(vec![
            Cell::from("Agent-02"),
            Cell::from("18"),
            Cell::from("3"),
            Cell::from("6.2/10"),
            Cell::from("62%"),
            Cell::from("1.8s"),
            Cell::from("🟡 Medium"),
        ]),
        Row::new(vec![
            Cell::from("Agent-03"),
            Cell::from("21"),
            Cell::from("4"),
            Cell::from("7.8/10"),
            Cell::from("78%"),
            Cell::from("1.9s"),
            Cell::from("🟡 Medium"),
        ]),
        Row::new(vec![
            Cell::from("Agent-04"),
            Cell::from("15"),
            Cell::from("2"),
            Cell::from("5.1/10"),
            Cell::from("51%"),
            Cell::from("1.6s"),
            Cell::from("🟢 Low"),
        ]),
        Row::new(vec![
            Cell::from("Agent-05"),
            Cell::from("3"),
            Cell::from("0"),
            Cell::from("2.1/10"),
            Cell::from("21%"),
            Cell::from("1.4s"),
            Cell::from("🟢 Low"),
        ]),
        Row::new(vec![
            Cell::from("Agent-06"),
            Cell::from("19"),
            Cell::from("3"),
            Cell::from("6.8/10"),
            Cell::from("68%"),
            Cell::from("1.7s"),
            Cell::from("🟡 Medium"),
        ]),
    ];

    let loads_table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" ⚖️ Agent Load Status "));

    f.render_widget(loads_table, area);
}

/// Draw balancing controls
fn draw_balancing_controls(f: &mut Frame, _app: &App, area: Rect) {
    let controls_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Rebalancing actions
            Constraint::Percentage(50), // Balancing strategy
        ])
        .split(area);

    // Rebalancing actions
    let actions_text = vec![
        Line::from("🔄 Rebalancing Actions"),
        Line::from(""),
        Line::from("Immediate Actions:"),
        Line::from(""),
        Line::from("🔴 Agent-01 (overloaded)"),
        Line::from("[1] Redistribute 5 tasks"),
        Line::from("[2] Enable burst capacity"),
        Line::from("[3] Activate secondary agents"),
        Line::from(""),
        Line::from("🟢 Agent-05 (underutilized)"),
        Line::from("[4] Route compatible tasks"),
        Line::from("[5] Cross-train for more types"),
        Line::from("[6] Increase routing weight"),
        Line::from(""),
        Line::from("System Actions:"),
        Line::from("[A] Auto-rebalance all"),
        Line::from("[M] Manual redistribution"),
        Line::from("[P] Predictive balancing"),
        Line::from("[E] Emergency load shedding"),
    ];

    let actions_paragraph = Paragraph::new(actions_text)
        .block(Block::default().borders(Borders::ALL).title(" 🔄 Actions "))
        .wrap(Wrap { trim: true });
    f.render_widget(actions_paragraph, controls_layout[0]);

    // Balancing strategy
    let strategy_text = vec![
        Line::from("⚙️ Balancing Strategy"),
        Line::from(""),
        Line::from("Current Strategy: Capability-Weighted"),
        Line::from(""),
        Line::from("Strategy Options:"),
        Line::from(""),
        Line::from("🟢 [1] Even Distribution"),
        Line::from("   Pros: Simple, fair"),
        Line::from("   Cons: Ignores capabilities"),
        Line::from(""),
        Line::from("⭐ [2] Capability-Weighted (Current)"),
        Line::from("   Pros: Efficient, quality-focused"),
        Line::from("   Cons: Can create imbalance"),
        Line::from(""),
        Line::from("⚡ [3] Performance-Weighted"),
        Line::from("   Pros: Speed-optimized"),
        Line::from("   Cons: May overload fast agents"),
        Line::from(""),
        Line::from("💰 [4] Cost-Optimized"),
        Line::from("   Pros: Budget-friendly"),
        Line::from("   Cons: May sacrifice quality"),
        Line::from(""),
        Line::from("[S] Switch Strategy  [C] Custom Weights"),
    ];

    let strategy_paragraph = Paragraph::new(strategy_text)
        .block(Block::default().borders(Borders::ALL).title(" ⚙️ Strategy "))
        .wrap(Wrap { trim: true });
    f.render_widget(strategy_paragraph, controls_layout[1]);
}

/// Draw specialization opportunities
fn draw_specialization_opportunities(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(70), // Opportunities table
            Constraint::Percentage(30), // Opportunity details
        ])
        .split(area);

    // Opportunities table
    draw_opportunities_table(f, app, chunks[0]);

    // Opportunity implementation
    draw_opportunity_implementation(f, app, chunks[1]);
}

/// Draw opportunities table
fn draw_opportunities_table(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Agent"),
        Cell::from("Opportunity"),
        Cell::from("Score"),
        Cell::from("Impact"),
        Cell::from("Effort"),
        Cell::from("Timeline"),
        Cell::from("Status"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("Agent-05"),
            Cell::from("Mathematics Mastery"),
            Cell::from("92%"),
            Cell::from("🟢 High"),
            Cell::from("🟡 Medium"),
            Cell::from("3 weeks"),
            Cell::from("📋 Planned"),
        ]),
        Row::new(vec![
            Cell::from("Agent-09"),
            Cell::from("NLP Specialization"),
            Cell::from("87%"),
            Cell::from("🟢 High"),
            Cell::from("🟢 Low"),
            Cell::from("2 weeks"),
            Cell::from("🚀 Ready"),
        ]),
        Row::new(vec![
            Cell::from("Agent-12"),
            Cell::from("Code + Docs Cross-training"),
            Cell::from("84%"),
            Cell::from("🟡 Medium"),
            Cell::from("🟢 Low"),
            Cell::from("4 weeks"),
            Cell::from("💭 Proposed"),
        ]),
        Row::new(vec![
            Cell::from("Agent-15"),
            Cell::from("Data Analysis Advanced"),
            Cell::from("79%"),
            Cell::from("🟡 Medium"),
            Cell::from("🟡 Medium"),
            Cell::from("5 weeks"),
            Cell::from("💭 Proposed"),
        ]),
        Row::new(vec![
            Cell::from("Agent-18"),
            Cell::from("Creative Writing"),
            Cell::from("76%"),
            Cell::from("🟡 Medium"),
            Cell::from("🔴 High"),
            Cell::from("8 weeks"),
            Cell::from("🔍 Analysis"),
        ]),
        Row::new(vec![
            Cell::from("Agent-21"),
            Cell::from("Research + Education"),
            Cell::from("73%"),
            Cell::from("🟢 High"),
            Cell::from("🟡 Medium"),
            Cell::from("6 weeks"),
            Cell::from("🔍 Analysis"),
        ]),
    ];

    let opportunities_table = Table::new(
        rows,
        [
            Constraint::Percentage(12),
            Constraint::Percentage(25),
            Constraint::Percentage(10),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(17),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 💡 Specialization Opportunities "));

    f.render_widget(opportunities_table, area);
}

/// Draw opportunity implementation details
fn draw_opportunity_implementation(f: &mut Frame, _app: &App, area: Rect) {
    let implementation_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Implementation plan
            Constraint::Percentage(50), // Expected benefits
        ])
        .split(area);

    // Implementation plan
    let plan_text = vec![
        Line::from("📋 Implementation Plan"),
        Line::from(""),
        Line::from("Selected: Agent-05 → Mathematics Mastery"),
        Line::from(""),
        Line::from("Training Approach: Intensive"),
        Line::from("Required Tasks: 50 mathematical problems"),
        Line::from("Timeline: 3 weeks"),
        Line::from(""),
        Line::from("Phase 1 (Week 1):"),
        Line::from("• Basic algebra and arithmetic"),
        Line::from("• Success target: 80%"),
        Line::from(""),
        Line::from("Phase 2 (Week 2):"),
        Line::from("• Calculus and statistics"),
        Line::from("• Success target: 85%"),
        Line::from(""),
        Line::from("Phase 3 (Week 3):"),
        Line::from("• Advanced mathematics"),
        Line::from("• Success target: 90%"),
        Line::from(""),
        Line::from("Current Progress: Not started"),
        Line::from("[S] Start Training  [C] Customize"),
    ];

    let plan_paragraph = Paragraph::new(plan_text)
        .block(Block::default().borders(Borders::ALL).title(" 📋 Plan "))
        .wrap(Wrap { trim: true });
    f.render_widget(plan_paragraph, implementation_layout[0]);

    // Expected benefits
    let benefits_text = vec![
        Line::from("🎯 Expected Benefits"),
        Line::from(""),
        Line::from("Performance Improvements:"),
        Line::from("• Math task success: 67% → 90%"),
        Line::from("• Response time: 2.3s → 1.8s"),
        Line::from("• Quality score: 0.74 → 0.88"),
        Line::from(""),
        Line::from("System Impact:"),
        Line::from("• Fill critical math specialization gap"),
        Line::from("• Reduce routing to suboptimal agents"),
        Line::from("• Improve overall success rate by 3%"),
        Line::from("• Better load distribution"),
        Line::from(""),
        Line::from("Strategic Value:"),
        Line::from("• High demand specialization"),
        Line::from("• 15% of tasks are math-related"),
        Line::from("• Enables complex problem solving"),
        Line::from("• Foundation for STEM education"),
        Line::from(""),
        Line::from("ROI: Implementation cost vs 23% efficiency gain"),
        Line::from("Confidence: 92% success probability"),
    ];

    let benefits_paragraph = Paragraph::new(benefits_text)
        .block(Block::default().borders(Borders::ALL).title(" 🎯 Benefits "))
        .wrap(Wrap { trim: true });
    f.render_widget(benefits_paragraph, implementation_layout[1]);
}
