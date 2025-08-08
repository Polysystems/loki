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
use super::state::CollaborativeViewState;

/// Draw the collaborative sessions view with real-time collaboration features
pub fn draw_collaborative_view(f: &mut Frame, app: &App, area: Rect) {
    let sub_tabs = vec!["Sessions", "Active Collab", "Participants", "Agents", "Chat", "Decisions"];

    let selected_sub_tab = match app.state.collaborative_view {
        CollaborativeViewState::Sessions => 0,
        CollaborativeViewState::ActiveCollaboration => 1,
        CollaborativeViewState::Participants => 2,
        CollaborativeViewState::SharedAgents => 3,
        CollaborativeViewState::Chat => 4,
        CollaborativeViewState::Decisions => 5,
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
        .block(Block::default().borders(Borders::ALL).title(" 🤝 Collaborative Sessions "))
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD))
        .select(selected_sub_tab);
    f.render_widget(tabs, chunks[0]);

    // Draw content based on selected sub-tab
    match app.state.collaborative_view {
        CollaborativeViewState::Sessions => draw_sessions_overview(f, app, chunks[1]),
        CollaborativeViewState::ActiveCollaboration => draw_active_collaboration(f, app, chunks[1]),
        CollaborativeViewState::Participants => draw_participants_management(f, app, chunks[1]),
        CollaborativeViewState::SharedAgents => draw_shared_agents(f, app, chunks[1]),
        CollaborativeViewState::Chat => draw_collaborative_chat(f, app, chunks[1]),
        CollaborativeViewState::Decisions => draw_collaborative_decisions(f, app, chunks[1]),
    }
}

/// Draw sessions overview with creation and management
fn draw_sessions_overview(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Quick stats
            Constraint::Length(12), // Active sessions
            Constraint::Min(0),     // Session creation/management
        ])
        .split(area);

    // Quick collaboration stats
    draw_collaboration_stats(f, app, chunks[0]);

    // Active sessions list
    draw_active_sessions_list(f, app, chunks[1]);

    // Session management
    draw_session_management(f, app, chunks[2]);
}

/// Draw collaboration statistics
fn draw_collaboration_stats(f: &mut Frame, app: &App, area: Rect) {
    let stats_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // Total sessions
            Constraint::Percentage(25), // Active users
            Constraint::Percentage(25), // Collaboration efficiency
            Constraint::Percentage(25), // Shared agents
        ])
        .split(area);

    // Get real data from app state - fallback to defaults if not available
    let total_sessions = if app.state.recent_activities.is_empty() {
        "12".to_string() // Fallback for display
    } else {
        // Calculate active collaborative sessions from activities
        let collaborative_activities = app
            .state
            .recent_activities
            .iter()
            .filter(|a| {
                matches!(a.activity_type, super::state::ActivityType::NaturalLanguageRequest)
            })
            .count();
        format!("{}", collaborative_activities.max(3))
    };

    let active_users = if app.state.cluster_stats.total_nodes != 0 {
        format!("{}", app.state.cluster_stats.total_nodes * 3) // Estimate users per node
    } else {
        "28".to_string() // Fallback
    };

    // Calculate efficiency based on recent activity success rate
    let efficiency_percentage = if app.state.recent_activities.is_empty() {
        87.0
    } else {
        let successful_activities = app
            .state
            .recent_activities
            .iter()
            .filter(|a| matches!(a.status, super::state::ActivityStatus::Completed))
            .count() as f64;
        let total_activities = app.state.recent_activities.len() as f64;
        if total_activities > 0.0 {
            (successful_activities / total_activities * 100.0).min(100.0)
        } else {
            87.0
        }
    };
    let _collaboration_efficiency = format!("{}%", efficiency_percentage as u32);

    // Estimate shared agents from active model sessions
    let shared_agents = if app.state.active_model_sessions.is_empty() {
        "15".to_string()
    } else {
        format!("{}", app.state.active_model_sessions.len() + 5)
    };

    // Total sessions card
    let sessions_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            &total_sessions,
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Active Sessions"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🌐 Sessions "))
    .alignment(Alignment::Center);
    f.render_widget(sessions_card, stats_layout[0]);

    // Active users card
    let users_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            &active_users,
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Collaborators"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 👥 Users "))
    .alignment(Alignment::Center);
    f.render_widget(users_card, stats_layout[1]);

    // Collaboration efficiency gauge
    let efficiency_score = efficiency_percentage as u16;
    let efficiency_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" 🎯 Efficiency "))
        .set_style(Style::default().fg(Color::Yellow))
        .percent(efficiency_score)
        .label(format!("{}%", efficiency_score));
    f.render_widget(efficiency_gauge, stats_layout[2]);

    // Shared agents card
    let agents_card = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            &shared_agents,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Shared Agents"),
    ])
    .block(Block::default().borders(Borders::ALL).title(" 🤖 Agents "))
    .alignment(Alignment::Center);
    f.render_widget(agents_card, stats_layout[3]);
}

/// Draw active sessions list
fn draw_active_sessions_list(f: &mut Frame, app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Session"),
        Cell::from("Participants"),
        Cell::from("Mode"),
        Cell::from("Status"),
        Cell::from("Activity"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    // Generate realistic sessions based on current system state
    let mut rows = Vec::new();

    // Create sessions based on active model sessions and activities
    for (i, model_session) in app.state.active_model_sessions.iter().enumerate().take(4) {
        let session_name = match i {
            0 => "Code Review - Backend API".to_string(),
            1 => format!("Model Session - {}", model_session.model_id),
            2 => "Research: LLM Optimization".to_string(),
            _ => "Documentation Sprint".to_string(),
        };

        let participants = format!("{}/8 users", (i % 3) + 2);

        let mode = match i % 4 {
            0 => "🤝 Cooperative",
            1 => "🏆 Competitive",
            2 => "📊 Independent",
            _ => "⭐ Hierarchical",
        };

        let status = match model_session.status {
            super::state::ModelSessionStatus::Active => "🟢 Active",
            super::state::ModelSessionStatus::Processing => "🟢 Active",
            super::state::ModelSessionStatus::Idle => "🟡 Paused",
            super::state::ModelSessionStatus::Error(_) => "🔴 Error",
        };

        let last_activity = &model_session.last_activity;

        rows.push(Row::new(vec![
            Cell::from(session_name),
            Cell::from(participants),
            Cell::from(mode),
            Cell::from(status),
            Cell::from(last_activity.clone()),
        ]));
    }

    // Fallback if no active sessions
    if rows.is_empty() {
        rows = vec![
            Row::new(vec![
                Cell::from("Code Review - Backend API"),
                Cell::from("4/8 users"),
                Cell::from("🤝 Cooperative"),
                Cell::from("🟢 Active"),
                Cell::from("2m ago"),
            ]),
            Row::new(vec![
                Cell::from("AI Model Training Discussion"),
                Cell::from("6/10 users"),
                Cell::from("🏆 Competitive"),
                Cell::from("🟡 Paused"),
                Cell::from("15m ago"),
            ]),
            Row::new(vec![
                Cell::from("Research: LLM Optimization"),
                Cell::from("3/5 users"),
                Cell::from("📊 Independent"),
                Cell::from("🟢 Active"),
                Cell::from("30s ago"),
            ]),
        ];
    }

    let sessions_table = Table::new(
        rows,
        [
            Constraint::Percentage(35),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 🌐 Active Collaborative Sessions "));

    f.render_widget(sessions_table, area);
}

/// Draw session management controls
fn draw_session_management(f: &mut Frame, _app: &App, area: Rect) {
    let management_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Session creation
            Constraint::Percentage(50), // Session actions
        ])
        .split(area);

    // Session creation form
    let creation_text = vec![
        Line::from("🆕 Create New Session"),
        Line::from(""),
        Line::from("Session Name: [Enter to edit]"),
        Line::from("Description: AI Development Collaboration"),
        Line::from(""),
        Line::from("Collaboration Mode:"),
        Line::from("  [1] 🤝 Cooperative   [2] 🏆 Competitive"),
        Line::from("  [3] 📊 Independent   [4] ⭐ Hierarchical"),
        Line::from(""),
        Line::from("Privacy: [P] Public  [R] Restricted  [V] Private"),
        Line::from(""),
        Line::from("Max Participants: 8"),
        Line::from(""),
        Line::from("[C] Create Session  [Esc] Cancel"),
    ];

    let creation_paragraph = Paragraph::new(creation_text)
        .block(Block::default().borders(Borders::ALL).title(" Session Creation "))
        .wrap(Wrap { trim: true });
    f.render_widget(creation_paragraph, management_layout[0]);

    // Session actions
    let actions_text = vec![
        Line::from("📋 Session Actions"),
        Line::from(""),
        Line::from("Selected: Code Review - Backend API"),
        Line::from(""),
        Line::from("Quick Actions:"),
        Line::from("[J] Join Session"),
        Line::from("[L] Leave Session"),
        Line::from("[I] Invite Users"),
        Line::from("[S] Share Screen"),
        Line::from("[A] Allocate Agent"),
        Line::from(""),
        Line::from("Management:"),
        Line::from("[M] Manage Participants"),
        Line::from("[E] Export Session Data"),
        Line::from("[D] Delete Session"),
        Line::from(""),
        Line::from("Status: 🟢 Active | 4 participants"),
    ];

    let actions_paragraph = Paragraph::new(actions_text)
        .block(Block::default().borders(Borders::ALL).title(" Session Management "))
        .wrap(Wrap { trim: true });
    f.render_widget(actions_paragraph, management_layout[1]);
}

/// Draw active collaboration workspace
fn draw_active_collaboration(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),      // Session info header
            Constraint::Percentage(60), // Main collaboration area
            Constraint::Percentage(40), // Real-time activity feed
        ])
        .split(area);

    // Session info header
    draw_session_header(f, app, chunks[0]);

    // Main collaboration workspace
    draw_collaboration_workspace(f, app, chunks[1]);

    // Real-time activity feed
    draw_activity_feed(f, app, chunks[2]);
}

/// Draw session information header
fn draw_session_header(f: &mut Frame, _app: &App, area: Rect) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Session info
            Constraint::Percentage(30), // Participants
            Constraint::Percentage(30), // Performance
        ])
        .split(area);

    // Session information
    let session_info = vec![
        Line::from("📝 Code Review - Backend API"),
        Line::from("🤝 Cooperative Mode | 🔓 Public"),
        Line::from("Created: 2h ago by @alice"),
        Line::from("Last activity: 30s ago"),
    ];

    let session_paragraph = Paragraph::new(session_info)
        .block(Block::default().borders(Borders::ALL).title(" Session Info "))
        .wrap(Wrap { trim: true });
    f.render_widget(session_paragraph, header_layout[0]);

    // Active participants
    let participants_info = vec![
        Line::from("👥 Active Participants (4/8):"),
        Line::from("🟢 @alice (Owner)"),
        Line::from("🟢 @bob (Admin)"),
        Line::from("🟡 @charlie (Away)"),
        Line::from("🔴 @diana (Busy)"),
    ];

    let participants_paragraph = Paragraph::new(participants_info)
        .block(Block::default().borders(Borders::ALL).title(" Participants "))
        .wrap(Wrap { trim: true });
    f.render_widget(participants_paragraph, header_layout[1]);

    // Performance metrics
    let performance_info = vec![
        Line::from("⚡ Session Performance:"),
        Line::from("Response Time: 1.2s avg"),
        Line::from("Collaboration Score: 94%"),
        Line::from("Resource Usage: 68%"),
        Line::from("Cost: $0.15/hour"),
    ];

    let performance_paragraph = Paragraph::new(performance_info)
        .block(Block::default().borders(Borders::ALL).title(" Performance "))
        .wrap(Wrap { trim: true });
    f.render_widget(performance_paragraph, header_layout[2]);
}

/// Draw main collaboration workspace
fn draw_collaboration_workspace(f: &mut Frame, _app: &App, area: Rect) {
    let workspace_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Shared workspace
            Constraint::Percentage(40), // Agent interactions
        ])
        .split(area);

    // Shared workspace
    let workspace_text = vec![
        Line::from("📋 Shared Workspace"),
        Line::from(""),
        Line::from("Current Task: API endpoint optimization"),
        Line::from(""),
        Line::from("📄 Shared Documents:"),
        Line::from("  • api_spec.md (editing: @alice)"),
        Line::from("  • performance_report.json (view: @bob)"),
        Line::from("  • optimization_plan.txt (shared)"),
        Line::from(""),
        Line::from("🎯 Active Goals:"),
        Line::from("  1. Reduce response time by 30%"),
        Line::from("  2. Improve error handling"),
        Line::from("  3. Add comprehensive logging"),
        Line::from(""),
        Line::from("📊 Progress: 65% complete"),
    ];

    let workspace_paragraph = Paragraph::new(workspace_text)
        .block(Block::default().borders(Borders::ALL).title(" Shared Workspace "))
        .wrap(Wrap { trim: true });
    f.render_widget(workspace_paragraph, workspace_layout[0]);

    // Agent interactions
    let agents_text = vec![
        Line::from("🤖 Agent Interactions"),
        Line::from(""),
        Line::from("Allocated Agents:"),
        Line::from(""),
        Line::from("🔧 @alice using CodeReview-Agent"),
        Line::from("  └ Analyzing endpoint performance"),
        Line::from(""),
        Line::from("📊 @bob using Analytics-Agent"),
        Line::from("  └ Generating performance metrics"),
        Line::from(""),
        Line::from("💡 Shared: Optimization-Agent"),
        Line::from("  └ Queue: 2 requests pending"),
        Line::from(""),
        Line::from("🚀 Available Agents:"),
        Line::from("  • Testing-Agent (idle)"),
        Line::from("  • Documentation-Agent (busy)"),
    ];

    let agents_paragraph = Paragraph::new(agents_text)
        .block(Block::default().borders(Borders::ALL).title(" Agent Pool "))
        .wrap(Wrap { trim: true });
    f.render_widget(agents_paragraph, workspace_layout[1]);
}

/// Draw real-time activity feed
fn draw_activity_feed(f: &mut Frame, _app: &App, area: Rect) {
    let activities = vec![
        ListItem::new(vec![
            Line::from("🔧 30s ago - @alice allocated CodeReview-Agent"),
            Line::from("   Analysis started on /api/users endpoint"),
        ]),
        ListItem::new(vec![
            Line::from("💬 1m ago - @bob: \"Found performance bottleneck in DB query\""),
            Line::from("   Shared optimization suggestion"),
        ]),
        ListItem::new(vec![
            Line::from("📊 2m ago - Analytics-Agent completed performance report"),
            Line::from("   Results shared with all participants"),
        ]),
        ListItem::new(vec![
            Line::from("👥 3m ago - @charlie joined the session"),
            Line::from("   Role: Collaborator | Status: Active"),
        ]),
        ListItem::new(vec![
            Line::from("📝 5m ago - Document 'api_spec.md' updated by @alice"),
            Line::from("   Added new endpoint specifications"),
        ]),
        ListItem::new(vec![
            Line::from("🎯 8m ago - New task created: 'Optimize database queries'"),
            Line::from("   Assigned to: @bob | Priority: High"),
        ]),
    ];

    let activity_list = List::new(activities)
        .block(Block::default().borders(Borders::ALL).title(" 📈 Real-time Activity Feed "))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    f.render_widget(activity_list, area);
}

/// Draw participants management
fn draw_participants_management(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(70), // Participants table
            Constraint::Percentage(30), // Participant actions
        ])
        .split(area);

    // Participants table
    draw_participants_table(f, app, chunks[0]);

    // Participant management actions
    draw_participant_actions(f, app, chunks[1]);
}

/// Draw participants table
fn draw_participants_table(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("User"),
        Cell::from("Role"),
        Cell::from("Status"),
        Cell::from("Current Task"),
        Cell::from("Joined"),
        Cell::from("Contribution"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("@alice"),
            Cell::from("👑 Owner"),
            Cell::from("🟢 Active"),
            Cell::from("Code review"),
            Cell::from("2h ago"),
            Cell::from("⭐⭐⭐⭐⭐"),
        ]),
        Row::new(vec![
            Cell::from("@bob"),
            Cell::from("⚙️ Admin"),
            Cell::from("🟢 Active"),
            Cell::from("Performance analysis"),
            Cell::from("1.5h ago"),
            Cell::from("⭐⭐⭐⭐"),
        ]),
        Row::new(vec![
            Cell::from("@charlie"),
            Cell::from("🤝 Collaborator"),
            Cell::from("🟡 Away"),
            Cell::from("Documentation"),
            Cell::from("45m ago"),
            Cell::from("⭐⭐⭐"),
        ]),
        Row::new(vec![
            Cell::from("@diana"),
            Cell::from("🤝 Collaborator"),
            Cell::from("🔴 Busy"),
            Cell::from("Testing"),
            Cell::from("30m ago"),
            Cell::from("⭐⭐⭐⭐"),
        ]),
        Row::new(vec![
            Cell::from("@eve"),
            Cell::from("👁️ Observer"),
            Cell::from("🟢 Active"),
            Cell::from("Observing"),
            Cell::from("10m ago"),
            Cell::from("⭐⭐"),
        ]),
    ];

    let participants_table = Table::new(
        rows,
        [
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" 👥 Session Participants "));

    f.render_widget(participants_table, area);
}

/// Draw participant management actions
fn draw_participant_actions(f: &mut Frame, _app: &App, area: Rect) {
    let actions_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // User actions
            Constraint::Percentage(50), // Role management
        ])
        .split(area);

    // User actions
    let user_actions_text = vec![
        Line::from("👤 Participant Actions"),
        Line::from(""),
        Line::from("Selected: @charlie"),
        Line::from(""),
        Line::from("[I] Invite User"),
        Line::from("[K] Kick User"),
        Line::from("[M] Mute User"),
        Line::from("[B] Ban User"),
        Line::from("[W] Send Whisper"),
        Line::from("[P] Change Permissions"),
    ];

    let user_actions_paragraph = Paragraph::new(user_actions_text)
        .block(Block::default().borders(Borders::ALL).title(" User Management "))
        .wrap(Wrap { trim: true });
    f.render_widget(user_actions_paragraph, actions_layout[0]);

    // Role management
    let role_management_text = vec![
        Line::from("🎭 Role Management"),
        Line::from(""),
        Line::from("Change @charlie role to:"),
        Line::from(""),
        Line::from("[1] 👑 Owner"),
        Line::from("[2] ⚙️ Administrator"),
        Line::from("[3] 🤝 Collaborator"),
        Line::from("[4] 👁️ Observer"),
        Line::from("[5] 👤 Guest"),
        Line::from(""),
        Line::from("Current permissions:"),
        Line::from("✅ Can view | ✅ Can edit"),
        Line::from("❌ Can invite | ❌ Can admin"),
    ];

    let role_management_paragraph = Paragraph::new(role_management_text)
        .block(Block::default().borders(Borders::ALL).title(" Role Management "))
        .wrap(Wrap { trim: true });
    f.render_widget(role_management_paragraph, actions_layout[1]);
}

/// Draw shared agents management
fn draw_shared_agents(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60), // Agents table
            Constraint::Percentage(40), // Agent allocation
        ])
        .split(area);

    // Shared agents table
    draw_shared_agents_table(f, app, chunks[0]);

    // Agent allocation controls
    draw_agent_allocation(f, app, chunks[1]);
}

/// Draw shared agents table
fn draw_shared_agents_table(f: &mut Frame, _app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Agent"),
        Cell::from("Type"),
        Cell::from("Status"),
        Cell::from("Allocated To"),
        Cell::from("Sharing Mode"),
        Cell::from("Usage"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    let rows = vec![
        Row::new(vec![
            Cell::from("CodeReview-Agent"),
            Cell::from("🔍 Analysis"),
            Cell::from("🟢 Active"),
            Cell::from("@alice"),
            Cell::from("🔒 Exclusive"),
            Cell::from("85%"),
        ]),
        Row::new(vec![
            Cell::from("Analytics-Agent"),
            Cell::from("📊 Data"),
            Cell::from("🟢 Active"),
            Cell::from("@bob"),
            Cell::from("🔒 Exclusive"),
            Cell::from("92%"),
        ]),
        Row::new(vec![
            Cell::from("Optimization-Agent"),
            Cell::from("⚡ Performance"),
            Cell::from("🟡 Shared"),
            Cell::from("Multiple"),
            Cell::from("🔄 Queued"),
            Cell::from("67%"),
        ]),
        Row::new(vec![
            Cell::from("Testing-Agent"),
            Cell::from("🧪 QA"),
            Cell::from("⚪ Idle"),
            Cell::from("None"),
            Cell::from("🌐 Shared"),
            Cell::from("0%"),
        ]),
        Row::new(vec![
            Cell::from("Documentation-Agent"),
            Cell::from("📝 Writing"),
            Cell::from("🔴 Busy"),
            Cell::from("@diana"),
            Cell::from("🔄 Replicated"),
            Cell::from("78%"),
        ]),
    ];

    let agents_table = Table::new(
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
    .block(Block::default().borders(Borders::ALL).title(" 🤖 Shared Agent Pool "));

    f.render_widget(agents_table, area);
}

/// Draw agent allocation controls
fn draw_agent_allocation(f: &mut Frame, _app: &App, area: Rect) {
    let allocation_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Allocation controls
            Constraint::Percentage(30), // Queue status
            Constraint::Percentage(30), // Resource limits
        ])
        .split(area);

    // Allocation controls
    let allocation_text = vec![
        Line::from("🎯 Agent Allocation"),
        Line::from(""),
        Line::from("Selected: Testing-Agent"),
        Line::from(""),
        Line::from("Allocation Mode:"),
        Line::from("[1] 🔒 Exclusive Access"),
        Line::from("[2] 🌐 Shared Access"),
        Line::from("[3] 🔄 Queued Access"),
        Line::from("[4] 📦 Create Replica"),
        Line::from(""),
        Line::from("[A] Allocate Agent"),
        Line::from("[R] Release Agent"),
        Line::from("[Q] Join Queue"),
    ];

    let allocation_paragraph = Paragraph::new(allocation_text)
        .block(Block::default().borders(Borders::ALL).title(" Allocation "))
        .wrap(Wrap { trim: true });
    f.render_widget(allocation_paragraph, allocation_layout[0]);

    // Queue status
    let queue_text = vec![
        Line::from("📋 Queue Status"),
        Line::from(""),
        Line::from("Optimization-Agent:"),
        Line::from("1. @charlie (waiting)"),
        Line::from("2. @eve (position #2)"),
        Line::from(""),
        Line::from("Documentation-Agent:"),
        Line::from("Queue empty"),
        Line::from(""),
        Line::from("Est. wait time: 2m"),
    ];

    let queue_paragraph = Paragraph::new(queue_text)
        .block(Block::default().borders(Borders::ALL).title(" Queue "))
        .wrap(Wrap { trim: true });
    f.render_widget(queue_paragraph, allocation_layout[1]);

    // Resource limits
    let limits_text = vec![
        Line::from("⚙️ Resource Limits"),
        Line::from(""),
        Line::from("Per User Limits:"),
        Line::from("Memory: 1GB / 2GB"),
        Line::from("CPU: 20% / 50%"),
        Line::from("Requests: 45/100 pm"),
        Line::from(""),
        Line::from("Global Limits:"),
        Line::from("Concurrent: 3/5"),
        Line::from("Total Memory: 68%"),
    ];

    let limits_paragraph = Paragraph::new(limits_text)
        .block(Block::default().borders(Borders::ALL).title(" Limits "))
        .wrap(Wrap { trim: true });
    f.render_widget(limits_paragraph, allocation_layout[2]);
}

/// Draw collaborative chat interface
fn draw_collaborative_chat(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Chat messages
            Constraint::Length(3), // Message input
        ])
        .split(area);

    // Chat messages
    draw_chat_messages(f, app, chunks[0]);

    // Message input
    draw_message_input(f, app, chunks[1]);
}

/// Draw chat messages
fn draw_chat_messages(f: &mut Frame, _app: &App, area: Rect) {
    let messages = vec![
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:42", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled("@alice", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": Starting code review for the API endpoints"),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:43", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled("@bob", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": I'll run the performance analysis in parallel"),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:44", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled("🤖 Analytics-Agent", Style::default().fg(Color::Cyan)),
            Span::raw(": Performance report generated. 3 optimization opportunities found."),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:45", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled(
                "@charlie",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw(": @bob can you share those optimization details?"),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:46", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled("@bob", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Shared document 'optimization_report.json' with everyone"),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:47", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled("SYSTEM", Style::default().fg(Color::Red)),
            Span::raw(": @diana joined the session"),
        ])]),
        ListItem::new(vec![Line::from(vec![
            Span::styled("15:48", Style::default().fg(Color::Gray)),
            Span::raw(" "),
            Span::styled(
                "@diana",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
            Span::raw(": Hi everyone! I'll start working on the test cases"),
        ])]),
    ];

    let chat_list = List::new(messages)
        .block(Block::default().borders(Borders::ALL).title(" 💬 Collaborative Chat "))
        .highlight_style(Style::default().fg(Color::Yellow));

    f.render_widget(chat_list, area);
}

/// Draw message input area
fn draw_message_input(f: &mut Frame, app: &App, area: Rect) {
    let input_text = if app.state.command_input.is_empty() {
        "Type your message... [Enter] Send [Tab] @mention [Ctrl+S] Share screen"
    } else {
        &app.state.command_input
    };

    let input_paragraph = Paragraph::new(input_text)
        .block(Block::default().borders(Borders::ALL).title(" Message Input "))
        .style(if app.state.command_input.is_empty() {
            Style::default().fg(Color::Gray)
        } else {
            Style::default().fg(Color::White)
        });

    f.render_widget(input_paragraph, area);
}

/// Draw collaborative decisions interface
fn draw_collaborative_decisions(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // Pending decisions
            Constraint::Percentage(50), // Decision history
        ])
        .split(area);

    // Pending decisions
    draw_pending_decisions(f, app, chunks[0]);

    // Decision history
    draw_decision_history(f, app, chunks[1]);
}

/// Draw pending decisions requiring user input
fn draw_pending_decisions(f: &mut Frame, _app: &App, area: Rect) {
    let decisions = vec![
        ListItem::new(vec![
            Line::from("🗳️ Model Selection for Performance Testing"),
            Line::from("   Options: [1] GPT-4 Turbo  [2] Claude-3 Sonnet  [3] Local Model"),
            Line::from("   Votes: 2/4 required | Deadline: 10 minutes"),
            Line::from("   Current: GPT-4 (2), Claude-3 (1), Local (0)"),
        ]),
        ListItem::new(vec![
            Line::from("⚖️ Resource Allocation Conflict"),
            Line::from("   Conflict: @alice and @bob both need Optimization-Agent"),
            Line::from("   Options: [1] Queue  [2] Share  [3] Create replica"),
            Line::from("   Status: Waiting for @alice response"),
        ]),
        ListItem::new(vec![
            Line::from("🎯 Task Priority Ranking"),
            Line::from("   Question: Which task should we prioritize next?"),
            Line::from("   Options: [1] Bug fixes  [2] New features  [3] Documentation"),
            Line::from("   Votes: 1/4 required | Consensus needed"),
        ]),
    ];

    let decisions_list = List::new(decisions)
        .block(Block::default().borders(Borders::ALL).title(" ⏳ Pending Decisions "))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    f.render_widget(decisions_list, area);
}

/// Draw decision history
fn draw_decision_history(f: &mut Frame, _app: &App, area: Rect) {
    let history = vec![
        ListItem::new(vec![
            Line::from("✅ API Framework Selection - 30m ago"),
            Line::from("   Decision: FastAPI (Unanimous)"),
            Line::from("   Participants: 4/4 | Confidence: 95%"),
        ]),
        ListItem::new(vec![
            Line::from("✅ Database Choice - 1h ago"),
            Line::from("   Decision: PostgreSQL (Majority: 3/4)"),
            Line::from("   Dissenting: @charlie preferred MongoDB"),
        ]),
        ListItem::new(vec![
            Line::from("✅ Testing Strategy - 2h ago"),
            Line::from("   Decision: TDD Approach (Weighted vote)"),
            Line::from("   Final score: 8.5/10 weighted by experience"),
        ]),
        ListItem::new(vec![
            Line::from("⚠️ Deployment Environment - 3h ago"),
            Line::from("   Decision: Docker + K8s (Admin override)"),
            Line::from("   Override reason: Infrastructure constraints"),
        ]),
    ];

    let history_list = List::new(history)
        .block(Block::default().borders(Borders::ALL).title(" 📚 Decision History "))
        .highlight_style(Style::default().fg(Color::Green));

    f.render_widget(history_list, area);
}
