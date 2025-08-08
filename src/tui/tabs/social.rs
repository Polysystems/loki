use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Color, Line, Span, Style, Modifier};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use crate::config::XTwitterConfig;
use crate::social::x_client::{XClient};
use crate::tui::App;
use crate::tui::ui::draw_sub_tab_navigation;

#[derive(Debug, Clone)]
pub struct SocialSettings {
    pub auto_post: bool,
    pub retry_attempts: u32,
    pub character_limit: u32,
    pub consistent_theme: bool,
}

impl Default for SocialSettings {
    fn default() -> Self {
        Self { auto_post: false, retry_attempts: 2, character_limit: 280, consistent_theme: true }
    }
}

#[derive(Debug, Clone)]
pub struct Tweet {
    pub id: String,
    pub text: String,
    pub author: String,
    pub timestamp: DateTime<Utc>,
    pub likes: u32,
    pub retweets: u32,
    pub replies: u32,
    pub is_reply: bool,
    pub is_retweet: bool,
}

impl Default for Tweet {
    fn default() -> Self {
        Self {
            id: String::new(),
            text: String::new(),
            author: String::new(),
            timestamp: Utc::now(),
            likes: 0,
            retweets: 0,
            replies: 0,
            is_reply: false,
            is_retweet: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccountManager {
    pub state: AccountManagerState,
    pub selected_index: usize,
    pub preferred_account_index: Option<usize>,
    pub input_buffer: String,
    pub accounts: Vec<SocialAccount>,
    pub active_client: Option<Arc<XClient>>,
    pub authenticated_user: Option<AuthenticatedUser>,

    pub temp_account_name: Option<String>,
    pub temp_api_key: Option<String>,
    pub temp_api_secret: Option<String>,
    pub temp_access_token: Option<String>,
    pub temp_access_token_secret: Option<String>,
    pub temp_bearer_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub id: String,
    pub username: String,
    pub name: String,
}

impl Default for AccountManager {
    fn default() -> Self {
        Self {
            state: AccountManagerState::Browsing,
            selected_index: 0,
            preferred_account_index: None,
            input_buffer: String::new(),
            accounts: vec![],
            active_client: None,
            authenticated_user: None,
            temp_account_name: None,
            temp_api_key: None,
            temp_api_secret: None,
            temp_access_token: None,
            temp_access_token_secret: None,
            temp_bearer_token: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AccountManagerState {
    Browsing,
    AddingAccount,
    EnteringAccountName,
    EnteringApiKey,
    EnteringApiSecret,
    EnteringAccessToken,
    EnteringAccessTokenSecret,
    EnteringBearerToken,
}

#[derive(Debug, Clone)]
pub struct SocialAccount {
    pub name: String,
    pub config: XTwitterConfig,
}

pub fn draw_tab_social(f: &mut Frame, app: &mut App, area: Rect) {
    // Check if system connector is available for enhanced rendering
    if app.system_connector.is_some() {
        draw_tab_social_enhanced(f, app, area);
        return;
    }

    // Legacy rendering
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    draw_sub_tab_navigation(f, &app.state.social_tabs, main_chunks[0]);

    match app.state.social_tabs.current_index {
        0 => draw_tweet_tab(f, app, main_chunks[1]),
        1 => draw_accounts_tab(f, app, main_chunks[1]),
        2 => draw_recent_tab(f, app, main_chunks[1]),
        3 => draw_social_settings_tab(f, app, main_chunks[1]),
        _ => draw_tweet_tab(f, app, main_chunks[1]),
    };
}

fn draw_tweet_tab(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Tweet input area
    let tweet_box = Block::default()
        .title(format!("Tweet Input ({}/280 chars)", app.state.tweet_input.len()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(
            if app.state.tweet_input.len() > 280 { Color::Red } else { Color::White }
        ));

    let input_text =
        if app.state.tweet_input.is_empty() { "What's happening?" } else { &app.state.tweet_input };

    let style = if app.state.tweet_input.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input_widget = Paragraph::new(input_text)
        .block(tweet_box)
        .style(style)
        .wrap(Wrap { trim: false });

    f.render_widget(input_widget, chunks[0]);

    // Status and account info
    let account_info = if let Some(user) = &app.state.account_manager.authenticated_user {
        format!("@{} | ", user.username)
    } else {
        "Not authenticated | ".to_string()
    };

    let status_text = if let Some(status) = &app.state.tweet_status {
        status.clone()
    } else {
        "Press Enter to tweet".to_string()
    };

    let status_line = Line::from(vec![
        Span::styled(account_info, Style::default().fg(Color::Cyan)),
        Span::styled(status_text, Style::default().fg(Color::Yellow)),
    ]);

    let status_widget = Paragraph::new(status_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::White)),
    );
    f.render_widget(status_widget, chunks[1]);

    // Help text
    let help_text = vec![
        Line::from("Keyboard shortcuts:"),
        Line::from(vec![
            Span::raw("  Enter: "),
            Span::styled("Send tweet", Style::default().fg(Color::Green)),
            Span::raw("  |  Ctrl+U: "),
            Span::styled("Clear input", Style::default().fg(Color::Yellow)),
            Span::raw("  |  Tab: "),
            Span::styled("Switch tabs", Style::default().fg(Color::Blue)),
        ]),
    ];

    let help_widget = Paragraph::new(help_text).block(
        Block::default()
            .title("Help")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(help_widget, chunks[2]);
}
fn draw_accounts_tab(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("Social Accounts")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::White));

    match app.state.account_manager.state {
        AccountManagerState::Browsing => {
            let mut items = Vec::new();

            for (i, acc) in app.state.account_manager.accounts.iter().enumerate() {
                let mut style = Style::default().fg(Color::White);

                if i == app.state.account_manager.selected_index {
                    style = style.bg(Color::Blue);
                }

                if Some(i) == app.state.account_manager.preferred_account_index {
                    style = style.bg(Color::Green).fg(Color::Black);
                }

                items.push(ListItem::new(acc.name.clone()).style(style));
            }

            // Add new account
            let add_style = if app.state.account_manager.selected_index
                == app.state.account_manager.accounts.len()
            {
                Style::default().bg(Color::Blue).fg(Color::White)
            } else {
                Style::default().fg(Color::Yellow)
            };

            items.push(ListItem::new("+ Add New Account").style(add_style));

            let list = List::new(items).block(block);
            f.render_widget(list, area);
        }

        AccountManagerState::EnteringAccountName
        | AccountManagerState::EnteringApiKey
        | AccountManagerState::EnteringApiSecret
        | AccountManagerState::EnteringAccessToken
        | AccountManagerState::EnteringAccessTokenSecret
        | AccountManagerState::EnteringBearerToken => {
            let prompt = match app.state.account_manager.state {
                AccountManagerState::EnteringAccountName => "Account Name",
                AccountManagerState::EnteringApiKey => "API Key",
                AccountManagerState::EnteringApiSecret => "API Secret",
                AccountManagerState::EnteringAccessToken => "Access Token",
                AccountManagerState::EnteringAccessTokenSecret => "Access Token Secret",
                AccountManagerState::EnteringBearerToken => "Bearer Token",
                _ => "",
            };

            let content = vec![
                Line::from(format!("Enter {}:", prompt)),
                Line::from(""),
                Line::from(app.state.account_manager.input_buffer.clone()),
                Line::from(""),
                Line::from("Press Enter to confirm, Escape to cancel"),
            ];

            let widget = Paragraph::new(content).block(block.title(prompt));
            f.render_widget(widget, area);
        }

        _ => {
            let widget = Paragraph::new("Loading...").block(block);
            f.render_widget(widget, area);
        }
    }
}

fn draw_recent_tab(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    let block = Block::default()
        .title(format!("Recent Tweets ({})", app.state.recent_tweets.len()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    if app.state.recent_tweets.is_empty() {
        let message = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("No recent tweets. "),
                Span::styled("Switch to the Tweet tab to post!", Style::default().fg(Color::Yellow)),
            ]),
        ];
        let widget = Paragraph::new(message).block(block).style(Style::default().fg(Color::DarkGray));
        f.render_widget(widget, chunks[0]);
    } else {
        // Create list items from tweets
        let items: Vec<ListItem> = app.state.recent_tweets
            .iter()
            .enumerate()
            .map(|(i, tweet)| {
                let time_ago = format_time_ago(&tweet.timestamp);
                let engagement = format!(
                    "❤ {} 🔁 {} 💬 {}",
                    tweet.likes, tweet.retweets, tweet.replies
                );

                let content = vec![
                    Line::from(vec![
                        Span::styled(format!("@{}", tweet.author), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::raw(" · "),
                        Span::styled(time_ago, Style::default().fg(Color::DarkGray)),
                    ]),
                    Line::from(tweet.text.clone()),
                    Line::from(vec![
                        Span::styled(engagement, Style::default().fg(Color::Gray)),
                    ]),
                    Line::from(""), // Empty line for spacing
                ];

                let style = if i == app.state.recent_tweets_scroll_index {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                };

                ListItem::new(content).style(style)
            })
            .collect();

        let tweets_list = List::new(items)
            .block(block)
            .highlight_style(Style::default().bg(Color::DarkGray));

        f.render_widget(tweets_list, chunks[0]);
    }

    // Navigation help
    let nav_help = Line::from(vec![
        Span::raw("Use "),
        Span::styled("↑/↓", Style::default().fg(Color::Yellow)),
        Span::raw(" to scroll | "),
        Span::styled("r", Style::default().fg(Color::Yellow)),
        Span::raw(" to refresh | "),
        Span::styled("Enter", Style::default().fg(Color::Yellow)),
        Span::raw(" to view details"),
    ]);

    let nav_widget = Paragraph::new(nav_help)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

    f.render_widget(nav_widget, chunks[1]);
}

/// Format timestamp as relative time (e.g., "2m ago", "1h ago", "3d ago")
fn format_time_ago(timestamp: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(*timestamp);

    if duration.num_seconds() < 60 {
        format!("{}s", duration.num_seconds())
    } else if duration.num_minutes() < 60 {
        format!("{}m", duration.num_minutes())
    } else if duration.num_hours() < 24 {
        format!("{}h", duration.num_hours())
    } else if duration.num_days() < 30 {
        format!("{}d", duration.num_days())
    } else {
        timestamp.format("%b %d").to_string()
    }
}

fn draw_social_settings_tab(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title("Social Settings")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));

    let auto_post_value = if app.state.social_settings.auto_post { "Enabled" } else { "Disabled" };
    let retry_value = format!("{} attempts", app.state.social_settings.retry_attempts);
    let char_limit_value = format!("{} characters", app.state.social_settings.character_limit);
    let theme_value =
        if app.state.social_settings.consistent_theme { "Consistent" } else { "Independent" };

    let settings = [
        ("Auto-post", auto_post_value),
        ("Retry on failure", retry_value.as_str()),
        ("Character Limit", char_limit_value.as_str()),
        ("Theme", theme_value),
    ];

    let mut items = Vec::new();

    for (i, (name, value)) in settings.iter().enumerate() {
        let style = if i == app.state.settings_manager.selected_index {
            Style::default().bg(Color::Blue).fg(Color::White)
        } else {
            Style::default().fg(Color::White)
        };

        let item_text = format!("{}: {}", name, value);
        items.push(ListItem::new(item_text).style(style));
    }

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

// Enhanced functions for real X/Twitter data integration

/// Enhanced social tab with real X/Twitter data
fn draw_tab_social_enhanced(f: &mut Frame, app: &mut App, area: Rect) {
    use crate::tui::visual_components::{ LoadingSpinner};

    // Get system connector
    let system_connector = match app.system_connector.as_ref() {
        Some(conn) => conn,
        None => {
            // Fall back to legacy
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)])
                .split(area);
            draw_sub_tab_navigation(f, &app.state.social_tabs, main_chunks[0]);
            match app.state.social_tabs.current_index {
                0 => draw_tweet_tab(f, app, main_chunks[1]),
                1 => draw_accounts_tab(f, app, main_chunks[1]),
                2 => draw_recent_tab(f, app, main_chunks[1]),
                3 => draw_social_settings_tab(f, app, main_chunks[1]),
                _ => draw_tweet_tab(f, app, main_chunks[1]),
            };
            return;
        }
    };

    // Check if X client is connected
    let x_data = match tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(
            system_connector.get_x_status()
        )
    }) {
        Ok(data) => data,
        Err(_) => {
            // Fall back to legacy if no X data available
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)])
                .split(area);
            draw_sub_tab_navigation(f, &app.state.social_tabs, main_chunks[0]);

            // Show connection status
            let loading = LoadingSpinner::new("Connecting to X/Twitter...".to_string());
            loading.render(f, main_chunks[1]);
            return;
        }
    };

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    draw_sub_tab_navigation(f, &app.state.social_tabs, main_chunks[0]);

    match app.state.social_tabs.current_index {
        0 => draw_tweet_tab_enhanced(f, app, main_chunks[1], &x_data),
        1 => draw_accounts_tab_enhanced(f, app, main_chunks[1], &x_data),
        2 => draw_recent_tab_enhanced(f, app, main_chunks[1], &x_data),
        3 => draw_social_settings_tab(f, app, main_chunks[1]),
        _ => draw_tweet_tab_enhanced(f, app, main_chunks[1], &x_data),
    };
}

/// Enhanced tweet tab with real-time status
fn draw_tweet_tab_enhanced(f: &mut Frame, app: &mut App, area: Rect, x_data: &crate::tui::connectors::system_connector::XTwitterData) {
    use crate::tui::visual_components::{MetricCard, TrendDirection, PulsingStatusIndicator};

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),  // Status cards
            Constraint::Min(5),     // Tweet input
            Constraint::Length(3),  // Actions
            Constraint::Min(0),     // Status
        ])
        .split(area);

    // Top status cards
    let status_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
        ])
        .split(chunks[0]);

    // Followers card
    let followers_card = MetricCard {
        title: "Followers".to_string(),
        value: format_number(x_data.account_stats.followers),
        subtitle: format!("+{}", x_data.account_stats.followers_change_24h),
        trend: if x_data.account_stats.followers_change_24h > 0 {
            TrendDirection::Up
        } else {
            TrendDirection::Stable
        },
        border_color: Color::Cyan,
    };
    followers_card.render(f, status_chunks[0]);

    // Engagement rate card
    let engagement_card = MetricCard {
        title: "Engagement".to_string(),
        value: format!("{:.1}%", x_data.account_stats.engagement_rate * 100.0),
        subtitle: "Last 7 days".to_string(),
        trend: if x_data.account_stats.engagement_rate > 0.02 {
            TrendDirection::Up
        } else {
            TrendDirection::Down
        },
        border_color: Color::Green,
    };
    engagement_card.render(f, status_chunks[1]);

    // Posts today card
    let posts_card = MetricCard {
        title: "Posts Today".to_string(),
        value: x_data.account_stats.posts_today.to_string(),
        subtitle: format!("of {} limit", x_data.rate_limits.daily_tweet_limit),
        trend: TrendDirection::Stable,
        border_color: Color::Blue,
    };
    posts_card.render(f, status_chunks[2]);

    // Rate limit card
    let rate_remaining = x_data.rate_limits.tweets_remaining;
    let rate_card = MetricCard {
        title: "API Rate".to_string(),
        value: format!("{}/{}", rate_remaining, x_data.rate_limits.tweets_per_15min),
        subtitle: "Resets in 15m".to_string(),
        trend: if rate_remaining < 10 {
            TrendDirection::Down
        } else {
            TrendDirection::Stable
        },
        border_color: if rate_remaining < 10 { Color::Red } else { Color::Yellow },
    };
    rate_card.render(f, status_chunks[3]);

    // Enhanced tweet input area
    let tweet_box = Block::default()
        .title(format!("Tweet Input ({}/280 chars) | @{}",
            app.state.tweet_input.len(),
            x_data.authenticated_user.as_ref().map(|u| u.username.clone()).unwrap_or_else(|| "not_connected".to_string())
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(
            if app.state.tweet_input.len() > 280 { Color::Red } else { Color::White }
        ));

    let input_text = if app.state.tweet_input.is_empty() {
        "What's happening?"
    } else {
        &app.state.tweet_input
    };

    let style = if app.state.tweet_input.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input_widget = Paragraph::new(input_text)
        .block(tweet_box)
        .style(style)
        .wrap(Wrap { trim: false });

    f.render_widget(input_widget, chunks[1]);

    // Actions with real status
    let actions = if x_data.is_connected {
        vec![
            Line::from(vec![
                Span::styled("[Enter]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(" Send Tweet | "),
                Span::styled("[Ctrl+S]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" Schedule | "),
                Span::styled("[Ctrl+T]", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                Span::raw(" Add Thread | "),
                Span::styled("[Esc]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Clear"),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::styled("⚠️ Not connected to X/Twitter", Style::default().fg(Color::Red)),
                Span::raw(" | "),
                Span::styled("[Tab]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" Go to Accounts tab to connect"),
            ]),
        ]
    };

    let actions_widget = Paragraph::new(actions)
        .block(Block::default().borders(Borders::ALL))
        .alignment(ratatui::layout::Alignment::Center);

    f.render_widget(actions_widget, chunks[2]);

    // Connection status
    let status = PulsingStatusIndicator {
        text: if x_data.is_connected {
            format!("✅ Connected as @{} | {} mentions | {} DMs",
                x_data.authenticated_user.as_ref().map(|u| u.username.clone()).unwrap_or_default(),
                x_data.unread_mentions,
                x_data.unread_dms
            )
        } else {
            "❌ Not connected to X/Twitter".to_string()
        },
        color: if x_data.is_connected { Color::Green } else { Color::Red },
        pulse_speed: 2.0,
    };
    status.render(f, chunks[3]);
}

/// Enhanced accounts tab with real account data
fn draw_accounts_tab_enhanced(f: &mut Frame, _app: &mut App, area: Rect, x_data: &crate::tui::connectors::system_connector::XTwitterData) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Left side - Account list
    let mut items = Vec::new();

    // Show authenticated account
    if let Some(user) = &x_data.authenticated_user {
        items.push(ListItem::new(vec![
            Line::from(vec![
                Span::raw("🟢 "),
                Span::styled(
                    format!("@{}", user.username),
                    Style::default().fg(Color::Green).add_modifier(ratatui::style::Modifier::BOLD)
                ),
                Span::raw(" (Active)"),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(user.display_name.clone(), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    format!("Username: @{}", user.username),
                    Style::default().fg(Color::DarkGray)
                ),
            ]),
        ]));
    } else {
        items.push(ListItem::new(vec![
            Line::from(vec![
                Span::raw("🔴 "),
                Span::styled(
                    "No account connected",
                    Style::default().fg(Color::Red)
                ),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    "Press [N] to add a new account",
                    Style::default().fg(Color::Yellow)
                ),
            ]),
        ]));
    }

    let accounts_list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("X/Twitter Accounts")
            .border_style(Style::default().fg(Color::Cyan)))
        .style(Style::default().fg(Color::White));

    f.render_widget(accounts_list, chunks[0]);

    // Right side - Account details
    if x_data.is_connected {
        let details = vec![
            Line::from(vec![Span::styled(
                "📊 Account Statistics",
                Style::default().fg(Color::Cyan).add_modifier(ratatui::style::Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Followers: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format_number(x_data.account_stats.followers),
                    Style::default().fg(Color::White).add_modifier(ratatui::style::Modifier::BOLD)
                ),
            ]),
            Line::from(vec![
                Span::styled("Following: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format_number(x_data.account_stats.following),
                    Style::default().fg(Color::White)
                ),
            ]),
            Line::from(vec![
                Span::styled("Total Posts: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format_number(x_data.account_stats.total_posts),
                    Style::default().fg(Color::White)
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Engagement Rate: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{:.2}%", x_data.account_stats.engagement_rate * 100.0),
                    Style::default().fg(
                        if x_data.account_stats.engagement_rate > 0.02 {
                            Color::Green
                        } else {
                            Color::Red
                        }
                    )
                ),
            ]),
            Line::from(vec![
                Span::styled("Posts Today: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{}/{}", x_data.account_stats.posts_today, x_data.rate_limits.daily_tweet_limit),
                    Style::default().fg(Color::White)
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "🔑 Actions",
                Style::default().fg(Color::Green).add_modifier(ratatui::style::Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled("[R]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" Refresh Stats"),
            ]),
            Line::from(vec![
                Span::styled("[D]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Disconnect Account"),
            ]),
        ];

        let details_widget = Paragraph::new(details)
            .block(Block::default()
                .borders(Borders::ALL)
                .title("Account Details")
                .border_style(Style::default().fg(Color::Green)));

        f.render_widget(details_widget, chunks[1]);
    } else {
        let setup_guide = vec![
            Line::from(vec![Span::styled(
                "🔧 Setup Guide",
                Style::default().fg(Color::Yellow).add_modifier(ratatui::style::Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from("1. Press [N] to add account"),
            Line::from("2. Enter your API credentials"),
            Line::from("3. Test the connection"),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Need API keys?",
                Style::default().fg(Color::Cyan),
            )]),
            Line::from("Visit developer.twitter.com"),
        ];

        let guide_widget = Paragraph::new(setup_guide)
            .block(Block::default()
                .borders(Borders::ALL)
                .title("Getting Started")
                .border_style(Style::default().fg(Color::Yellow)));

        f.render_widget(guide_widget, chunks[1]);
    }
}

/// Enhanced recent tweets tab with real timeline data
fn draw_recent_tab_enhanced(f: &mut Frame, _app: &App, area: Rect, x_data: &crate::tui::connectors::system_connector::XTwitterData) {
    use crate::tui::visual_components::{AnimatedList, AnimatedListItem};

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    if x_data.recent_posts.is_empty() {
        let message = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("No recent tweets. "),
                Span::styled(
                    if x_data.is_connected {
                        "Switch to the Tweet tab to post!"
                    } else {
                        "Connect your X/Twitter account first!"
                    },
                    Style::default().fg(Color::Yellow)
                ),
            ]),
        ];

        let widget = Paragraph::new(message)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(format!("Recent Tweets (0)"))
                .border_style(Style::default().fg(Color::Cyan)))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(ratatui::layout::Alignment::Center);

        f.render_widget(widget, chunks[0]);
    } else {
        // Create animated list items from real tweets
        let mut items = Vec::new();

        for (i, post) in x_data.recent_posts.iter().enumerate().take(20) {
            let time_ago = format_time_ago(&post.timestamp);
            let engagement = format!(
                "❤ {} 🔁 {} 💬 {}",
                post.likes, post.retweets, post.replies
            );

            let color = Color::White; // TwitterPost doesn't have is_reply/is_retweet fields

            items.push(AnimatedListItem {
                content: Line::from(vec![
                    Span::styled(
                        format!("@{}", "loki_ai"), // TwitterPost doesn't have author_username field
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    ),
                    Span::raw(" · "),
                    Span::styled(time_ago, Style::default().fg(Color::DarkGray)),
                    Span::raw(" "),
                    Span::styled(engagement, Style::default().fg(Color::Gray)),
                ]),
                highlight_color: color,
                animation_offset: i as f32 * 0.1,
            });

            // Add tweet text as separate item for better readability
            items.push(AnimatedListItem {
                content: Line::from(vec![
                    Span::raw("  "),
                    Span::styled(post.text.clone(), Style::default().fg(color)),
                ]),
                highlight_color: color,
                animation_offset: i as f32 * 0.1 + 0.05,
            });

            // Add spacing
            items.push(AnimatedListItem {
                content: Line::from(""),
                highlight_color: Color::Black,
                animation_offset: 0.0,
            });
        }

        let animated_list = AnimatedList {
            items,
            title: format!("Recent Tweets ({})", x_data.recent_posts.len()),
            border_color: Color::Cyan,
            animation_speed: 0.5,
        };

        animated_list.render(f, chunks[0]);
    }

    // Status bar
    let status_text = if x_data.is_connected {
        format!(
            "📊 {} new mentions | {} new followers | Last refresh: {}",
            x_data.unread_mentions,
            x_data.account_stats.followers_change_24h,
            x_data.last_refresh.format("%H:%M:%S")
        )
    } else {
        "❌ Not connected to X/Twitter".to_string()
    };

    let status_widget = Paragraph::new(status_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(
                if x_data.is_connected { Color::Green } else { Color::Red }
            )))
        .alignment(ratatui::layout::Alignment::Center);

    f.render_widget(status_widget, chunks[1]);
}

/// Format large numbers with K/M suffixes
fn format_number(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f32 / 1_000.0)
    } else {
        n.to_string()
    }
}
