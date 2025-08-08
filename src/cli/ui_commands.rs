// CLI Commands for UI Testing
//
// This provides command-line interface for testing the new UI system.

use anyhow::Result;
use clap::{Args, Subcommand};
use std::sync::Arc;
use tracing::{info, warn};

use crate::models::IntegratedModelSystem;
use crate::ui::{SimplifiedAPI, UserId, SetupId, initialize_ui_system};

#[derive(Debug, Args)]
pub struct UICommands {
    #[command(subcommand)]
    pub command: UICommand,
}

#[derive(Debug, Subcommand)]
pub enum UICommand {
    /// Start the web UI server
    Server {
        /// Port to run the server on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
    /// List available setup templates
    ListSetups,
    /// Launch a setup template
    LaunchSetup {
        /// Setup ID to launch
        setup_id: String,
        /// User ID (optional, generates random if not provided)
        #[arg(short, long)]
        user_id: Option<String>,
    },
    /// Test message sending
    TestMessage {
        /// Session ID to send message to
        session_id: String,
        /// Message to send
        message: String,
    },
    /// Show session status
    SessionStatus {
        /// Session ID to check
        session_id: String,
    },
    /// Initialize with default templates
    InitDefaults,
    /// Show user favorites
    UserFavorites {
        /// User ID to check
        user_id: String,
    },
    /// Add setup to favorites
    AddFavorite {
        /// User ID
        user_id: String,
        /// Setup ID to add
        setup_id: String,
    },
}

pub async fn handle_ui_command(
    command: UICommands,
    orchestrator: Arc<IntegratedModelSystem>,
) -> Result<()> {
    let api = initialize_ui_system(orchestrator).await?;

    match command.command {
        UICommand::Server { port } => {
            handle_server(api, port).await
        }
        UICommand::ListSetups => {
            handle_list_setups(api).await
        }
        UICommand::LaunchSetup { setup_id, user_id } => {
            handle_launch_setup(api, setup_id, user_id).await
        }
        UICommand::TestMessage { session_id, message } => {
            handle_test_message(api, session_id, message).await
        }
        UICommand::SessionStatus { session_id } => {
            handle_session_status(api, session_id).await
        }
        UICommand::InitDefaults => {
            handle_init_defaults(api).await
        }
        UICommand::UserFavorites { user_id } => {
            handle_user_favorites(api, user_id).await
        }
        UICommand::AddFavorite { user_id, setup_id } => {
            handle_add_favorite(api, user_id, setup_id).await
        }
    }
}

async fn handle_server(api: SimplifiedAPI, port: u16) -> Result<()> {
    info!("🚀 Starting Loki UI Server on port {}", port);

    let web_server = crate::ui::WebUIServer::new(Arc::new(api), port);

    println!("🌐 Loki UI Server starting...");
    println!("📂 Open http://localhost:{} in your browser", port);
    println!("🛑 Press Ctrl+C to stop\n");

    web_server.start().await?;

    Ok(())
}

async fn handle_list_setups(api: SimplifiedAPI) -> Result<()> {
    println!("📋 Available Setup Templates:\n");

    let setups = api.get_available_setups().await?;

    for setup in setups {
        println!("🎯 {} ({})", setup.name, setup.id.as_str());
        println!("   📝 {}", setup.description);
        println!("   🏷️  Category: {:?}", setup.category);
        println!("   💰 Cost: {:?} (${:.2}/hour)",
                 setup.cost_estimate.cost_class,
                 setup.cost_estimate.hourly_cost_cents / 100.0);
        println!("   ⚡ Performance: {:?} response, {:?} quality",
                 setup.performance_profile.response_time,
                 setup.performance_profile.quality_level);
        println!("   🤖 Models: {}",
                 setup.models.iter()
                     .map(|m| m.model_id.as_str())
                     .collect::<Vec<_>>()
                     .join(", "));

        if setup.is_featured {
            println!("   ⭐ Featured Setup");
        }
        println!("   ⭐ Rating: {:.1}/5.0 ({} uses)", setup.rating, setup.usage_count);
        println!();
    }

    Ok(())
}

async fn handle_launch_setup(
    api: SimplifiedAPI,
    setup_id: String,
    user_id: Option<String>,
) -> Result<()> {
    let setup_id = SetupId::from_string(setup_id);
    let user_id = user_id.map(UserId::from_string)
        .unwrap_or_else(|| UserId::new());

    println!("🚀 Launching setup {} for user {}", setup_id, user_id);

    match api.launch_setup(&setup_id, &user_id).await {
        Ok(session_id) => {
            println!("✅ Setup launched successfully!");
            println!("📋 Session ID: {}", session_id);
            println!("🔗 You can now send messages using:");
            println!("   loki ui test-message {} \"Your message here\"", session_id);
        }
        Err(e) => {
            warn!("❌ Failed to launch setup: {}", e);
            println!("💡 Try running 'loki ui list-setups' to see available setups");
        }
    }

    Ok(())
}

async fn handle_test_message(
    api: SimplifiedAPI,
    session_id: String,
    message: String,
) -> Result<()> {
    let session_id = crate::ui::SessionId::from_string(session_id);

    println!("💬 Sending message to session {}", session_id);
    println!("📝 Message: {}\n", message);

    match api.send_message(&session_id, &message).await {
        Ok(response) => {
            println!("🤖 Response from {}:", response.model_used.model_id());
            println!("📄 Content:\n{}\n", response.content);

            if let Some(time_ms) = response.generation_time_ms {
                println!("⚡ Generation time: {}ms", time_ms);
            }

            if response.quality_score > 0.0 {
                println!("🎯 Quality score: {:.2}", response.quality_score);
            }

            if let Some(cost_cents) = response.cost_cents {
                println!("💰 Cost: ${:.4}", cost_cents / 100.0);
            } else if let Some(cost_info) = response.cost_info {
                println!("💰 Cost info: {}", cost_info);
            }
        }
        Err(e) => {
            warn!("❌ Failed to send message: {}", e);
            println!("💡 Check if the session is still active with:");
            println!("   loki ui session-status {}", session_id);
        }
    }

    Ok(())
}

async fn handle_session_status(
    api: SimplifiedAPI,
    session_id: String,
) -> Result<()> {
    let session_id = crate::ui::SessionId::from_string(session_id);

    match api.get_session_status(&session_id).await {
        Ok(status) => {
            println!("📊 Session Status for {}\n", session_id);
            println!("🎯 Setup: {}", status.setup_name);
            println!("🤖 Active Models: {}", status.active_models.join(", "));
            println!("💰 Current Cost: ${:.4}", status.current_cost_cents / 100.0);
            println!("💬 Messages Sent: {}", status.message_count);
            println!("⏱️  Uptime: {} seconds", status.uptime_seconds);
            println!("❤️  Health: {:?}", status.status);
        }
        Err(e) => {
            warn!("❌ Failed to get session status: {}", e);
            println!("💡 Session may have expired or doesn't exist");
        }
    }

    Ok(())
}

async fn handle_init_defaults(api: SimplifiedAPI) -> Result<()> {
    println!("🔧 Initializing default setup templates...");

    // Templates are loaded automatically when the API is created
    let setups = api.get_available_setups().await?;

    println!("✅ Loaded {} default setup templates:", setups.len());
    for setup in setups {
        println!("   📋 {} - {}", setup.name, setup.description);
    }

    println!("\n💡 Use 'loki ui list-setups' to see all available templates");
    println!("🚀 Use 'loki ui launch-setup <id>' to start using a setup");

    Ok(())
}

async fn handle_user_favorites(api: SimplifiedAPI, user_id: String) -> Result<()> {
    let user_id = UserId::from_string(user_id);

    println!("⭐ Favorite setups for user {}\n", user_id);

    match api.get_user_favorites(&user_id).await {
        Ok(favorites) => {
            if favorites.is_empty() {
                println!("📭 No favorites yet!");
                println!("💡 Add favorites with: loki ui add-favorite <user_id> <setup_id>");
            } else {
                for setup in favorites {
                    println!("🎯 {} ({})", setup.name, setup.id.as_str());
                    println!("   📝 {}", setup.description);
                    println!("   ⭐ Rating: {:.1}/5.0", setup.rating);
                    println!();
                }
            }
        }
        Err(e) => {
            warn!("❌ Failed to get user favorites: {}", e);
        }
    }

    Ok(())
}

async fn handle_add_favorite(
    api: SimplifiedAPI,
    user_id: String,
    setup_id: String,
) -> Result<()> {
    let user_id = UserId::from_string(user_id);
    let setup_id = SetupId::from_string(setup_id);

    println!("⭐ Adding setup {} to favorites for user {}", setup_id, user_id);

    match api.save_favorite_setup(&user_id, &setup_id).await {
        Ok(()) => {
            println!("✅ Added to favorites successfully!");
            println!("📋 View all favorites with: loki ui user-favorites {}", user_id.as_str());
        }
        Err(e) => {
            warn!("❌ Failed to add favorite: {}", e);
            println!("💡 Make sure the setup ID exists. Use 'loki ui list-setups' to see available setups");
        }
    }

    Ok(())
}

