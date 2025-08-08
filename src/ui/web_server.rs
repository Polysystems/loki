// Web Server - Production-ready HTTP/WebSocket server for the UI
//
// This provides a comprehensive web interface with real-time communication
// using axum framework with proper routing, middleware, and WebSocket support.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, Query, State, WebSocketUpgrade};
use axum::extract::ws::Utf8Bytes;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};

use super::api::{ResponseChunk, SessionStatus};
use super::{SessionId, SetupId, SimplifiedAPI, UserId};
use crate::models::TaskResponse;

/// Production-ready Web UI Server with HTTP and WebSocket support
pub struct WebUIServer {
    api: Arc<SimplifiedAPI>,
    port: u16,
}

impl WebUIServer {
    pub fn new(api: Arc<SimplifiedAPI>, port: u16) -> Self {
        Self { api, port }
    }

    /// Start the production web server with comprehensive functionality
    pub async fn start(&self) -> Result<()> {
        self.run_server().await
    }

    /// Create the application with all routes and middleware
    async fn create_app(&self) -> Router {
        let api_state = AppState { api: self.api.clone() };

        // API routes
        let api_routes = Router::new()
            .route("/health", get(health_check))
            .route("/setups", get(get_setups))
            .route("/setups/category/:category", get(get_setups_by_category))
            .route("/sessions", post(create_session))
            .route("/sessions/:session_id/messages", post(send_message))
            .route("/sessions/:session_id/status", get(get_session_status))
            .route("/sessions/:session_id/stream", get(websocket_handler))
            .route("/sessions/:session_id/stop", post(stop_session))
            .route("/users/:user_id/favorites", get(get_user_favorites))
            .route("/users/:user_id/favorites", post(save_favorite_setup))
            .route("/users/:user_id/recents", get(get_user_recents))
            .route("/users/:user_id/recommendations", get(get_recommendations))
            .route(
                "/users/:user_id/sessions/:session_id/optimize",
                get(get_optimization_suggestions),
            );

        // Main application with middleware
        Router::new()
            .route("/", get(serve_ui))
            .nest("/api", api_routes)
            .nest_service("/static", ServeDir::new("static"))
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive())
                    .layer(TimeoutLayer::new(Duration::from_secs(30)))
                    .layer(DefaultBodyLimit::max(1024 * 1024)), // 1MB max body size
            )
            .with_state(api_state)
    }

    /// Run the HTTP server
    async fn run_server(&self) -> Result<()> {
        let app = self.create_app().await;
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], self.port));

        info!("🌐 Starting HTTP server on http://{}", addr);

        // Use tokio to bind and serve
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        info!("✅ Server listening on http://{}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// Application state shared across all handlers
#[derive(Clone)]
struct AppState {
    api: Arc<SimplifiedAPI>,
}

// HTTP Route Handlers

/// Serve the main UI (HTML page)
async fn serve_ui() -> impl IntoResponse {
    Html(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>Loki AI System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        .endpoint { margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #007acc; }
        .method { font-weight: bold; color: #007acc; }
        code { background: #eee; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Loki AI System - Web Interface</h1>
        <p>Welcome to the Loki AI System web interface. The server is running and ready to handle API requests.</p>

        <h2>📋 Available API Endpoints</h2>
        <div class="endpoint">
            <span class="method">GET</span> <code>/api/health</code> - System health check
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <code>/api/setups</code> - List all available setups
        </div>
        <div class="endpoint">
            <span class="method">POST</span> <code>/api/sessions</code> - Create a new session
        </div>
        <div class="endpoint">
            <span class="method">POST</span> <code>/api/sessions/:id/messages</code> - Send message to session
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <code>/api/sessions/:id/stream</code> - WebSocket streaming
        </div>

        <h2>🔧 WebSocket Connection</h2>
        <p>Connect to <code>ws://localhost:PORT/api/sessions/SESSION_ID/stream</code> for real-time communication.</p>

        <h2>📊 System Status</h2>
        <p>✅ Production web server active</p>
        <p>✅ HTTP/WebSocket support enabled</p>
        <p>✅ CORS and security middleware configured</p>
    </div>
</body>
</html>
    "#,
    )
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string()),
    })
}

/// Get all available setups
async fn get_setups(
    State(state): State<AppState>,
) -> Result<Json<Vec<super::SetupTemplate>>, AppError> {
    let setups = state.api.get_available_setups().await?;
    Ok(Json(setups))
}

/// Get setups by category
async fn get_setups_by_category(
    Path(category): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<super::SetupTemplate>>, AppError> {
    let setups = state.api.get_setups_by_category(&category).await?;
    Ok(Json(setups))
}

/// Create a new session
async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, AppError> {
    debug!("Creating session for setup: {} user: {}", request.setup_id, request.user_id);

    let session_id =
        state.api.launch_setup(&SetupId(request.setup_id), &UserId(request.user_id)).await?;

    Ok(Json(CreateSessionResponse { session_id: session_id.0, status: "created".to_string() }))
}

/// Send a message to a session
#[axum::debug_handler]
async fn send_message(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<SendMessageRequest>,
) -> Result<Json<TaskResponse>, AppError> {
    debug!("Sending message to session {}: {}", session_id, request.message);

    let response = state.api.send_message(&SessionId(session_id), &request.message).await?;
    Ok(Json(response))
}

/// Get session status
async fn get_session_status(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<SessionStatus>, AppError> {
    let status = state.api.get_session_status(&SessionId(session_id)).await?;
    Ok(Json(status))
}

/// Stop a session
async fn stop_session(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<StopSessionResponse>, AppError> {
    state.api.stop_session(&SessionId(session_id)).await?;
    Ok(Json(StopSessionResponse { status: "stopped".to_string() }))
}

/// Get user's favorite setups
async fn get_user_favorites(
    Path(user_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<super::SetupTemplate>>, AppError> {
    let favorites = state.api.get_user_favorites(&UserId(user_id)).await?;
    Ok(Json(favorites))
}

/// Save a setup as favorite
async fn save_favorite_setup(
    Path(user_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<SaveFavoriteRequest>,
) -> Result<Json<SaveFavoriteResponse>, AppError> {
    state.api.save_favorite_setup(&UserId(user_id), &SetupId(request.setup_id)).await?;
    Ok(Json(SaveFavoriteResponse { status: "saved".to_string() }))
}

/// Get user's recent setups
async fn get_user_recents(
    Path(user_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<super::SetupTemplate>>, AppError> {
    let recents = state.api.get_user_recents(&UserId(user_id)).await?;
    Ok(Json(recents))
}

/// Get personalized recommendations
async fn get_recommendations(
    Path(user_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<super::Recommendation>>, AppError> {
    let recommendations = state.api.get_recommended_setups(&UserId(user_id)).await?;
    Ok(Json(recommendations))
}

/// Get optimization suggestions for a session
async fn get_optimization_suggestions(
    Path((_user_id, session_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<super::OptimizationSuggestion>>, AppError> {
    let suggestions = state.api.get_optimization_suggestions(&SessionId(session_id)).await?;
    Ok(Json(suggestions))
}

// WebSocket Handler

/// WebSocket upgrade handler for real-time streaming
async fn websocket_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> Response {
    let user_agent = params.get("user_agent").cloned().unwrap_or_default();

    ws.on_upgrade(move |socket| handle_websocket(socket, session_id, state, user_agent))
}

/// Handle WebSocket connection for real-time communication
async fn handle_websocket(
    mut socket: axum::extract::ws::WebSocket,
    session_id: String,
    state: AppState,
    user_agent: String,
) {
    info!("🔌 WebSocket connected for session: {} ({})", session_id, user_agent);

    use axum::extract::ws::Message;

    loop {
        tokio::select! {
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        debug!("📨 WebSocket message received: {}", text);

                        // Parse the message
                        if let Ok(ws_request) = serde_json::from_str::<WebSocketMessage>(&text) {
                            match ws_request.message_type.as_str() {
                                "send_message" => {
                                    if let Some(content) = ws_request.content {
                                        // Process the message and stream response
                                        match state.api.send_message(&SessionId(session_id.clone()), &content).await {
                                            Ok(response) => {
                                                let chunk = ResponseChunk {
                                                    content: response.content,
                                                    is_final: true,
                                                    model_used: response.model_used.model_id(),
                                                    generation_time_ms: response.generation_time_ms,
                                                };

                                                let ws_response = WebSocketResponse {
                                                    response_type: "message_response".to_string(),
                                                    chunk: Some(chunk),
                                                    status: None,
                                                    error: None,
                                                };

                                                if let Ok(json) = serde_json::to_string(&ws_response) {
                                                    if socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err() {
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                let error_response = WebSocketResponse {
                                                    response_type: "error".to_string(),
                                                    chunk: None,
                                                    status: None,
                                                    error: Some(e.to_string()),
                                                };

                                                if let Ok(json) = serde_json::to_string(&error_response) {
                                                    if socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err() {
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                "get_status" => {
                                    // Send session status
                                    match state.api.get_session_status(&SessionId(session_id.clone())).await {
                                        Ok(status) => {
                                            let ws_response = WebSocketResponse {
                                                response_type: "status_update".to_string(),
                                                chunk: None,
                                                status: Some(status),
                                                error: None,
                                            };

                                            if let Ok(json) = serde_json::to_string(&ws_response) {
                                                if socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err() {
                                                    break;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to get session status: {}", e);
                                        }
                                    }
                                }
                                _ => {
                                    warn!("Unknown WebSocket message type: {}", ws_request.message_type);
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        info!("🔌 WebSocket closed for session: {}", session_id);
                        break;
                    }
                    Some(Err(e)) => {
                        error!("WebSocket error for session {}: {}", session_id, e);
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }

            // Send periodic status updates
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                match state.api.get_session_status(&SessionId(session_id.clone())).await {
                    Ok(status) => {
                        let ws_response = WebSocketResponse {
                            response_type: "status_update".to_string(),
                            chunk: None,
                            status: Some(status),
                            error: None,
                        };

                        if let Ok(json) = serde_json::to_string(&ws_response) {
                            if socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        debug!("Failed to get periodic status for session {}: {}", session_id, e);
                    }
                }
            }
        }
    }

    info!("🔌 WebSocket disconnected for session: {}", session_id);
}

// Request/Response Types

#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    setup_id: String,
    user_id: String,
}

#[derive(Debug, Serialize)]
struct CreateSessionResponse {
    session_id: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct SendMessageRequest {
    message: String,
}

#[derive(Debug, Serialize)]
struct StopSessionResponse {
    status: String,
}

#[derive(Debug, Deserialize)]
struct SaveFavoriteRequest {
    setup_id: String,
}

#[derive(Debug, Serialize)]
struct SaveFavoriteResponse {
    status: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
    version: String,
}

#[derive(Debug, Deserialize)]
struct WebSocketMessage {
    message_type: String,
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct WebSocketResponse {
    response_type: String,
    chunk: Option<ResponseChunk>,
    status: Option<SessionStatus>,
    error: Option<String>,
}

// Error Handling

#[derive(Debug)]
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("API error: {}", self.0);

        let error_response = serde_json::json!({
            "error": self.0.to_string(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
