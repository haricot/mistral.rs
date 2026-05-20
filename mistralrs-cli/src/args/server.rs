//! Server configuration options

use clap::Args;
use serde::Deserialize;

/// HTTP server configuration
#[derive(Args, Clone, Deserialize)]
pub struct ServerOptions {
    /// HTTP server port
    #[arg(short = 'p', long, default_value_t = 1234)]
    #[serde(default = "default_port")]
    pub port: u16,

    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    #[serde(default = "default_host")]
    pub host: String,

    /// Disable the built-in web UI (served at /ui by default).
    #[arg(long)]
    #[serde(default)]
    pub no_ui: bool,

    /// Default maximum tool-call rounds for the agentic loop.
    /// Per-request values from the HTTP API override this. Safety cap: 256 if unset.
    #[arg(long)]
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,

    /// URL to POST tool calls to for server-side execution.
    /// For security, this is only configurable server-side (not per-request via HTTP API).
    #[arg(long)]
    #[serde(default)]
    pub tool_dispatch_url: Option<String>,

    /// CORS allowed origins. Permissive by default.
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub cors_origins: Option<Vec<String>>,

    /// Base path prefix for Swagger UI routes.
    #[arg(long)]
    #[serde(default)]
    pub base_path: Option<String>,

    /// Whether to include Swagger/OpenAPI documentation routes.
    #[arg(long, default_value_t = true)]
    #[serde(default = "default_true")]
    pub include_swagger_routes: bool,

    /// Maximum request body limit in bytes.
    #[arg(long)]
    #[serde(default)]
    pub max_body_limit: Option<usize>,
}

#[derive(Deserialize, Default)]
pub struct ServerConfig {
    pub server: Option<ServerOptions>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: 1234,
            host: "0.0.0.0".to_string(),
            no_ui: false,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            cors_origins: None,
            base_path: None,
            include_swagger_routes: true,
            max_body_limit: None,
        }
    }
}

fn default_port() -> u16 {
    1234
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_true() -> bool {
    true
}
