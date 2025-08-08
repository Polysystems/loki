use reqwest::Client;
use serde_json::json;
use crate::config::ApiKeysConfig;
use crate::tui::App;

impl App {
    pub async fn post_tweet(&mut self, message: String) -> Result<(), Box<dyn std::error::Error>> {
        let apiconfig = ApiKeysConfig::from_env().expect("failed to load api keys").x_twitter;

        let Some(x) = apiconfig else {
            let error_msg = "Twitter API keys not configured.";
            self.state.tweet_status = Some(error_msg.to_string());
            return Err(error_msg.into());
        };

        let client = Client::new();
        let url = "https://api.twitter.com/2/tweets";

        let res = client
            .post(url)
            .bearer_auth(x.bearer_token)
            .json(&json!({
                "text": message
            }))
            .send()
            .await?;

        let status = res.status();
        let body = res.text().await?;

        if status.is_success() {
            self.state.tweet_status = Some("Tweet posted successfully!".to_string());
            Ok(())
        } else {
            let err_msg = format!("Twitter API error ({}): {}", status, body);
            self.state.tweet_status = Some(err_msg.clone());
            Err(err_msg.into())
        }
    }
}
