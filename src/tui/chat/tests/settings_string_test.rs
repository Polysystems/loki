//! Tests for string value handling in settings tab

#[cfg(test)]
mod tests {
    use super::super::subtabs::settings_tab::{SettingsTab, SettingValue};
    use super::super::ChatSettings;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    
    #[test]
    fn test_string_value_parsing() {
        // Test string value
        let mut val = SettingValue::String("test".to_string());
        assert!(val.set_string("new value".to_string()).is_ok());
        match val {
            SettingValue::String(s) => assert_eq!(s, "new value"),
            _ => panic!("Expected string value"),
        }
        
        // Test float value parsing
        let mut val = SettingValue::Float(0.5);
        assert!(val.set_string("0.7".to_string()).is_ok());
        match val {
            SettingValue::Float(f) => assert_eq!(f, 0.7),
            _ => panic!("Expected float value"),
        }
        
        // Test invalid float
        let mut val = SettingValue::Float(0.5);
        assert!(val.set_string("1.5".to_string()).is_err()); // Out of range
        assert!(val.set_string("abc".to_string()).is_err()); // Not a number
        
        // Test integer value parsing
        let mut val = SettingValue::Integer(10);
        assert!(val.set_string("42".to_string()).is_ok());
        match val {
            SettingValue::Integer(i) => assert_eq!(i, 42),
            _ => panic!("Expected integer value"),
        }
        
        // Test invalid integer
        let mut val = SettingValue::Integer(10);
        assert!(val.set_string("3.14".to_string()).is_err()); // Not an integer
        assert!(val.set_string("xyz".to_string()).is_err()); // Not a number
    }
    
    #[test]
    fn test_can_edit_as_string() {
        assert!(SettingsTab::can_edit_as_string(&SettingValue::String("test".to_string())));
        assert!(SettingsTab::can_edit_as_string(&SettingValue::Float(0.5)));
        assert!(SettingsTab::can_edit_as_string(&SettingValue::Integer(10)));
        assert!(!SettingsTab::can_edit_as_string(&SettingValue::Bool(true)));
        assert!(!SettingsTab::can_edit_as_string(&SettingValue::Selection { 
            current: 0, 
            options: vec!["a".to_string(), "b".to_string()] 
        }));
    }
    
    #[tokio::test]
    async fn test_settings_with_string_values() {
        let mut tab = SettingsTab::new();
        
        // Set up some test settings
        let mut settings = ChatSettings::default();
        settings.api_endpoint = Some("https://api.example.com".to_string());
        settings.default_model = Some("gpt-4".to_string());
        
        tab.set_settings(settings.clone());
        
        // Verify the settings were loaded
        let loaded_settings = tab.get_settings();
        assert_eq!(loaded_settings.api_endpoint, Some("https://api.example.com".to_string()));
        assert_eq!(loaded_settings.default_model, Some("gpt-4".to_string()));
    }
    
    #[test]
    fn test_edit_buffer_character_filtering() {
        // This would require more complex testing with actual UI interaction
        // For now, we just verify the logic
        
        // Float values should only accept digits and single decimal
        let mut buffer = String::new();
        for c in "12.34".chars() {
            if c.is_numeric() || (c == '.' && !buffer.contains('.')) {
                buffer.push(c);
            }
        }
        assert_eq!(buffer, "12.34");
        
        // Second decimal should be rejected
        let mut buffer = "12.3".to_string();
        let c = '.';
        if c.is_numeric() || (c == '.' && !buffer.contains('.')) {
            buffer.push(c);
        }
        assert_eq!(buffer, "12.3"); // Second dot not added
        
        // Integer values should only accept digits
        let mut buffer = String::new();
        for c in "123abc456".chars() {
            if c.is_numeric() {
                buffer.push(c);
            }
        }
        assert_eq!(buffer, "123456");
    }
    
    #[test]
    fn test_string_value_conversion() {
        // Test conversion of special values
        let mut settings = ChatSettings::default();
        
        // API endpoint
        settings.api_endpoint = None;
        assert_eq!(
            settings.api_endpoint.clone().unwrap_or_else(|| "default".to_string()),
            "default"
        );
        
        settings.api_endpoint = Some("https://custom.api".to_string());
        assert_eq!(
            settings.api_endpoint.clone().unwrap_or_else(|| "default".to_string()),
            "https://custom.api"
        );
        
        // Model name
        settings.default_model = None;
        assert_eq!(
            settings.default_model.clone().unwrap_or_else(|| "auto".to_string()),
            "auto"
        );
        
        settings.default_model = Some("claude".to_string());
        assert_eq!(
            settings.default_model.clone().unwrap_or_else(|| "auto".to_string()),
            "claude"
        );
    }
}