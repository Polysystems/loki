use tokio;
use loki::tui::ui::tabs::chat::ChatManager;
use loki::cognitive::CognitiveSystem;
use loki::memory::CognitiveMemory;
use loki::models::ApiKeysConfig;

#[tokio::test]
async fn test_chat_manager_initialization() {
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();

    println!("🔬 DIAGNOSTIC TEST: ChatManager Initialization");
    
    // Test 1: Basic ChatManager creation
    println!("\n📋 TEST 1: Creating basic ChatManager");
    let mut chat_manager = ChatManager::new();
    println!("✅ Basic ChatManager created");
    
    // Test 2: Check initial state
    println!("\n📋 TEST 2: Checking initial component state");
    println!("🔍 Natural Language Orchestrator: {:?}", chat_manager.natural_language_orchestrator.is_some());
    println!("🔍 Cognitive System: {:?}", chat_manager.cognitive_system.is_some());
    println!("🔍 Memory: {:?}", chat_manager.memory.is_some());
    println!("🔍 Tool Manager: {:?}", chat_manager.tool_manager.is_some());
    println!("🔍 Streaming Enabled: {:?}", chat_manager.streaming_enabled);
    
    // Test 3: API Keys Configuration
    println!("\n📋 TEST 3: Checking API configuration");
    match ApiKeysConfig::from_env() {
        Ok(config) => {
            println!("✅ API configuration loaded successfully");
            println!("🔑 OpenAI key present: {}", config.openai_api_key.is_some());
            println!("🔑 Anthropic key present: {}", config.anthropic_api_key.is_some());
        }
        Err(e) => {
            println!("❌ API configuration failed: {}", e);
        }
    }
    
    // Test 4: Memory initialization
    println!("\n📋 TEST 4: Testing memory initialization");
    match CognitiveMemory::new_minimal().await {
        Ok(_memory) => {
            println!("✅ Memory initialization successful");
        }
        Err(e) => {
            println!("❌ Memory initialization failed: {}", e);
        }
    }
    
    // Test 5: Full system initialization
    println!("\n📋 TEST 5: Testing full system initialization");
    match chat_manager.initialize_full_system().await {
        Ok(()) => {
            println!("✅ Full system initialization successful");
            
            // Check post-initialization state
            println!("\n🔍 POST-INITIALIZATION STATE:");
            println!("🔍 Natural Language Orchestrator: {:?}", chat_manager.natural_language_orchestrator.is_some());
            println!("🔍 Cognitive System: {:?}", chat_manager.cognitive_system.is_some());
            println!("🔍 Memory: {:?}", chat_manager.memory.is_some());
            println!("🔍 Tool Manager: {:?}", chat_manager.tool_manager.is_some());
            println!("🔍 Streaming Enabled: {:?}", chat_manager.streaming_enabled);
        }
        Err(e) => {
            println!("❌ Full system initialization FAILED: {}", e);
            println!("💥 This explains why requests go directly to model!");
        }
    }
    
    // Test 6: Mock message processing
    println!("\n📋 TEST 6: Testing message processing pathway");
    let test_input = "help me setup parallel models".to_string();
    let session_id = 1;
    
    println!("🎯 Processing test message: '{}'", test_input);
    let response = chat_manager.handle_model_task(test_input, session_id).await;
    
    match response {
        loki::tui::run::AssistantResponseType::Streaming(_) => {
            println!("✅ Message routed through streaming system");
        }
        loki::tui::run::AssistantResponseType::Complete(msg) => {
            println!("⚠️ Message completed immediately: {}", msg.content);
            if msg.content.contains("I don't have the capability") {
                println!("❌ CONFIRMED: Still getting generic model responses!");
            }
        }
    }
}

#[tokio::test]
async fn test_individual_components() {
    println!("🔬 INDIVIDUAL COMPONENT TEST");
    
    // Test each component individually to isolate the failure point
    println!("\n📋 Testing API Keys");
    let api_result = ApiKeysConfig::from_env();
    println!("API Keys: {:?}", api_result.is_ok());
    
    if let Ok(config) = api_result {
        println!("\n📋 Testing Cognitive Memory");
        let memory_result = CognitiveMemory::new_minimal().await;
        println!("Memory: {:?}", memory_result.is_ok());
        
        if memory_result.is_ok() {
            println!("\n📋 Testing Cognitive System");
            let cognitive_config = loki::cognitive::CognitiveConfig::default();
            let cognitive_result = CognitiveSystem::new(config.clone(), cognitive_config).await;
            println!("Cognitive System: {:?}", cognitive_result.is_ok());
            
            if let Err(e) = cognitive_result {
                println!("❌ Cognitive System failed: {}", e);
            }
        } else if let Err(e) = memory_result {
            println!("❌ Memory failed: {}", e);
        }
    } else if let Err(e) = api_result {
        println!("❌ API Keys failed: {}", e);
    }
}