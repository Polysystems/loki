# Final Cleanup Plan for Chat System

## Functions to Remove from chat.rs

### Main Drawing Functions (lines ~12089-13906)
- [ ] `draw_tab_chat` - Main entry point (replaced by draw_modular_chat_tab)
- [ ] `draw_chat_content` - Chat messages and input (replaced by chat_content_impl)
- [ ] `draw_models_content` - Models subtab (replaced by chat_content_impl)
- [ ] `draw_history_content` - History subtab (replaced by chat_content_impl)
- [ ] `draw_chat_settings_content` - Settings subtab (replaced by settings_impl)
- [ ] `draw_orchestration_content` - Orchestration subtab (replaced by settings_impl)
- [ ] `draw_agents_content` - Agents subtab (replaced by settings_impl)
- [ ] `draw_cli_content` - CLI subtab (replaced by settings_impl)

### Supporting Functions to Check
- [ ] `draw_chat_messages` - Helper for chat content
- [ ] `draw_chat_input_panel` - Input area rendering
- [ ] `draw_context_panel` - Context panel rendering
- [ ] `draw_tool_results_panel` - Tool results rendering
- [ ] `try_render_modular` - Bridge function (no longer needed)

## Other Cleanup Tasks
1. Remove any helper functions only used by the above
2. Update any remaining imports
3. Ensure ChatManager only contains coordination logic
4. Document the new modular structure

## Verification Steps
1. Build succeeds without errors
2. All chat functionality works in TUI
3. No dead code warnings for removed functions
4. ChatManager is under 500 lines (target)