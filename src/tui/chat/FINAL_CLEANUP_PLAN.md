# Final Cleanup Plan

## Current Situation
- Old chat.rs is now 12,595 lines (down from 14,445)
- Modular system still depends on old ChatManager and types
- New modular ChatManager exists but is just a placeholder

## What We Can Clean Up Now

### 1. Remove Backup Files
- [x] chat.rs.backup_phase5
- [x] chat.rs.backup_before_draw_removal

### 2. Clean Up Unused Imports in chat.rs
- Remove imports only used by deleted draw functions
- Remove dead code that's no longer called

### 3. Move Core Types to Modular System
- [ ] ChatState → Already in modular system
- [ ] ChatManager → Keep old one for now (modular is placeholder)
- [ ] OrchestrationManager → Already in modular system
- [ ] AgentManager → Already in modular system
- [ ] ChatSettings → Move to modular
- [ ] NaturalLanguageResponse → Move to modular

### 4. Update Import Paths
- Update modular files to import from modular locations where possible
- Keep ChatManager imports from old location for now

## What Must Stay (For Now)
1. ChatManager struct and implementation
2. Core functionality that ChatManager depends on
3. Types that are used by ChatManager

## Next Phase (Future Work)
1. Implement the modular ChatManager properly
2. Migrate all ChatManager functionality
3. Then remove old chat.rs completely

## Safe Actions Now
1. Remove backup files
2. Clean up imports
3. Move simple types to modular system
4. Document what remains and why