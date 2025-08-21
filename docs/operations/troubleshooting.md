# 🔧 Troubleshooting Guide

## Common Issues

### Installation Problems

#### "Command not found"
```bash
# Check if loki is in PATH
which loki

# Add to PATH
export PATH=$PATH:/usr/local/bin

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
```

#### "Permission denied"
```bash
# Fix executable permissions
chmod +x /usr/local/bin/loki

# Fix data directory permissions
chmod -R 755 ~/.loki
```

### API Connection Issues

#### "API key not found"
```bash
# Check environment variables
env | grep API_KEY

# Set API key
export OPENAI_API_KEY="sk-..."

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

#### "Connection timeout"
```bash
# Test API connection
loki check-apis --verbose

# Check network
curl https://api.openai.com/v1/models

# Use proxy if needed
export HTTP_PROXY=http://proxy:8080
```

### Performance Issues

#### High Memory Usage
```bash
# Check memory usage
loki diagnostic --memory

# Reduce cache size
export LOKI_CACHE_SIZE=512MB

# Limit working memory
loki config set memory.working.capacity 5
```

#### Slow Responses
```bash
# Check performance metrics
loki diagnostic --performance

# Reduce cognitive depth
loki config set cognitive.reasoning.max_depth 5

# Use faster model
loki config set llm.default_model gpt-3.5-turbo
```

### TUI Display Issues

#### Garbled Display
```bash
# Reset terminal
reset
clear

# Check terminal type
echo $TERM

# Use compatible mode
loki tui --compat
```

#### Colors Not Working
```bash
# Force color output
export LOKI_LOG_COLOR=always

# Check terminal capabilities
tput colors
```

### Memory System Issues

#### "Memory database corrupted"
```bash
# Backup current data
cp -r ~/.loki/data ~/.loki/data.backup

# Repair database
loki memory repair

# If repair fails, reset
loki memory reset --confirm
```

#### "Cannot retrieve memories"
```bash
# Check memory stats
loki memory stats

# Rebuild indexes
loki memory reindex

# Clear cache
loki memory clear-cache
```

### Tool Execution Failures

#### "Tool not found"
```bash
# List available tools
loki tool list

# Install missing tool
loki tool install <tool-name>

# Check tool configuration
loki tool config <tool-name>
```

#### "Rate limit exceeded"
```bash
# Check rate limits
loki tool limits

# Configure rate limiting
loki config set tools.rate_limits.github 1000
```

### Agent Issues

#### "Agent failed to deploy"
```bash
# Check agent status
loki agent list --all

# View agent logs
loki agent logs <agent-id>

# Restart agent
loki agent restart <agent-id>
```

#### "Agents not communicating"
```bash
# Check coordination
loki agent coordination status

# Reset communication channels
loki agent reset-channels
```

## Diagnostic Commands

### System Diagnostics
```bash
# Full diagnostic
loki diagnostic --full

# Specific checks
loki diagnostic --memory
loki diagnostic --performance
loki diagnostic --network
loki diagnostic --storage
```

### Health Checks
```bash
# Overall health
loki health

# Component health
loki health cognitive
loki health memory
loki health tools
```

### Log Analysis
```bash
# View logs
tail -f ~/.loki/logs/loki.log

# Search logs
grep ERROR ~/.loki/logs/loki.log

# Log levels
export RUST_LOG=debug  # trace, debug, info, warn, error
```

## Recovery Procedures

### Reset Configuration
```bash
# Backup current config
cp ~/.loki/config.yaml ~/.loki/config.yaml.backup

# Reset to defaults
loki config reset

# Restore from backup
cp ~/.loki/config.yaml.backup ~/.loki/config.yaml
```

### Clear Cache
```bash
# Clear all caches
loki cache clear --all

# Clear specific cache
loki cache clear --memory
loki cache clear --tools
```

### Database Recovery
```bash
# Backup database
loki backup create

# Verify integrity
loki database verify

# Compact database
loki database compact

# Restore from backup
loki restore <backup-file>
```

## Error Messages

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `TokenLimitExceeded` | Request too large | Reduce input size or increase limits |
| `ModelNotAvailable` | Model not accessible | Check API key and model name |
| `MemoryAllocationFailed` | Out of memory | Increase system memory or reduce cache |
| `ToolExecutionTimeout` | Tool took too long | Increase timeout or optimize tool |
| `CognitiveOverload` | Too complex task | Reduce reasoning depth |
| `SafetyViolation` | Unsafe content | Review safety settings |

## Platform-Specific Issues

### macOS
```bash
# Code signing issues
xattr -d com.apple.quarantine /usr/local/bin/loki

# Library issues
brew install openssl
```

### Linux
```bash
# Missing libraries
sudo apt-get install libssl-dev

# SELinux issues
sudo setenforce 0  # Temporary
```

### Windows
```powershell
# Path issues
$env:Path += ";C:\Program Files\Loki"

# Permissions
Run as Administrator
```

## Getting Help

### Self-Help Resources
1. Check this troubleshooting guide
2. Search error in documentation
3. Run diagnostic commands
4. Check logs for details

### Community Support
- Discord: https://discord.gg/eigencode
- GitHub Issues: https://github.com/polysystems/loki/issues
- Stack Overflow: Tag `loki-ai`

### Reporting Issues
Include:
- Loki version: `loki --version`
- OS and version
- Error messages
- Steps to reproduce
- Diagnostic output

---

Next: [Monitoring](monitoring.md) | [Operations](../README.md)