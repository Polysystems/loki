# Contributing to Loki AI

Thank you for your interest in contributing to Loki AI! This document outlines the contribution process and includes our Contributor License Agreement and Responsible Use Guidelines.

## Contributor License Agreement (CLA)

By submitting a contribution to this project, you agree to the following terms:

### 1. Grant of Rights

You grant to Polysys Aps (CVR: 45773078) and recipients of software distributed by Polysys Aps:

- A perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contributions and such derivative works.

- A perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work.

### 2. Representations

You represent that:
- You are legally entitled to grant the above licenses
- Each contribution is your original creation
- Your contribution does not violate any third-party rights
- You will notify us if you become aware of any issues regarding your contributions

### 3. Moral Rights

To the fullest extent permitted under applicable law, you waive and agree not to assert any moral rights in your contributions.

## Responsible Use Guidelines for Contributors

When contributing to Loki AI, you agree to follow these responsible use principles:

### Safety Considerations

1. **Code Safety**
   - Do not introduce vulnerabilities or backdoors
   - Consider potential misuse of features you implement
   - Document any safety considerations for your contributions
   - Test thoroughly, especially for cognitive and autonomous features

2. **AI Ethics**
   - Consider the ethical implications of AI behaviors you implement
   - Ensure transparency in decision-making algorithms
   - Respect user privacy and data protection requirements
   - Implement appropriate boundaries for autonomous operations

3. **Quality Standards**
   - Follow Rust 2025 best practices
   - Maintain cognitive complexity limits (max 25)
   - Write comprehensive tests for new features
   - Document your code thoroughly

### Development Process

1. **Before Contributing**
   - Read the architecture documentation in `docs/`
   - Check existing issues and pull requests
   - Discuss major changes in an issue first
   - Ensure your development environment matches project standards

2. **Code Standards**
   - Use `cargo fmt` before committing
   - Ensure `cargo clippy` passes with no warnings
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit Guidelines**
   - Use clear, descriptive commit messages
   - Reference issues in commit messages when applicable
   - Sign your commits with `git commit -s`

### Cognitive Module Guidelines

When working on consciousness, reasoning, or decision-making modules:

1. **Documentation Requirements**
   - Explain the cognitive model or algorithm used
   - Document potential emergent behaviors
   - Describe safety constraints implemented

2. **Testing Requirements**
   - Include unit tests for individual components
   - Add integration tests for system interactions
   - Test edge cases and failure modes
   - Consider adversarial inputs

3. **Review Process**
   - Cognitive module changes require additional review
   - Be prepared to explain design decisions
   - Address safety and ethics concerns raised during review

## Development Setup

### Prerequisites
```bash
# Required
- Rust 1.83+ (nightly-2025-06-09)
- Git
- 16GB+ RAM recommended
- 50GB+ free disk space for full build

# Optional but recommended
- CUDA 11.8+ (for GPU acceleration)
- Docker (for containerized development)
```

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/[repository]/loki.git
cd loki

# Setup development environment
./scripts/dev_session_start.sh

# Build with all features
cargo build --release --features all

# Run tests
cargo test --workspace

# Run clippy
cargo clippy -- -D warnings
```

## Submission Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the guidelines above
4. Commit your changes (`git commit -s -m 'Add amazing feature'`)
5. Push to your branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Test results and coverage information
   - Any safety or ethical considerations

## Testing Requirements

### Test Coverage
- Overall: 70% minimum
- Cognitive Modules: 85% minimum
- Safety Systems: 95% minimum
- New Features: 80% minimum

### Test Types Required
- Unit tests for individual components
- Integration tests for system interactions
- Property tests for invariants (using `proptest`)
- Performance benchmarks for critical paths

## Safety Requirements

### Critical Safety Rules

1. **Never Remove Safety Checks**
   - All autonomous operations must validate through safety layer
   - Resource limits must be enforced
   - Timeouts must be implemented

2. **Bound All Recursive Operations**
   - Maximum recursion depth must be defined
   - Stack overflow prevention required
   - Memory usage must be monitored

3. **Document Behavioral Changes**
   - Any change to cognitive behavior must be documented
   - Potential emergent behaviors must be noted
   - Safety implications must be analyzed

## Legal Compliance

By contributing, you acknowledge that:

- Loki AI is owned by Polysys Aps, a Danish company (CVR: 45773078)
- Contributions are subject to Danish and EU law
- You have read and agree to the terms in this document
- Violations of these guidelines may result in rejection of contributions

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors must:

- Be respectful and professional in all interactions
- Focus on what is best for the community and project
- Accept constructive criticism gracefully
- Respect differing viewpoints and experiences

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Publishing others' private information
- Other conduct reasonably considered inappropriate

## Questions and Support

For questions about contributing:
- Review existing documentation in `docs/`
- Check the issue tracker for similar questions
- Contact the maintainers at: thermo@polysystems.ai

## Existing Contributors

Please see the existing CONTRIBUTING.md sections below for additional detailed guidelines on:
- Architecture overview
- Specific coding standards
- PR templates
- Community channels
- First-time contributor guidance

## Acceptance

By submitting a pull request or contribution to this project, you acknowledge that you have read, understood, and agree to be bound by the terms of this Contributor License Agreement and Responsible Use Guidelines.

---

*This agreement is governed by Danish law. Any disputes shall be resolved in Danish courts, subject to applicable EU regulations.*

*Effective Date: 2025-08-07*  
*Version: 1.0*

---

# Original Contributing Guidelines

[The rest of the original CONTRIBUTING.md content follows below...]

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup-1)
- [Architecture Overview](#architecture-overview)
- [Contribution Guidelines](#contribution-guidelines)
- [Safety Requirements](#safety-requirements-1)
- [Pull Request Process](#pull-request-process)
- [Testing Standards](#testing-standards)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior
- Be respectful and constructive in discussions
- Focus on what is best for the community and project
- Show empathy towards other community members
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Trolling or insulting/derogatory remarks
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs
Before creating bug reports, please check existing issues to avoid duplicates.

**When reporting bugs, include:**
- Loki version (`loki --version`)
- Rust version (`rustc --version`)
- Operating system and version
- Detailed steps to reproduce
- Expected vs actual behavior
- Error messages and logs
- Relevant configuration

### 💡 Suggesting Enhancements

We welcome enhancement suggestions! Please provide:
- Clear use case description
- Rationale for the enhancement
- Possible implementation approach
- Impact on existing functionality
- Safety considerations

### 🔧 Areas Needing Contribution

#### High Priority
- **Safety Validators**: Enhanced validation for autonomous operations
- **Test Coverage**: Especially for cognitive modules
- **Documentation**: API docs, architecture guides, tutorials
- **Performance**: SIMD optimizations, parallel processing improvements

#### Cognitive Systems
- **Reasoning Engines**: New reasoning algorithms and approaches
- **Learning Systems**: Meta-learning and adaptation improvements
- **Creativity Modules**: Novel generation techniques
- **Social Intelligence**: Enhanced empathy and theory of mind

#### Infrastructure
- **Bridge Architecture**: Cross-component communication enhancements
- **Tool Integrations**: New external tool support
- **Monitoring**: Better observability and metrics
- **Deployment**: Container support, cloud deployments

## 🛠️ Development Setup

### Prerequisites
```bash
# Required
- Rust 1.83+ (nightly-2025-06-09)
- Git
- 16GB+ RAM
- 50GB+ free disk space

# Optional but recommended
- CUDA 11.8+ (for GPU acceleration)
- Docker (for containerized development)
- VS Code or IntelliJ with Rust plugin
```

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/loki.git
cd loki

# Setup development environment
./scripts/dev_session_start.sh

# Install git hooks
./scripts/install-hooks.sh

# Build with all features
cargo build --release --features all

# Run tests
cargo test --workspace

# Run clippy
cargo clippy -- -D warnings
```

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
cargo test --all
cargo clippy -- -D warnings
cargo fmt --check

# Run specific module tests
cargo test cognitive::reasoning --nocapture

# Benchmark changes
cargo bench --bench your_benchmark
```

## 🏗️ Architecture Overview

Understanding Loki's architecture is crucial for contributions:

### Core Layers
```
1. Consciousness Layer (src/cognitive/)
   - 100+ modules for reasoning and decision-making
   - Critical for safety - changes require extensive review

2. Bridge Layer (src/tui/bridges/)
   - EventBridge, CognitiveBridge, MemoryBridge, ToolBridge
   - Ensures component communication

3. Memory Layer (src/memory/)
   - Hierarchical storage with SIMD optimization
   - Performance-critical code

4. Tool Layer (src/tools/)
   - External integrations
   - Must maintain security boundaries

5. Safety Layer (src/safety/)
   - Validation and monitoring
   - NEVER bypass safety checks
```

### Module Guidelines

#### When modifying cognitive modules:
- Document reasoning behind changes
- Add comprehensive tests
- Consider emergent behaviors
- Implement safety bounds
- Update relevant documentation

#### When adding new features:
- Follow existing architectural patterns
- Use appropriate abstraction layers
- Implement proper error handling
- Add monitoring/metrics
- Consider performance impact

## 📏 Contribution Guidelines

### Code Style

#### Rust Standards
```rust
// Follow Rust 2025 best practices
// Use structured concurrency
use tokio::sync::mpsc;
use std::sync::Arc;

// Comprehensive error handling
use anyhow::{Result, Context};

// SIMD where appropriate
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Type safety with custom types
pub struct GoalId(String);
pub struct MemoryId(String);
```

#### Formatting
- Use `rustfmt` with project settings
- Line width: 100 characters
- Cognitive complexity limit: 25

#### Naming Conventions
- Modules: snake_case
- Types: PascalCase
- Functions: snake_case
- Constants: SCREAMING_SNAKE_CASE

### Commit Messages

Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `docs`: Documentation
- `safety`: Safety enhancements
- `cognitive`: Cognitive system changes

Examples:
```
feat(reasoning): add probabilistic inference engine
fix(memory): resolve SIMD alignment issue on ARM
safety(validation): add bounds checking for goal priorities
cognitive(consciousness): enhance meta-cognition feedback loop
```

## 🛡️ Safety Requirements

### Critical Safety Rules

1. **Never Remove Safety Checks**
   ```rust
   // ❌ NEVER DO THIS
   // self.safety_validator.validate(action).await?;
   
   // ✅ ALWAYS VALIDATE
   self.safety_validator.validate(action).await
       .context("Safety validation failed")?;
   ```

2. **Bound All Recursive Operations**
   ```rust
   // ❌ Unbounded recursion
   fn process(&self) {
       self.process();
   }
   
   // ✅ Bounded with safety
   fn process(&self, depth: u32) -> Result<()> {
       if depth > MAX_RECURSION_DEPTH {
           return Err(anyhow!("Max recursion depth exceeded"));
       }
       // ...
   }
   ```

3. **Resource Limits Required**
   ```rust
   // All autonomous operations must have:
   - Timeout mechanisms
   - Memory bounds
   - API call limits
   - CPU usage caps
   ```

### Cognitive Module Safety

When modifying cognitive modules (`src/cognitive/`):

1. **Document Behavioral Changes**
   ```rust
   /// SAFETY: This modification changes decision-making behavior
   /// Previous: Conservative threshold (0.8)
   /// New: Adaptive threshold (0.6-0.9)
   /// Impact: More dynamic but requires monitoring
   ```

2. **Add Safety Tests**
   ```rust
   #[test]
   fn test_reasoning_bounds() {
       // Test edge cases
       // Test resource limits
       // Test timeout behavior
       // Test error conditions
   }
   ```

3. **Implement Safeguards**
   ```rust
   pub struct ReasoningEngine {
       max_chain_depth: u32,      // Prevent infinite chains
       confidence_threshold: f64,  // Minimum confidence required
       timeout: Duration,         // Maximum processing time
       safety_validator: Arc<SafetyValidator>,
   }
   ```

## 🔄 Pull Request Process

### Before Submitting

1. **Run Full Test Suite**
   ```bash
   cargo test --all-features
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

2. **Check Performance**
   ```bash
   cargo bench
   # Ensure no significant regressions
   ```

3. **Update Documentation**
   - Update relevant docs in `docs/`
   - Update inline documentation
   - Update CLAUDE.md if architecture changes

### PR Requirements

Your PR must include:
- [ ] Clear description of changes
- [ ] Link to related issue (if applicable)
- [ ] Tests for new functionality
- [ ] Documentation updates
- [ ] Safety impact assessment
- [ ] Performance impact statement
- [ ] Breaking changes noted

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Documentation
- [ ] Safety enhancement

## Safety Checklist
- [ ] No safety validators removed
- [ ] Resource limits maintained
- [ ] Error handling comprehensive
- [ ] Cognitive changes documented

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks run

## Breaking Changes
List any breaking changes

## Additional Notes
Any additional context
```

### Review Process

1. **Automatic Checks**
   - CI/CD runs tests
   - Clippy analysis
   - Format verification
   - Benchmark comparison

2. **Human Review**
   - Code quality review
   - Architecture compliance
   - Safety assessment
   - Documentation check

3. **Cognitive Module Changes**
   - Require 2+ reviewers
   - Extended testing period
   - Safety team review
   - Performance validation

## 🧪 Testing Standards

### Test Coverage Requirements

- **Overall**: 70% minimum
- **Cognitive Modules**: 85% minimum
- **Safety Systems**: 95% minimum
- **New Features**: 80% minimum

### Test Types

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reasoning_chain() {
        // Test individual components
    }
}
```

#### Integration Tests
```rust
// tests/integration/cognitive_test.rs
#[tokio::test]
async fn test_cognitive_integration() {
    // Test component interactions
}
```

#### Property Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_memory_consistency(
        key in "[a-z]+",
        value in any::<String>()
    ) {
        // Property-based testing
    }
}
```

#### Performance Tests
```rust
use criterion::{black_box, criterion_group, Criterion};

fn benchmark_simd(c: &mut Criterion) {
    c.bench_function("simd_similarity", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });
}
```

## 📚 Documentation

### Documentation Requirements

All public APIs must have:
```rust
/// Brief description
/// 
/// # Arguments
/// * `param` - Description
/// 
/// # Returns
/// Description of return value
/// 
/// # Errors
/// When this function returns errors
/// 
/// # Safety
/// Any safety considerations
/// 
/// # Examples
/// ```
/// use loki::module;
/// let result = module::function();
/// ```
```

### Architecture Documentation

When changing architecture:
1. Update `docs/architecture.md`
2. Update relevant diagrams
3. Document decision rationale
4. Note performance implications
5. Describe safety measures

## 🌐 Community

### Getting Help

- **Discord**: [Join our Discord](https://discord.gg/loki-ai)
- **GitHub Discussions**: For design discussions
- **Stack Overflow**: Tag with `loki-ai`
- **Email**: thermo@polysystems.ai

### Communication Channels

- **#general** - General discussion
- **#development** - Development coordination
- **#cognitive-systems** - Cognitive architecture discussion
- **#safety** - Safety considerations
- **#help** - Get help with contributions

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project website
- Annual contributor report

## 🎯 First-Time Contributors

### Good First Issues

Look for issues labeled:
- `good-first-issue` - Simple, well-defined tasks
- `help-wanted` - Areas needing assistance
- `documentation` - Documentation improvements
- `testing` - Test additions

### Getting Started

1. **Find an Issue**
   - Browse [good first issues](https://github.com/yourusername/loki/labels/good-first-issue)
   - Comment to claim it
   - Ask questions if needed

2. **Setup Environment**
   - Follow development setup
   - Join Discord for help
   - Read relevant docs

3. **Make Changes**
   - Create feature branch
   - Make small, focused changes
   - Test thoroughly

4. **Submit PR**
   - Follow PR template
   - Be patient with review
   - Address feedback promptly

## 🔌 Plugin Development

### Creating Plugins

Loki supports both WebAssembly and native plugins:

#### WebAssembly Plugin
```rust
// plugins/example/src/lib.rs
use loki_plugin_sdk::*;

#[no_mangle]
pub extern "C" fn execute(input: &str) -> String {
    // Plugin logic
}
```

#### Native Plugin
```rust
// Follow plugin SDK documentation
use loki::plugin::{Plugin, PluginResult};

impl Plugin for MyPlugin {
    fn execute(&self, context: Context) -> PluginResult {
        // Implementation
    }
}
```

### Plugin Guidelines
- Respect safety boundaries
- Handle errors gracefully
- Document capabilities
- Include tests
- Follow security best practices

## 📮 Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search closed issues
3. Ask in Discord
4. Email thermo@polysystems.ai

---

**Thank you for contributing to Loki AI! Your contributions help advance autonomous AI systems while maintaining safety and reliability.**

Remember: Every contribution, no matter how small, is valuable. Whether it's fixing a typo, adding a test, or implementing a new cognitive algorithm, you're helping build the future of AI systems.