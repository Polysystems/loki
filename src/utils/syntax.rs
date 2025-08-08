use std::fmt;

/// ANSI color codes for syntax highlighting
#[derive(Debug, Clone, Copy)]
pub enum Color {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl Color {
    /// Get ANSI escape code for foreground color
    pub fn fg(&self) -> &'static str {
        match self {
            Color::Black => "\x1b[30m",
            Color::Red => "\x1b[31m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Blue => "\x1b[34m",
            Color::Magenta => "\x1b[35m",
            Color::Cyan => "\x1b[36m",
            Color::White => "\x1b[37m",
            Color::BrightBlack => "\x1b[90m",
            Color::BrightRed => "\x1b[91m",
            Color::BrightGreen => "\x1b[92m",
            Color::BrightYellow => "\x1b[93m",
            Color::BrightBlue => "\x1b[94m",
            Color::BrightMagenta => "\x1b[95m",
            Color::BrightCyan => "\x1b[96m",
            Color::BrightWhite => "\x1b[97m",
        }
    }
}

/// Reset ANSI color
pub const RESET: &str = "\x1b[0m";

/// Colored text wrapper
pub struct Colored<T> {
    value: T,
    color: Color,
}

impl<T: fmt::Display> fmt::Display for Colored<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.color.fg(), self.value, RESET)
    }
}

/// Extension trait for coloring
pub trait Colorize: Sized {
    fn color(self, color: Color) -> Colored<Self> {
        Colored { value: self, color }
    }

    fn red(self) -> Colored<Self> {
        self.color(Color::Red)
    }

    fn green(self) -> Colored<Self> {
        self.color(Color::Green)
    }

    fn yellow(self) -> Colored<Self> {
        self.color(Color::Yellow)
    }

    fn blue(self) -> Colored<Self> {
        self.color(Color::Blue)
    }

    fn magenta(self) -> Colored<Self> {
        self.color(Color::Magenta)
    }

    fn cyan(self) -> Colored<Self> {
        self.color(Color::Cyan)
    }

    fn bright_black(self) -> Colored<Self> {
        self.color(Color::BrightBlack)
    }
}

impl<T: fmt::Display> Colorize for T {}

/// Simple syntax highlighter for code snippets
pub fn highlight_code(code: &str, language: &str) -> String {
    // This is a very basic implementation
    // In a real implementation, you would use tree-sitter for proper parsing

    let lines: Vec<&str> = code.lines().collect();
    let mut result = String::new();

    for line in lines {
        let highlighted = match language {
            "rust" => highlight_rust_line(line),
            "python" => highlight_python_line(line),
            _ => line.to_string(),
        };
        result.push_str(&highlighted);
        result.push('\n');
    }

    result
}

fn highlight_rust_line(line: &str) -> String {
    // Very basic Rust highlighting
    let mut result = line.to_string();

    // Keywords
    let keywords = ["fn", "let", "mut", "const", "use", "mod", "pub", "struct", "impl", "trait"];
    for keyword in &keywords {
        result = result.replace(keyword, &format!("{}", keyword.blue()));
    }

    // Comments
    if line.trim().starts_with("//") {
        return format!("{}", line.bright_black());
    }

    result
}

fn highlight_python_line(line: &str) -> String {
    // Very basic Python highlighting
    let mut result = line.to_string();

    // Keywords
    let keywords =
        ["def", "class", "import", "from", "if", "else", "elif", "for", "while", "return"];
    for keyword in &keywords {
        result = result.replace(keyword, &format!("{}", keyword.blue()));
    }

    // Comments
    if line.trim().starts_with("#") {
        return format!("{}", line.bright_black());
    }

    result
}
