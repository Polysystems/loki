//! Syntax highlighting for code blocks in the chat interface
//! 
//! Provides terminal-compatible syntax highlighting for various programming
//! languages using ANSI color codes and terminal styling.

use ratatui::style::{Color, Modifier, Style};
use std::collections::HashMap;
use regex::Regex;

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Json,
    Yaml,
    Toml,
    Markdown,
    Bash,
    Sql,
    Html,
    Css,
    Cpp,
    Java,
    Ruby,
    Php,
    Swift,
    Kotlin,
    Csharp,
    Lua,
    R,
    Dart,
    Elixir,
    Haskell,
    Scala,
    Clojure,
    Zig,
    Nim,
    Crystal,
    Julia,
    Ocaml,
    Erlang,
    Fsharp,
    Racket,
    Scheme,
    CommonLisp,
    Assembly,
    Dockerfile,
    Makefile,
    Terraform,
    Nix,
    Vim,
    Emacs,
}

impl Language {
    /// Parse language from string identifier
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rust" | "rs" => Some(Language::Rust),
            "python" | "py" => Some(Language::Python),
            "javascript" | "js" => Some(Language::JavaScript),
            "typescript" | "ts" => Some(Language::TypeScript),
            "go" | "golang" => Some(Language::Go),
            "json" => Some(Language::Json),
            "yaml" | "yml" => Some(Language::Yaml),
            "toml" => Some(Language::Toml),
            "markdown" | "md" => Some(Language::Markdown),
            "bash" | "sh" | "shell" => Some(Language::Bash),
            "sql" => Some(Language::Sql),
            "html" => Some(Language::Html),
            "css" => Some(Language::Css),
            "cpp" | "c++" | "cc" => Some(Language::Cpp),
            "java" => Some(Language::Java),
            "ruby" | "rb" => Some(Language::Ruby),
            "php" => Some(Language::Php),
            "swift" => Some(Language::Swift),
            "kotlin" | "kt" => Some(Language::Kotlin),
            "csharp" | "c#" | "cs" => Some(Language::Csharp),
            "lua" => Some(Language::Lua),
            "r" => Some(Language::R),
            "dart" => Some(Language::Dart),
            "elixir" | "ex" | "exs" => Some(Language::Elixir),
            "haskell" | "hs" => Some(Language::Haskell),
            "scala" => Some(Language::Scala),
            "clojure" | "clj" => Some(Language::Clojure),
            "zig" => Some(Language::Zig),
            "nim" => Some(Language::Nim),
            "crystal" | "cr" => Some(Language::Crystal),
            "julia" | "jl" => Some(Language::Julia),
            "ocaml" | "ml" => Some(Language::Ocaml),
            "erlang" | "erl" => Some(Language::Erlang),
            "fsharp" | "f#" | "fs" => Some(Language::Fsharp),
            "racket" | "rkt" => Some(Language::Racket),
            "scheme" | "scm" => Some(Language::Scheme),
            "commonlisp" | "lisp" | "cl" => Some(Language::CommonLisp),
            "assembly" | "asm" | "s" => Some(Language::Assembly),
            "dockerfile" | "docker" => Some(Language::Dockerfile),
            "makefile" | "make" => Some(Language::Makefile),
            "terraform" | "tf" | "hcl" => Some(Language::Terraform),
            "nix" => Some(Language::Nix),
            "vim" | "vimscript" => Some(Language::Vim),
            "emacs" | "elisp" | "el" => Some(Language::Emacs),
            _ => None,
        }
    }
}

/// Token type for syntax highlighting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Keyword,
    String,
    Number,
    Comment,
    Function,
    Type,
    Operator,
    Punctuation,
    Variable,
    Constant,
    Attribute,
    Builtin,
    Error,
    Normal,
}

/// Highlighted token with style information
#[derive(Debug, Clone)]
pub struct HighlightedToken {
    pub text: String,
    pub token_type: TokenType,
    pub style: Style,
}

/// Language-specific highlighting rules
#[derive(Clone)]
struct HighlightRules {
    keywords: Vec<&'static str>,
    types: Vec<&'static str>,
    builtins: Vec<&'static str>,
    string_pattern: Regex,
    comment_pattern: Regex,
    number_pattern: Regex,
    function_pattern: Option<Regex>,
    operator_pattern: Regex,
}

/// Syntax highlighter for terminal rendering
#[derive(Clone)]
pub struct SyntaxHighlighter {
    /// Language-specific rules
    rules: HashMap<Language, HighlightRules>,
    
    /// Color scheme for token types
    colors: HashMap<TokenType, Style>,
}

impl SyntaxHighlighter {
    pub fn new() -> Self {
        let mut highlighter = Self {
            rules: HashMap::new(),
            colors: Self::default_color_scheme(),
        };
        
        highlighter.init_language_rules();
        highlighter
    }
    
    /// Create default color scheme
    fn default_color_scheme() -> HashMap<TokenType, Style> {
        let mut colors = HashMap::new();
        
        colors.insert(TokenType::Keyword, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD));
        colors.insert(TokenType::String, Style::default().fg(Color::Green));
        colors.insert(TokenType::Number, Style::default().fg(Color::Cyan));
        colors.insert(TokenType::Comment, Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC));
        colors.insert(TokenType::Function, Style::default().fg(Color::Yellow));
        colors.insert(TokenType::Type, Style::default().fg(Color::Blue));
        colors.insert(TokenType::Operator, Style::default().fg(Color::Red));
        colors.insert(TokenType::Punctuation, Style::default().fg(Color::White));
        colors.insert(TokenType::Variable, Style::default().fg(Color::White));
        colors.insert(TokenType::Constant, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        colors.insert(TokenType::Attribute, Style::default().fg(Color::Yellow));
        colors.insert(TokenType::Builtin, Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD));
        colors.insert(TokenType::Error, Style::default().fg(Color::Red).add_modifier(Modifier::UNDERLINED));
        colors.insert(TokenType::Normal, Style::default());
        
        colors
    }
    
    /// Initialize language-specific rules
    fn init_language_rules(&mut self) {
        // Rust highlighting rules
        self.rules.insert(Language::Rust, HighlightRules {
            keywords: vec![
                "fn", "let", "mut", "const", "if", "else", "match", "for", "while", "loop",
                "break", "continue", "return", "use", "mod", "pub", "struct", "enum", "trait",
                "impl", "self", "Self", "super", "crate", "async", "await", "move", "where",
                "type", "static", "extern", "unsafe", "as", "in", "ref", "dyn",
            ],
            types: vec![
                "i8", "i16", "i32", "i64", "i128", "isize",
                "u8", "u16", "u32", "u64", "u128", "usize",
                "f32", "f64", "bool", "char", "str", "String",
                "Vec", "HashMap", "HashSet", "Option", "Result",
                "Box", "Rc", "Arc", "RefCell", "Mutex", "RwLock",
            ],
            builtins: vec![
                "println!", "print!", "eprintln!", "eprint!", "format!", "vec!",
                "assert!", "assert_eq!", "assert_ne!", "debug_assert!",
                "todo!", "unimplemented!", "unreachable!", "panic!",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"(//[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\d+\.?\d*([eE][+-]?\d+)?[fui]?\d*)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%@~?:]+").unwrap(),
        });
        
        // Python highlighting rules
        self.rules.insert(Language::Python, HighlightRules {
            keywords: vec![
                "def", "class", "if", "elif", "else", "for", "while", "break", "continue",
                "return", "import", "from", "as", "try", "except", "finally", "raise",
                "with", "pass", "lambda", "yield", "global", "nonlocal", "assert",
                "async", "await", "del", "is", "not", "in", "and", "or",
            ],
            types: vec![
                "int", "float", "str", "bool", "list", "dict", "tuple", "set",
                "bytes", "bytearray", "complex", "frozenset", "memoryview",
            ],
            builtins: vec![
                "print", "len", "range", "enumerate", "zip", "map", "filter",
                "sorted", "reversed", "sum", "min", "max", "abs", "round",
                "open", "input", "type", "isinstance", "hasattr", "getattr",
            ],
            string_pattern: Regex::new(r#"(?s)(""".*?"""|'''.*?'''|"([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"#[^\n]*").unwrap(),
            number_pattern: Regex::new(r"\b(\d+\.?\d*([eE][+-]?\d+)?[jJ]?)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%@~?:]+").unwrap(),
        });
        
        // JavaScript/TypeScript highlighting rules
        let js_rules = HighlightRules {
            keywords: vec![
                "function", "var", "let", "const", "if", "else", "for", "while", "do",
                "break", "continue", "return", "switch", "case", "default", "try",
                "catch", "finally", "throw", "new", "delete", "typeof", "instanceof",
                "void", "this", "super", "class", "extends", "static", "async", "await",
                "import", "export", "from", "as", "in", "of", "yield",
            ],
            types: vec![
                "number", "string", "boolean", "object", "undefined", "null",
                "Array", "Object", "Function", "Promise", "Map", "Set", "WeakMap",
                "WeakSet", "Symbol", "BigInt", "Date", "RegExp", "Error",
            ],
            builtins: vec![
                "console", "Math", "JSON", "parseInt", "parseFloat", "isNaN",
                "isFinite", "encodeURI", "decodeURI", "setTimeout", "setInterval",
                "clearTimeout", "clearInterval", "require", "module", "exports",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|`([^`\\]|\\.)*`)"#).unwrap(),
            comment_pattern: Regex::new(r"(//[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(\d+\.?\d*([eE][+-]?\d+)?)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%?:~]+|\.{3}").unwrap(),
        };
        
        self.rules.insert(Language::JavaScript, js_rules.clone());
        self.rules.insert(Language::TypeScript, js_rules);
        
        // Go highlighting rules
        self.rules.insert(Language::Go, HighlightRules {
            keywords: vec![
                "func", "var", "const", "if", "else", "for", "range", "switch", "case",
                "default", "break", "continue", "return", "go", "chan", "select", "defer",
                "package", "import", "type", "struct", "interface", "map", "make", "new",
                "nil", "append", "copy", "delete", "len", "cap", "panic", "recover",
            ],
            types: vec![
                "bool", "byte", "rune", "int", "int8", "int16", "int32", "int64",
                "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
                "float32", "float64", "complex64", "complex128", "string",
                "error", "any", "comparable",
            ],
            builtins: vec![
                "fmt", "log", "os", "io", "time", "strings", "strconv", "errors",
                "context", "sync", "http", "json", "reflect", "runtime",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|`[^`]*`)"#).unwrap(),
            comment_pattern: Regex::new(r"(//[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(0x[0-9a-fA-F]+|0b[01]+|0o[0-7]+|\d+\.?\d*([eE][+-]?\d+)?)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%:]+|<-|:=").unwrap(),
        });
        
        // Java highlighting rules
        self.rules.insert(Language::Java, HighlightRules {
            keywords: vec![
                "public", "private", "protected", "static", "final", "abstract", "synchronized",
                "volatile", "transient", "native", "strictfp", "class", "interface", "enum",
                "extends", "implements", "import", "package", "if", "else", "for", "while",
                "do", "switch", "case", "default", "break", "continue", "return", "try",
                "catch", "finally", "throw", "throws", "new", "this", "super", "instanceof",
                "assert", "void", "const", "goto",
            ],
            types: vec![
                "boolean", "byte", "char", "short", "int", "long", "float", "double",
                "Boolean", "Byte", "Character", "Short", "Integer", "Long", "Float", "Double",
                "String", "Object", "Class", "System", "Thread", "Runnable", "Exception",
                "List", "ArrayList", "Map", "HashMap", "Set", "HashSet",
            ],
            builtins: vec![
                "System.out", "System.err", "System.in", "Math", "Arrays", "Collections",
                "Objects", "Optional", "Stream", "LocalDate", "LocalDateTime",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"(//[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(\d+\.?\d*[fFlLdD]?)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%?:]+|\.{3}|::").unwrap(),
        });
        
        // C++ highlighting rules
        self.rules.insert(Language::Cpp, HighlightRules {
            keywords: vec![
                "auto", "break", "case", "char", "const", "continue", "default", "do",
                "double", "else", "enum", "extern", "float", "for", "goto", "if",
                "inline", "int", "long", "register", "return", "short", "signed", "sizeof",
                "static", "struct", "switch", "typedef", "union", "unsigned", "void",
                "volatile", "while", "class", "private", "protected", "public", "virtual",
                "explicit", "friend", "mutable", "namespace", "new", "delete", "operator",
                "template", "this", "throw", "try", "catch", "typename", "using",
                "constexpr", "nullptr", "decltype", "static_assert", "noexcept",
            ],
            types: vec![
                "bool", "char", "int", "float", "double", "void", "wchar_t", "char16_t",
                "char32_t", "size_t", "ptrdiff_t", "nullptr_t", "int8_t", "int16_t",
                "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                "string", "vector", "map", "set", "unordered_map", "unordered_set",
                "unique_ptr", "shared_ptr", "weak_ptr", "array", "deque", "list",
            ],
            builtins: vec![
                "std", "cout", "cin", "cerr", "endl", "printf", "scanf", "malloc",
                "free", "memcpy", "memset", "strlen", "strcpy", "strcmp",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|R"\([^)]*\)")"#).unwrap(),
            comment_pattern: Regex::new(r"(//[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.?\d*([eE][+-]?\d+)?[fFlLuU]*)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!&|^%?:~]+|<<|>>|->|\.\*|::").unwrap(),
        });
        
        // SQL highlighting rules
        self.rules.insert(Language::Sql, HighlightRules {
            keywords: vec![
                "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
                "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "INDEX", "VIEW", "TRIGGER",
                "PROCEDURE", "FUNCTION", "IF", "ELSE", "CASE", "WHEN", "THEN", "END",
                "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "IS", "NULL",
                "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON", "USING",
                "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET",
                "UNION", "ALL", "DISTINCT", "AS", "WITH", "RECURSIVE",
            ],
            types: vec![
                "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "NUMERIC",
                "FLOAT", "REAL", "DOUBLE", "CHAR", "VARCHAR", "TEXT", "DATE", "TIME",
                "TIMESTAMP", "DATETIME", "BOOLEAN", "BOOL", "BLOB", "JSON", "UUID",
            ],
            builtins: vec![
                "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "FLOOR", "CEIL",
                "CONCAT", "LENGTH", "SUBSTR", "SUBSTRING", "TRIM", "UPPER", "LOWER",
                "NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
            ],
            string_pattern: Regex::new(r#"(?s)('([^'\\]|\\.)*'|"([^"\\]|\\.)*")"#).unwrap(),
            comment_pattern: Regex::new(r"(--[^\n]*|/\*[\s\S]*?\*/)").unwrap(),
            number_pattern: Regex::new(r"\b(\d+\.?\d*)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[+\-*/=<>!]+|\|\|").unwrap(),
        });
        
        // HTML highlighting rules
        self.rules.insert(Language::Html, HighlightRules {
            keywords: vec![
                "DOCTYPE", "html", "head", "body", "title", "meta", "link", "script",
                "style", "div", "span", "p", "a", "img", "ul", "ol", "li", "table",
                "tr", "td", "th", "form", "input", "button", "select", "option",
                "textarea", "label", "h1", "h2", "h3", "h4", "h5", "h6", "header",
                "footer", "nav", "main", "section", "article", "aside", "figure",
            ],
            types: vec![
                "class", "id", "src", "href", "alt", "width", "height", "style",
                "type", "name", "value", "placeholder", "required", "disabled",
                "checked", "selected", "readonly", "multiple", "min", "max",
            ],
            builtins: vec![],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"<!--[\s\S]*?-->").unwrap(),
            number_pattern: Regex::new(r"\b(\d+)\b").unwrap(),
            function_pattern: None,
            operator_pattern: Regex::new(r"[<>/=]").unwrap(),
        });
        
        // CSS highlighting rules
        self.rules.insert(Language::Css, HighlightRules {
            keywords: vec![
                "@import", "@media", "@keyframes", "@font-face", "@charset", "@namespace",
                "@supports", "@page", "!important", "from", "to",
            ],
            types: vec![
                "color", "background", "border", "margin", "padding", "font", "width",
                "height", "display", "position", "top", "right", "bottom", "left",
                "float", "clear", "text-align", "vertical-align", "line-height",
                "z-index", "overflow", "cursor", "visibility", "opacity", "transform",
                "transition", "animation", "flex", "grid", "box-shadow", "text-shadow",
            ],
            builtins: vec![
                "px", "em", "rem", "vh", "vw", "%", "pt", "cm", "mm", "in", "pc",
                "deg", "rad", "grad", "turn", "s", "ms", "Hz", "kHz",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"/\*[\s\S]*?\*/").unwrap(),
            number_pattern: Regex::new(r"\b(\d+\.?\d*)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z-]+)\s*\(").unwrap()),
            operator_pattern: Regex::new(r"[{}:;,>~+.]").unwrap(),
        });
        
        // Bash highlighting rules
        self.rules.insert(Language::Bash, HighlightRules {
            keywords: vec![
                "if", "then", "else", "elif", "fi", "case", "esac", "for", "while",
                "do", "done", "function", "return", "break", "continue", "exit",
                "source", "alias", "unalias", "set", "unset", "export", "readonly",
                "local", "declare", "typeset", "trap", "shift", "getopts",
            ],
            types: vec![],
            builtins: vec![
                "echo", "printf", "read", "cd", "pwd", "ls", "cp", "mv", "rm", "mkdir",
                "rmdir", "touch", "cat", "grep", "sed", "awk", "find", "sort", "uniq",
                "wc", "head", "tail", "cut", "paste", "tr", "chmod", "chown", "ps",
                "kill", "jobs", "fg", "bg", "wait", "sleep", "date", "time", "test",
            ],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|\$'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"#[^\n]*").unwrap(),
            number_pattern: Regex::new(r"\b(\d+)\b").unwrap(),
            function_pattern: Some(Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)").unwrap()),
            operator_pattern: Regex::new("[|&;<>()$`\\\\\"'*?#~=%\\[\\]{}!]+").unwrap(),
        });
        
        // JSON highlighting rules
        self.rules.insert(Language::Json, HighlightRules {
            keywords: vec!["true", "false", "null"],
            types: vec![],
            builtins: vec![],
            string_pattern: Regex::new(r#""([^"\\]|\\.)*""#).unwrap(),
            comment_pattern: Regex::new(r"$^").unwrap(), // JSON doesn't support comments
            number_pattern: Regex::new(r"-?\b(\d+\.?\d*([eE][+-]?\d+)?)\b").unwrap(),
            function_pattern: None,
            operator_pattern: Regex::new(r"[:,\[\]{}]").unwrap(),
        });
        
        // YAML highlighting rules
        self.rules.insert(Language::Yaml, HighlightRules {
            keywords: vec!["true", "false", "null", "yes", "no", "on", "off"],
            types: vec![],
            builtins: vec![],
            string_pattern: Regex::new(r#"(?s)("([^"\\]|\\.)*"|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"#[^\n]*").unwrap(),
            number_pattern: Regex::new(r"-?\b(\d+\.?\d*([eE][+-]?\d+)?)\b").unwrap(),
            function_pattern: None,
            operator_pattern: Regex::new(r"[-:>|&*!%@`]").unwrap(),
        });
        
        // TOML highlighting rules
        self.rules.insert(Language::Toml, HighlightRules {
            keywords: vec!["true", "false"],
            types: vec![],
            builtins: vec![],
            string_pattern: Regex::new(r#"(?s)(""".*?"""|"([^"\\]|\\.)*"|'''.*?'''|'([^'\\]|\\.)*')"#).unwrap(),
            comment_pattern: Regex::new(r"#[^\n]*").unwrap(),
            number_pattern: Regex::new(r"-?\b(\d+\.?\d*([eE][+-]?\d+)?)\b").unwrap(),
            function_pattern: None,
            operator_pattern: Regex::new(r"[=,\[\]{}.]").unwrap(),
        });
        
        // Markdown highlighting rules
        self.rules.insert(Language::Markdown, HighlightRules {
            keywords: vec![],
            types: vec![],
            builtins: vec![],
            string_pattern: Regex::new(r"`[^`]+`").unwrap(), // Inline code
            comment_pattern: Regex::new(r"<!--[\s\S]*?-->").unwrap(),
            number_pattern: Regex::new(r"\b(\d+)\b").unwrap(),
            function_pattern: None,
            operator_pattern: Regex::new(r"[*_~#>\[\]()!]").unwrap(),
        });
    }
    
    /// Highlight code with the specified language
    pub fn highlight(&self, code: &str, language: &Language) -> Vec<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut highlighted_lines = Vec::new();
        
        for line in lines {
            highlighted_lines.push(self.highlight_line(line, language));
        }
        
        highlighted_lines
    }
    
    /// Highlight a single line of code
    fn highlight_line(&self, line: &str, language: &Language) -> String {
        if let Some(rules) = self.rules.get(language) {
            let tokens = self.tokenize_line(line, rules);
            self.render_tokens(&tokens)
        } else {
            // Fallback to basic highlighting
            line.to_string()
        }
    }
    
    /// Tokenize a line of code
    fn tokenize_line(&self, line: &str, rules: &HighlightRules) -> Vec<HighlightedToken> {
        let mut tokens = Vec::new();
        let mut remaining = line;
        let mut offset = 0;
        
        while !remaining.is_empty() {
            let mut matched = false;
            
            // Check for comments
            if let Some(mat) = rules.comment_pattern.find(remaining) {
                if mat.start() == 0 {
                    tokens.push(HighlightedToken {
                        text: mat.as_str().to_string(),
                        token_type: TokenType::Comment,
                        style: self.colors[&TokenType::Comment],
                    });
                    offset += mat.len();
                    remaining = &line[offset..];
                    matched = true;
                }
            }
            
            // Check for strings
            if !matched {
                if let Some(mat) = rules.string_pattern.find(remaining) {
                    if mat.start() == 0 {
                        tokens.push(HighlightedToken {
                            text: mat.as_str().to_string(),
                            token_type: TokenType::String,
                            style: self.colors[&TokenType::String],
                        });
                        offset += mat.len();
                        remaining = &line[offset..];
                        matched = true;
                    }
                }
            }
            
            // Check for numbers
            if !matched {
                if let Some(mat) = rules.number_pattern.find(remaining) {
                    if mat.start() == 0 {
                        tokens.push(HighlightedToken {
                            text: mat.as_str().to_string(),
                            token_type: TokenType::Number,
                            style: self.colors[&TokenType::Number],
                        });
                        offset += mat.len();
                        remaining = &line[offset..];
                        matched = true;
                    }
                }
            }
            
            // Check for identifiers (keywords, types, functions)
            if !matched {
                if let Some(mat) = Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*").unwrap().find(remaining) {
                    let word = mat.as_str();
                    
                    let token_type = if rules.keywords.contains(&word) {
                        TokenType::Keyword
                    } else if rules.types.contains(&word) {
                        TokenType::Type
                    } else if rules.builtins.contains(&word) {
                        TokenType::Builtin
                    } else {
                        TokenType::Variable
                    };
                    
                    tokens.push(HighlightedToken {
                        text: word.to_string(),
                        token_type,
                        style: self.colors[&token_type],
                    });
                    
                    offset += mat.len();
                    remaining = &line[offset..];
                    matched = true;
                }
            }
            
            // Check for operators
            if !matched {
                if let Some(mat) = rules.operator_pattern.find(remaining) {
                    if mat.start() == 0 {
                        tokens.push(HighlightedToken {
                            text: mat.as_str().to_string(),
                            token_type: TokenType::Operator,
                            style: self.colors[&TokenType::Operator],
                        });
                        offset += mat.len();
                        remaining = &line[offset..];
                        matched = true;
                    }
                }
            }
            
            // Default: single character
            if !matched {
                let ch = remaining.chars().next().unwrap();
                let token_type = if ch.is_whitespace() {
                    TokenType::Normal
                } else if "()[]{},.;:".contains(ch) {
                    TokenType::Punctuation
                } else {
                    TokenType::Normal
                };
                
                tokens.push(HighlightedToken {
                    text: ch.to_string(),
                    token_type,
                    style: self.colors[&token_type],
                });
                
                offset += ch.len_utf8();
                remaining = &line[offset..];
            }
        }
        
        tokens
    }
    
    /// Render tokens to a styled string
    fn render_tokens(&self, tokens: &[HighlightedToken]) -> String {
        // For terminal rendering, we'll return the plain text
        // The actual styling will be applied by the ratatui renderer
        tokens.iter()
            .map(|token| token.text.as_str())
            .collect()
    }
    
    /// Get highlighted spans for ratatui rendering
    pub fn get_highlighted_spans(&self, code: &str, language: &Language) -> Vec<Vec<(String, Style)>> {
        let lines: Vec<&str> = code.lines().collect();
        let mut highlighted_lines = Vec::new();
        
        for line in lines {
            if let Some(rules) = self.rules.get(language) {
                let tokens = self.tokenize_line(line, rules);
                let spans: Vec<(String, Style)> = tokens.into_iter()
                    .map(|token| (token.text, token.style))
                    .collect();
                highlighted_lines.push(spans);
            } else {
                highlighted_lines.push(vec![(line.to_string(), Style::default())]);
            }
        }
        
        highlighted_lines
    }
}

/// Theme for syntax highlighting
pub struct SyntaxTheme {
    pub name: String,
    pub colors: HashMap<TokenType, Style>,
}

impl SyntaxTheme {
    /// Create a Monokai-inspired theme
    pub fn monokai() -> Self {
        let mut colors = HashMap::new();
        
        colors.insert(TokenType::Keyword, Style::default().fg(Color::Rgb(249, 38, 114)));
        colors.insert(TokenType::String, Style::default().fg(Color::Rgb(230, 219, 116)));
        colors.insert(TokenType::Number, Style::default().fg(Color::Rgb(174, 129, 255)));
        colors.insert(TokenType::Comment, Style::default().fg(Color::Rgb(117, 113, 94)));
        colors.insert(TokenType::Function, Style::default().fg(Color::Rgb(166, 226, 46)));
        colors.insert(TokenType::Type, Style::default().fg(Color::Rgb(102, 217, 239)));
        colors.insert(TokenType::Operator, Style::default().fg(Color::Rgb(249, 38, 114)));
        colors.insert(TokenType::Punctuation, Style::default().fg(Color::Rgb(248, 248, 242)));
        colors.insert(TokenType::Variable, Style::default().fg(Color::Rgb(248, 248, 242)));
        colors.insert(TokenType::Constant, Style::default().fg(Color::Rgb(174, 129, 255)));
        colors.insert(TokenType::Attribute, Style::default().fg(Color::Rgb(166, 226, 46)));
        colors.insert(TokenType::Builtin, Style::default().fg(Color::Rgb(102, 217, 239)));
        colors.insert(TokenType::Error, Style::default().fg(Color::Rgb(249, 38, 114)).add_modifier(Modifier::UNDERLINED));
        colors.insert(TokenType::Normal, Style::default().fg(Color::Rgb(248, 248, 242)));
        
        Self {
            name: "Monokai".to_string(),
            colors,
        }
    }
    
    /// Create a Dracula-inspired theme
    pub fn dracula() -> Self {
        let mut colors = HashMap::new();
        
        colors.insert(TokenType::Keyword, Style::default().fg(Color::Rgb(255, 121, 198)));
        colors.insert(TokenType::String, Style::default().fg(Color::Rgb(241, 250, 140)));
        colors.insert(TokenType::Number, Style::default().fg(Color::Rgb(189, 147, 249)));
        colors.insert(TokenType::Comment, Style::default().fg(Color::Rgb(98, 114, 164)));
        colors.insert(TokenType::Function, Style::default().fg(Color::Rgb(80, 250, 123)));
        colors.insert(TokenType::Type, Style::default().fg(Color::Rgb(139, 233, 253)));
        colors.insert(TokenType::Operator, Style::default().fg(Color::Rgb(255, 121, 198)));
        colors.insert(TokenType::Punctuation, Style::default().fg(Color::Rgb(248, 248, 242)));
        colors.insert(TokenType::Variable, Style::default().fg(Color::Rgb(248, 248, 242)));
        colors.insert(TokenType::Constant, Style::default().fg(Color::Rgb(189, 147, 249)));
        colors.insert(TokenType::Attribute, Style::default().fg(Color::Rgb(80, 250, 123)));
        colors.insert(TokenType::Builtin, Style::default().fg(Color::Rgb(139, 233, 253)));
        colors.insert(TokenType::Error, Style::default().fg(Color::Rgb(255, 85, 85)).add_modifier(Modifier::UNDERLINED));
        colors.insert(TokenType::Normal, Style::default().fg(Color::Rgb(248, 248, 242)));
        
        Self {
            name: "Dracula".to_string(),
            colors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_language_parsing() {
        assert_eq!(Language::from_str("rust"), Some(Language::Rust));
        assert_eq!(Language::from_str("rs"), Some(Language::Rust));
        assert_eq!(Language::from_str("python"), Some(Language::Python));
        assert_eq!(Language::from_str("js"), Some(Language::JavaScript));
        assert_eq!(Language::from_str("unknown"), None);
    }
    
    #[test]
    fn test_basic_highlighting() {
        let highlighter = SyntaxHighlighter::new();
        let code = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let highlighted = highlighter.highlight(code, &Language::Rust);
        
        assert_eq!(highlighted.len(), 3);
        assert!(highlighted[0].contains("fn"));
        assert!(highlighted[1].contains("println!"));
    }
}