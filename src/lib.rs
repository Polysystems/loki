#![allow(unused_variables)]
#![cfg_attr(feature = "simd-optimizations", feature(portable_simd))]
#![feature(likely_unlikely)]

// Performance optimization macros for critical hot paths
#[macro_export]
macro_rules! hot_path {
    ($($code:tt)*) => {{
        // TODO
        $($code)*
    }}
}

// Core error handling system
pub mod error;
pub mod infrastructure;
pub mod zero_cost_validation;
pub mod code_generation_analysis;
pub mod compiler_backend_optimization;
pub mod cli;
pub mod auth;
pub mod cluster;
pub mod cognitive;
pub mod compute;
pub mod config;
pub mod core;
pub mod daemon;
pub mod database;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod monitoring;
pub mod ollama;
pub mod persistence;
pub mod plugins;
pub mod safety;
pub mod social;
pub mod storage;
pub mod story;
pub mod streaming;
pub mod tasks;
pub mod tools;
pub mod tui;
pub mod ui;
pub mod utils;

// Re-export panic_safe utilities for easy access
pub use core::*;
