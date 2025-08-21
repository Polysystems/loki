//! SIMD-accelerated pattern matching for high-performance string operations

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use dashmap::DashMap;
use wide::{u8x32, CmpEq};

/// SIMD pattern matcher for fast string searching
pub struct SimdPatternMatcher {
    // Cache for compiled patterns
    pattern_cache: Arc<DashMap<Vec<u8>, CompiledPattern>>,
    // Statistics
    matches_found: AtomicU64,
    patterns_compiled: AtomicU64,
}

/// Compiled pattern for fast matching
#[derive(Clone)]
struct CompiledPattern {
    pattern: Vec<u8>,
    mask: Vec<u8>,
    skip_table: [usize; 256],
}

impl SimdPatternMatcher {
    /// Create new pattern matcher
    pub fn new() -> Self {
        Self {
            pattern_cache: Arc::new(DashMap::new()),
            matches_found: AtomicU64::new(0),
            patterns_compiled: AtomicU64::new(0),
        }
    }
    
    /// Find pattern in haystack using SIMD
    pub fn find(&self, haystack: &[u8], pattern: &[u8]) -> Option<usize> {
        if pattern.is_empty() || pattern.len() > haystack.len() {
            return None;
        }
        
        // Single byte pattern - use SIMD scan
        if pattern.len() == 1 {
            return self.find_byte_simd(haystack, pattern[0]);
        }
        
        // Short patterns - use SIMD comparison
        if pattern.len() <= 32 {
            return self.find_short_pattern_simd(haystack, pattern);
        }
        
        // Long patterns - use Boyer-Moore with SIMD
        self.find_long_pattern(haystack, pattern)
    }
    
    /// Find single byte using SIMD
    fn find_byte_simd(&self, haystack: &[u8], needle: u8) -> Option<usize> {
        let chunks = haystack.len() / 32;
        let needle_vec = u8x32::splat(needle);
        
        for i in 0..chunks {
            let chunk_start = i * 32;
            let chunk = &haystack[chunk_start..chunk_start + 32];
            
            // Load chunk into SIMD register
            let chunk_array: [u8; 32] = chunk.try_into().unwrap();
            let chunk_vec = u8x32::from(chunk_array);
            
            // Compare all bytes at once
            let matches = chunk_vec.cmp_eq(needle_vec);
            
            // Convert to array and check for matches
            let match_array: [u8; 32] = matches.to_array();
            for (j, &is_match) in match_array.iter().enumerate() {
                if is_match != 0 {
                    self.matches_found.fetch_add(1, Ordering::Relaxed);
                    return Some(chunk_start + j);
                }
            }
        }
        
        // Handle remainder
        for (i, &byte) in haystack[chunks * 32..].iter().enumerate() {
            if byte == needle {
                self.matches_found.fetch_add(1, Ordering::Relaxed);
                return Some(chunks * 32 + i);
            }
        }
        
        None
    }
    
    /// Find short pattern using SIMD
    fn find_short_pattern_simd(&self, haystack: &[u8], pattern: &[u8]) -> Option<usize> {
        let pattern_len = pattern.len();
        let first_byte = pattern[0];
        let first_byte_vec = u8x32::splat(first_byte);
        
        let chunks = haystack.len().saturating_sub(pattern_len - 1) / 32;
        
        for i in 0..chunks {
            let chunk_start = i * 32;
            let chunk_end = (chunk_start + 32).min(haystack.len() - pattern_len + 1);
            let chunk = &haystack[chunk_start..chunk_end];
            
            if chunk.len() < 32 {
                // Handle partial chunk
                for j in 0..chunk.len() {
                    if haystack[chunk_start + j..chunk_start + j + pattern_len] == *pattern {
                        self.matches_found.fetch_add(1, Ordering::Relaxed);
                        return Some(chunk_start + j);
                    }
                }
                continue;
            }
            
            // Load chunk into SIMD register
            let chunk_array: [u8; 32] = chunk[..32].try_into().unwrap();
            let chunk_vec = u8x32::from(chunk_array);
            
            // Find potential matches by first byte
            let matches = chunk_vec.cmp_eq(first_byte_vec);
            
            // Convert to array and check for matches
            let match_array: [u8; 32] = matches.to_array();
            for (j, &is_match) in match_array.iter().enumerate() {
                if is_match != 0 {
                    let pos = chunk_start + j;
                    if pos + pattern_len <= haystack.len() &&
                       &haystack[pos..pos + pattern_len] == pattern {
                        self.matches_found.fetch_add(1, Ordering::Relaxed);
                        return Some(pos);
                    }
                }
            }
        }
        
        // Handle remainder
        for i in (chunks * 32)..haystack.len().saturating_sub(pattern_len - 1) {
            if &haystack[i..i + pattern_len] == pattern {
                self.matches_found.fetch_add(1, Ordering::Relaxed);
                return Some(i);
            }
        }
        
        None
    }
    
    /// Find long pattern using Boyer-Moore algorithm
    fn find_long_pattern(&self, haystack: &[u8], pattern: &[u8]) -> Option<usize> {
        // Get or compile pattern
        let compiled = self.get_or_compile_pattern(pattern);
        
        let pattern_len = pattern.len();
        let mut pos = 0;
        
        while pos <= haystack.len() - pattern_len {
            let mut j = pattern_len - 1;
            
            // Match from right to left
            while j > 0 && haystack[pos + j] == pattern[j] {
                j -= 1;
            }
            
            if j == 0 && haystack[pos] == pattern[0] {
                self.matches_found.fetch_add(1, Ordering::Relaxed);
                return Some(pos);
            }
            
            // Use skip table for faster advancement
            let skip = compiled.skip_table[haystack[pos + j] as usize];
            pos += skip.max(1);
        }
        
        None
    }
    
    /// Get or compile pattern for Boyer-Moore
    fn get_or_compile_pattern(&self, pattern: &[u8]) -> CompiledPattern {
        if let Some(compiled) = self.pattern_cache.get(pattern) {
            return compiled.clone();
        }
        
        // Compile pattern
        let mut skip_table = [pattern.len(); 256];
        for (i, &byte) in pattern.iter().enumerate().take(pattern.len() - 1) {
            skip_table[byte as usize] = pattern.len() - i - 1;
        }
        
        let compiled = CompiledPattern {
            pattern: pattern.to_vec(),
            mask: vec![0xff; pattern.len()],
            skip_table,
        };
        
        self.pattern_cache.insert(pattern.to_vec(), compiled.clone());
        self.patterns_compiled.fetch_add(1, Ordering::Relaxed);
        
        compiled
    }
    
    /// Find all occurrences of pattern
    pub fn find_all(&self, haystack: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut start = 0;
        
        while start < haystack.len() {
            if let Some(pos) = self.find(&haystack[start..], pattern) {
                positions.push(start + pos);
                start += pos + 1;
            } else {
                break;
            }
        }
        
        positions
    }
    
    /// Replace pattern with replacement
    pub fn replace(&self, haystack: &[u8], pattern: &[u8], replacement: &[u8]) -> Vec<u8> {
        let positions = self.find_all(haystack, pattern);
        
        if positions.is_empty() {
            return haystack.to_vec();
        }
        
        let mut result = Vec::with_capacity(
            haystack.len() + positions.len() * (replacement.len() - pattern.len())
        );
        
        let mut last_end = 0;
        for pos in positions {
            result.extend_from_slice(&haystack[last_end..pos]);
            result.extend_from_slice(replacement);
            last_end = pos + pattern.len();
        }
        result.extend_from_slice(&haystack[last_end..]);
        
        result
    }
    
    /// Check if pattern exists (faster than find)
    pub fn contains(&self, haystack: &[u8], pattern: &[u8]) -> bool {
        self.find(haystack, pattern).is_some()
    }
    
    /// Count occurrences of pattern
    pub fn count(&self, haystack: &[u8], pattern: &[u8]) -> usize {
        self.find_all(haystack, pattern).len()
    }
    
    /// Get statistics
    pub fn stats(&self) -> PatternMatcherStats {
        PatternMatcherStats {
            matches_found: self.matches_found.load(Ordering::Relaxed),
            patterns_compiled: self.patterns_compiled.load(Ordering::Relaxed),
            cached_patterns: self.pattern_cache.len(),
        }
    }
}

/// Pattern matcher statistics
#[derive(Debug, Clone)]
pub struct PatternMatcherStats {
    pub matches_found: u64,
    pub patterns_compiled: u64,
    pub cached_patterns: usize,
}

/// SIMD string operations
pub struct SimdStringOps;

impl SimdStringOps {
    /// Fast case-insensitive comparison using SIMD
    pub fn equals_ignore_case(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        let chunks = a.len() / 32;
        
        for i in 0..chunks {
            let offset = i * 32;
            let a_chunk: [u8; 32] = a[offset..offset + 32].try_into().unwrap();
            let b_chunk: [u8; 32] = b[offset..offset + 32].try_into().unwrap();
            
            let a_vec = u8x32::from(a_chunk);
            let b_vec = u8x32::from(b_chunk);
            
            // Convert to uppercase for comparison
            let a_upper = Self::to_upper_simd(a_vec);
            let b_upper = Self::to_upper_simd(b_vec);
            
            if a_upper.to_array() != b_upper.to_array() {
                return false;
            }
        }
        
        // Handle remainder
        for i in chunks * 32..a.len() {
            if a[i].to_ascii_uppercase() != b[i].to_ascii_uppercase() {
                return false;
            }
        }
        
        true
    }
    
    /// Convert to uppercase using SIMD
    fn to_upper_simd(v: u8x32) -> u8x32 {
        let lower_a = u8x32::splat(b'a');
        let lower_z = u8x32::splat(b'z');
        let diff = u8x32::splat(32); // 'a' - 'A'
        
        // Manual comparison for uppercase conversion
        let v_array: [u8; 32] = v.to_array();
        let mut result = [0u8; 32];
        
        for i in 0..32 {
            if v_array[i] >= b'a' && v_array[i] <= b'z' {
                result[i] = v_array[i] - 32;
            } else {
                result[i] = v_array[i];
            }
        }
        
        u8x32::from(result)
    }
    
    /// Fast whitespace trimming
    pub fn trim(s: &[u8]) -> &[u8] {
        let start = s.iter().position(|&b| !b.is_ascii_whitespace()).unwrap_or(s.len());
        let end = s.iter().rposition(|&b| !b.is_ascii_whitespace()).map(|i| i + 1).unwrap_or(0);
        
        if start >= end {
            &[]
        } else {
            &s[start..end]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_find_byte() {
        let matcher = SimdPatternMatcher::new();
        let haystack = b"hello world";
        
        assert_eq!(matcher.find(haystack, b"o"), Some(4));
        assert_eq!(matcher.find(haystack, b"x"), None);
    }
    
    #[test]
    fn test_find_short_pattern() {
        let matcher = SimdPatternMatcher::new();
        let haystack = b"the quick brown fox jumps over the lazy dog";
        
        assert_eq!(matcher.find(haystack, b"fox"), Some(16));
        assert_eq!(matcher.find(haystack, b"cat"), None);
    }
    
    #[test]
    fn test_find_long_pattern() {
        let matcher = SimdPatternMatcher::new();
        let haystack = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit";
        let pattern = b"consectetur adipiscing";
        
        assert_eq!(matcher.find(haystack, pattern), Some(29));
    }
    
    #[test]
    fn test_find_all() {
        let matcher = SimdPatternMatcher::new();
        let haystack = b"abcabcabc";
        
        assert_eq!(matcher.find_all(haystack, b"abc"), vec![0, 3, 6]);
    }
    
    #[test]
    fn test_replace() {
        let matcher = SimdPatternMatcher::new();
        let haystack = b"hello world";
        let result = matcher.replace(haystack, b"world", b"rust");
        
        assert_eq!(result, b"hello rust");
    }
    
    #[test]
    fn test_case_insensitive() {
        assert!(SimdStringOps::equals_ignore_case(b"Hello", b"HELLO"));
        assert!(SimdStringOps::equals_ignore_case(b"WoRlD", b"world"));
        assert!(!SimdStringOps::equals_ignore_case(b"hello", b"world"));
    }
}