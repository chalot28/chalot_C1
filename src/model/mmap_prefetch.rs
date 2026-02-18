// =============================================================================
// model/mmap_prefetch.rs — Memory-Mapped File Prefetching Optimization
// =============================================================================
//
// Problem: mmap relies on OS page faults to load data on-demand. This causes
// "jittery" latency spikes when accessing new expert weights.
//
// Solution: Use platform-specific hints (madvise/PrefetchVirtualMemory) to
// tell the OS which memory regions we'll need soon.
//
// Benefits:
// - Reduced page fault latency (data pre-loaded into RAM)
// - Smoother inference (no "lag spikes" when switching experts)
// - Better pipeline utilization (CPU doesn't stall waiting for I/O)
//
// Platform support:
//   - Linux/macOS: madvise(MADV_WILLNEED, MADV_SEQUENTIAL)
//   - Windows: PrefetchVirtualMemory
//   - Fallback: No-op (safe but unoptimized)
// =============================================================================

use memmap2::Mmap;

/// Prefetch hint for sequential access pattern
#[allow(dead_code)]
#[cfg(target_os = "linux")]
pub fn prefetch_sequential(mmap: &Mmap, offset: usize, len: usize) -> Result<(), String> {
    use std::os::unix::io::AsRawFd;
    
    if offset + len > mmap.len() {
        return Err("Prefetch range exceeds mmap size".to_string());
    }
    
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *mut libc::c_void;
        let result = libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        if result != 0 {
            return Err(format!("madvise(MADV_SEQUENTIAL) failed: {}", result));
        }
        
        let result = libc::madvise(ptr, len, libc::MADV_WILLNEED);
        if result != 0 {
            return Err(format!("madvise(MADV_WILLNEED) failed: {}", result));
        }
    }
    
    Ok(())
}

/// Prefetch hint for sequential access pattern (macOS)
#[allow(dead_code)]
#[cfg(target_os = "macos")]
pub fn prefetch_sequential(mmap: &Mmap, offset: usize, len: usize) -> Result<(), String> {
    if offset + len > mmap.len() {
        return Err("Prefetch range exceeds mmap size".to_string());
    }
    
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *mut libc::c_void;
        let result = libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        if result != 0 {
            return Err(format!("madvise(MADV_SEQUENTIAL) failed: {}", result));
        }
        
        let result = libc::madvise(ptr, len, libc::MADV_WILLNEED);
        if result != 0 {
            return Err(format!("madvise(MADV_WILLNEED) failed: {}", result));
        }
    }
    
    Ok(())
}

/// Prefetch hint for Windows
#[allow(dead_code)]
#[cfg(target_os = "windows")]
pub fn prefetch_sequential(mmap: &Mmap, offset: usize, len: usize) -> Result<(), String> {
    // Windows prefetching using VirtualLock
    // PrefetchVirtualMemory requires Windows 8+, so we use a safer approach
    use std::ptr;
    
    if offset + len > mmap.len() {
        return Err("Prefetch range exceeds mmap size".to_string());
    }
    
    // Touch the memory pages to force them into RAM
    unsafe {
        let ptr = mmap.as_ptr().add(offset);
        let page_size = 4096;
        let num_pages = (len + page_size - 1) / page_size;
        
        for i in 0..num_pages {
            let page_ptr = ptr.add(i * page_size);
            if (i * page_size) < len {
                // Volatile read to force page load
                ptr::read_volatile(page_ptr);
            }
        }
    }
    
    Ok(())
}

/// Fallback for unsupported platforms (no-op)
#[allow(dead_code)]
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub fn prefetch_sequential(_mmap: &Mmap, _offset: usize, _len: usize) -> Result<(), String> {
    // No-op on unsupported platforms
    Ok(())
}

/// Prefetch a specific expert's weights before computation
#[allow(dead_code)]
pub fn prefetch_expert_weights(
    mmap: &Mmap,
    gate_up_offset: usize,
    gate_up_bytes: usize,
    down_offset: usize,
    down_bytes: usize,
) -> Result<(), String> {
    // Prefetch gate+up weights
    prefetch_sequential(mmap, gate_up_offset, gate_up_bytes)?;
    
    // Prefetch down weights
    prefetch_sequential(mmap, down_offset, down_bytes)?;
    
    Ok(())
}

/// Prefetch attention weights for a layer
#[allow(dead_code)]
pub fn prefetch_attention_weights(
    mmap: &Mmap,
    qkv_offset: usize,
    qkv_bytes: usize,
    out_offset: usize,
    out_bytes: usize,
) -> Result<(), String> {
    // Prefetch QKV projection weights
    prefetch_sequential(mmap, qkv_offset, qkv_bytes)?;
    
    // Prefetch output projection weights
    prefetch_sequential(mmap, out_offset, out_bytes)?;
    
    Ok(())
}

/// Mark memory region as "done" (won't be needed soon)
/// This allows OS to evict pages sooner if memory is tight
#[allow(dead_code)]
#[cfg(target_os = "linux")]
pub fn mark_done(mmap: &Mmap, offset: usize, len: usize) -> Result<(), String> {
    if offset + len > mmap.len() {
        return Err("Mark done range exceeds mmap size".to_string());
    }
    
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *mut libc::c_void;
        let result = libc::madvise(ptr, len, libc::MADV_DONTNEED);
        if result != 0 {
            return Err(format!("madvise(MADV_DONTNEED) failed: {}", result));
        }
    }
    
    Ok(())
}

/// Mark memory region as "done" (macOS)
#[allow(dead_code)]
#[cfg(target_os = "macos")]
pub fn mark_done(mmap: &Mmap, offset: usize, len: usize) -> Result<(), String> {
    if offset + len > mmap.len() {
        return Err("Mark done range exceeds mmap size".to_string());
    }
    
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *mut libc::c_void;
        let result = libc::madvise(ptr, len, libc::MADV_FREE);
        if result != 0 {
            return Err(format!("madvise(MADV_FREE) failed: {}", result));
        }
    }
    
    Ok(())
}

/// Mark memory region as "done" (Windows - no-op)
#[allow(dead_code)]
#[cfg(target_os = "windows")]
pub fn mark_done(_mmap: &Mmap, _offset: usize, _len: usize) -> Result<(), String> {
    // Windows doesn't have a direct equivalent
    // Could use VirtualUnlock, but it's not necessary
    Ok(())
}

/// Mark memory region as "done" (fallback)
#[allow(dead_code)]
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub fn mark_done(_mmap: &Mmap, _offset: usize, _len: usize) -> Result<(), String> {
    Ok(())
}

/// Prefetch strategy for predictive loading
#[allow(dead_code)]
pub struct PrefetchStrategy {
    /// Enable prefetching
    pub enabled: bool,
    /// Number of experts to prefetch ahead
    pub lookahead: usize,
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        Self {
            enabled: true,
            lookahead: 2,
        }
    }
}

impl PrefetchStrategy {
    /// Prefetch top-K experts for a layer
    #[allow(dead_code)]
    pub fn prefetch_top_experts(
        &self,
        mmap: &Mmap,
        expert_indices: &[(usize, usize, usize, usize, usize)], // (id, gu_off, gu_bytes, down_off, down_bytes)
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        
        for (_, gu_off, gu_bytes, down_off, down_bytes) in expert_indices.iter().take(self.lookahead) {
            prefetch_expert_weights(mmap, *gu_off, *gu_bytes, *down_off, *down_bytes)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prefetch_strategy() {
        let strategy = PrefetchStrategy::default();
        assert!(strategy.enabled);
        assert_eq!(strategy.lookahead, 2);
    }
}
