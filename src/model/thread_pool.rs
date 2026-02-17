// =============================================================================
// model/thread_pool.rs — Persistent Thread Pool for MoE Expert Execution
// =============================================================================
//
// Problem: Creating and destroying threads (spawn/join) on every forward pass
// is expensive (OS overhead, memory allocation, context switching).
//
// Solution: Pre-allocate a fixed thread pool at Engine initialization and
// reuse the same threads for all expert computations.
//
// Benefits:
// - ~10-50× faster thread reuse (no spawn overhead)
// - Reduced memory allocation
// - Lower CPU usage and heat generation
// - Better cache locality
// =============================================================================

use std::sync::{Arc, Mutex};
use std::thread;

/// A work item for an expert computation
pub struct ExpertTask {
    pub expert_id: usize,
    pub input: Vec<f32>,
    pub gate_up_packed: Vec<u8>,
    pub gate_up_scales: Vec<f32>,
    pub down_packed: Vec<u8>,
    pub down_scales: Vec<f32>,
    pub dim: usize,
    pub hidden: usize,
}

/// Result from an expert computation
pub struct ExpertResult {
    pub expert_id: usize,
    pub output: Vec<f32>,
}

/// A job in the queue
enum Job {
    Task(ExpertTask),
    Shutdown,
}

/// Thread pool for MoE expert processing
pub struct ExpertThreadPool {
    workers: Vec<Worker>,
    sender: std::sync::mpsc::Sender<Job>,
    result_receiver: Arc<Mutex<std::sync::mpsc::Receiver<ExpertResult>>>,
}

impl ExpertThreadPool {
    /// Create a new thread pool with the specified number of threads.
    /// Typically, use num_cpus::get() or a fraction thereof.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0);
        
        let (job_sender, job_receiver) = std::sync::mpsc::channel();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let result_receiver = Arc::new(Mutex::new(result_receiver));
        
        let mut workers = Vec::with_capacity(num_threads);
        
        for id in 0..num_threads {
            workers.push(Worker::new(
                id,
                Arc::clone(&job_receiver),
                result_sender.clone(),
            ));
        }
        
        ExpertThreadPool {
            workers,
            sender: job_sender,
            result_receiver,
        }
    }
    
    /// Submit a task to the pool. Non-blocking.
    pub fn submit(&self, task: ExpertTask) -> Result<(), String> {
        self.sender.send(Job::Task(task))
            .map_err(|_| "Failed to send task to thread pool".to_string())
    }
    
    /// Collect results. Blocks until all submitted tasks are complete.
    /// Returns results in the order they complete (not submission order).
    pub fn collect_results(&self, count: usize) -> Vec<ExpertResult> {
        let mut results = Vec::with_capacity(count);
        let receiver = self.result_receiver.lock().unwrap();
        
        for _ in 0..count {
            match receiver.recv() {
                Ok(result) => results.push(result),
                Err(_) => break,
            }
        }
        
        results
    }
    
    /// Get number of worker threads
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ExpertThreadPool {
    fn drop(&mut self) {
        // Send shutdown signal to all workers
        for _ in &self.workers {
            let _ = self.sender.send(Job::Shutdown);
        }
        
        // Wait for all workers to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// A worker thread in the pool
struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<std::sync::mpsc::Receiver<Job>>>,
        result_sender: std::sync::mpsc::Sender<ExpertResult>,
    ) -> Self {
        let thread = thread::spawn(move || {
            loop {
                let job = {
                    let receiver = receiver.lock().unwrap();
                    receiver.recv()
                };
                
                match job {
                    Ok(Job::Task(task)) => {
                        // Process the expert task
                        let result = process_expert_task(task);
                        let _ = result_sender.send(result);
                    }
                    Ok(Job::Shutdown) | Err(_) => {
                        break;
                    }
                }
            }
        });
        
        Worker {
            id,
            thread: Some(thread),
        }
    }
}

/// Process a single expert task
fn process_expert_task(task: ExpertTask) -> ExpertResult {
    use crate::tensor::{matmul_int4, swiglu_fused};
    
    let dim = task.dim;
    let hidden = task.hidden;
    
    // Allocate local buffers
    let mut local_hb = vec![0.0f32; hidden];
    let mut local_hb2 = vec![0.0f32; hidden];
    let mut local_tmp = vec![0.0f32; dim.max(hidden)];
    let mut out_buf = vec![0.0f32; dim];
    
    // Copy input
    local_tmp[..dim].copy_from_slice(&task.input);
    
    // Gate projection
    matmul_int4(
        &mut local_hb,
        &task.gate_up_packed[..task.gate_up_packed.len() / 2],
        &task.gate_up_scales[..task.gate_up_scales.len() / 2],
        &local_tmp[..dim],
        hidden,
        dim,
    );
    
    // Up projection
    matmul_int4(
        &mut local_hb2,
        &task.gate_up_packed[task.gate_up_packed.len() / 2..],
        &task.gate_up_scales[task.gate_up_scales.len() / 2..],
        &local_tmp[..dim],
        hidden,
        dim,
    );
    
    // SwiGLU activation
    swiglu_fused(&mut local_hb, &local_hb2);
    
    // Down projection
    local_tmp[..hidden].copy_from_slice(&local_hb);
    matmul_int4(
        &mut out_buf,
        &task.down_packed,
        &task.down_scales,
        &local_tmp[..hidden],
        dim,
        hidden,
    );
    
    ExpertResult {
        expert_id: task.expert_id,
        output: out_buf,
    }
}

// ---------------------------------------------------------------------------
// Rayon-based alternative (simpler but less control)
// ---------------------------------------------------------------------------

/// Execute experts in parallel using Rayon's thread pool.
/// This is simpler than the custom pool but gives less control.
#[allow(dead_code)]
pub fn execute_experts_rayon<F>(
    expert_ids: &[(usize, f32)],
    process_fn: F,
) -> Vec<Vec<f32>>
where
    F: Fn(usize) -> Vec<f32> + Send + Sync,
{
    use rayon::prelude::*;
    
    expert_ids
        .par_iter()
        .map(|(expert_id, _)| process_fn(*expert_id))
        .collect()
}
