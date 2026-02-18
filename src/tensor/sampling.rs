// =============================================================================
// tensor/sampling.rs — Token sampling strategies
// =============================================================================

use super::normalization::softmax;

/// Greedy argmax over logits. Returns the token id.
#[inline(always)]
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Top-K sampling with temperature.
#[allow(dead_code)]
pub fn sample_top_k(logits: &mut [f32], k: usize, temperature: f32) -> usize {
    // Apply temperature
    if temperature > 0.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() {
            *v *= inv_t;
        }
    }

    // Find top-k indices
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    indices.truncate(k);

    // Softmax over top-k
    let max_val = logits[indices[0]];
    let mut probs: Vec<f32> = indices.iter().map(|&i| (logits[i] - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Simple random selection using a basic LCG (no external deps)
    // In production, use a proper RNG.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return indices[i];
        }
    }
    indices[indices.len() - 1]
}

/// Top-p (nucleus) sampling: sample from the smallest token set whose
/// cumulative probability exceeds `p`. Superior to fixed top-k for reasoning
/// because the candidate pool adapts to model confidence.
#[allow(dead_code)]
pub fn sample_top_p(logits: &mut [f32], p: f32, temperature: f32) -> usize {
    let n = logits.len();
    if n == 0 { return 0; }

    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() { *v *= inv_t; }
    }

    softmax(logits);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let mut cumulative = 0.0f32;
    let mut cutoff = n;
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += logits[idx];
        if cumulative > p {
            cutoff = rank + 1;
            break;
        }
    }
    indices.truncate(cutoff);

    let mut probs: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 1e-12 {
        for val in probs.iter_mut() { *val /= sum; }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cum = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cum += prob;
        if r < cum { return indices[i]; }
    }
    indices[indices.len() - 1]
}

/// Min-P sampling: keeps tokens with prob ≥ min_p × max_prob.
/// Adaptive cutoff that automatically scales with model confidence.
/// More principled than fixed top-k or top-p for variable-difficulty tasks.
#[allow(dead_code)]
pub fn sample_min_p(logits: &mut [f32], min_p: f32, temperature: f32) -> usize {
    let n = logits.len();
    if n == 0 { return 0; }

    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() { *v *= inv_t; }
    }

    softmax(logits);

    let max_prob = logits.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_prob * min_p;

    let indices: Vec<usize> = (0..n).filter(|&i| logits[i] >= threshold).collect();
    if indices.is_empty() {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }

    let mut probs: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 1e-12 {
        for val in probs.iter_mut() { *val /= sum; }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cum = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cum += prob;
        if r < cum { return indices[i]; }
    }
    indices[indices.len() - 1]
}

/// Apply repetition penalty: reduce probability of recently generated tokens.
/// penalty > 1.0 discourages repetition; typical range: 1.05–1.3
#[allow(dead_code)]
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[usize], penalty: f32) {
    for &tok in history {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}
