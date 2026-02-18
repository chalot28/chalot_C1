// =============================================================================
// main.rs — Sparse MoE AI Engine + Multi-Source Data Crawler
// =============================================================================
//
// Architecture: ~272M param Sparse MoE Transformer (v3 Extreme Reasoning)
//   dim=512, hidden=1536, 12 layers, 8 heads, vocab=32000
//   8 experts/layer (top-2 active) + Adaptive Depth + Int4 + AVX2 SIMD
//
// CLI Commands:
//   demo                           Create 150M MoE model, run forward pass
//   info    <model.myai>           Show model + MoE + memory info
//   crawl   <model.myai> <URL>     Crawl URL → tokenize → inference
//   api     <model.myai> <URL>     Call API endpoint
//   file    <model.myai> <PATH>    Read file → tokenize → inference
//   sys     <model.myai> <CMD>     Run command → capture → inference
//   gen     <model.myai> <TEXT>    Tokenize text → inference
//   train-tok <corpus> [N]        Train BPE tokenizer on corpus file
//
// Sparse Loading: model uses mmap — only active experts paged into RAM
// Adaptive Depth: model self-assesses query complexity for early exit
// =============================================================================

mod tensor;
mod model;
mod tokenizer;
mod crawler;

use model::{Engine, ModelConfig, MAX_SEQ_LEN, create_dummy_model, DEPTH_ROUTER_AFTER_LAYER, SamplingConfig, SamplingStrategy};
use tokenizer::Tokenizer;
use crawler::Crawler;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::io::Write;

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------
fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "help" | "--help" | "-h" => print_help(),
        "demo" => cmd_demo(),
        "info" => cmd_info(&args),
        "crawl" => cmd_crawl(&args),
        "api" => cmd_api(&args),
        "file" => cmd_file(&args),
        "sys" => cmd_sys(&args),
        "gen" => cmd_gen(&args),
        "train-tok" => cmd_train_tok(&args),
        other => {
            // If first arg looks like a .myai file, treat it as: <model> <cmd> ...
            if other.ends_with(".myai") {
                if args.len() >= 3 {
                    let mut shifted = vec![args[0].clone(), args[2].clone(), args[1].clone()];
                    shifted.extend(args[3..].iter().cloned());
                    match shifted[1].as_str() {
                        "crawl" => cmd_crawl(&shifted),
                        "api" => cmd_api(&shifted),
                        "file" => cmd_file(&shifted),
                        "sys" => cmd_sys(&shifted),
                        "gen" => cmd_gen(&shifted),
                        "info" => cmd_info(&shifted),
                        _ => {
                            eprintln!("[error] Unknown command: {}", shifted[1]);
                            print_help();
                        }
                    }
                } else {
                    cmd_info(&args);
                }
            } else {
                eprintln!("[error] Unknown command: {}", other);
                print_help();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
fn print_help() {
    println!(
        r#"
=== AI_chalot_C1 v3 — Extreme Reasoning MoE Engine (~272M params) ===

ARCHITECTURE:
    Sparse Mixture-of-Experts Transformer (v3)
    dim=512, hidden=1536, 12 layers, 8 heads, vocab=32000
    8 experts/layer, top-2 activated per token
    Adaptive depth: continuous depth routing with GELU-gated MLP
    Mixed quantization: Int8 attention + Int4 experts (group-wise)
    Attention logit soft-capping (anti-entropy collapse)
    AVX2 SIMD: 2-row parallel int8 matmul kernel

USAGE:
    AI_chalot_C1 <command> [args...]

COMMANDS:
    demo                            Create ~272M model + test forward pass
    info    <model.myai>            Show model, MoE & memory info
    crawl   <model.myai> <URL>      Crawl web page → tokenize → inference
    api     <model.myai> <URL>      Call API endpoint → tokenize → inference
                                     [-m METHOD] [-b BODY] [-H "Key: Value"]
    file    <model.myai> <PATH>     Read file → tokenize → inference
    sys     <model.myai> <CMD>      Run command → capture → inference
    gen     <model.myai> <TEXT>     Generate from text input
    train-tok <corpus_file> [N]     Train BPE tokenizer (N merges, default 4000)
    help                            Show this help

EXAMPLES:
    AI_chalot_C1 demo
    AI_chalot_C1 crawl model.myai https://example.com
    AI_chalot_C1 gen model.myai "Hello world"

SPARSE LOADING:
    Model weights stay on disk via mmap.  Only expert weights that the
    router selects are paged into RAM.  Simple queries trigger early-exit
    (fewer layers), saving compute and memory automatically.
"#
    );
}

// ---------------------------------------------------------------------------
// Command: demo
// ---------------------------------------------------------------------------
fn cmd_demo() {
    println!("=== Demo Mode — Sparse MoE 150M Architecture ===\n");

    let model_path = PathBuf::from("demo_model.myai");
    let cfg = default_config();
    let param_m = cfg.param_count() as f64 / 1e6;

    println!("[demo] Creating {:.1}M-param Sparse MoE model (v2)...", param_m);
    println!("[demo]   {} experts/layer, top-{} active, Int4 group={}",
        cfg.n_experts, cfg.top_k, cfg.int4_group_size);
    println!("[demo]   Adaptive depth router after layer {}\n", cfg.depth_router_layer);
    create_dummy_model(&model_path, &cfg).expect("Failed to create dummy model");

    let mut engine = load_engine(&model_path);

    print_memory_report(&engine);

    // Run forward pass
    let gen_len = 32;
    let (tokens, elapsed) = run_inference(&mut engine, 1, gen_len);

    println!(
        "[demo] Generated {} tokens in {:.1} ms ({:.1} tok/s)",
        gen_len,
        elapsed * 1000.0,
        gen_len as f64 / elapsed
    );
    println!("[demo] {}", engine.sparse_stats_report());
    println!("Token IDs: {:?}\n", tokens);

    // Test tokenizer
    println!("[demo] Testing tokenizer...");
    let tok = Tokenizer::new(cfg.vocab_size);
    let text = "Hello, World! Xin chao!";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    println!("  Input:   \"{}\"", text);
    println!("  Tokens:  {} IDs", ids.len());
    println!("  Decoded: \"{}\"", decoded);

    // Test crawler
    println!("\n[demo] Testing system info crawler...");
    let cr = Crawler::new();
    match cr.system_info() {
        Ok(r) => {
            println!("  Source: {}", r.source);
            println!("  Size:   {} bytes", r.size_bytes);
            for line in r.text.lines().take(5) {
                println!("  > {}", line);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Cleanup
    std::fs::remove_file(&model_path).ok();
    println!("\n[demo] Done. Cleaned up demo model.");
}

// ---------------------------------------------------------------------------
// Command: info
// ---------------------------------------------------------------------------
fn cmd_info(args: &[String]) {
    let model_path = match get_model_path(args, 2) {
        Some(p) => p,
        None => {
            eprintln!("[error] Usage: info <model.myai>");
            return;
        }
    };
    let engine = load_engine(&model_path);
    print_memory_report(&engine);
}

// ---------------------------------------------------------------------------
// Command: crawl
// ---------------------------------------------------------------------------
fn cmd_crawl(args: &[String]) {
    if args.len() < 4 {
        eprintln!("[error] Usage: crawl <model.myai> <URL>");
        return;
    }
    let model_path = PathBuf::from(&args[2]);
    let url = &args[3];

    let mut engine = load_engine(&model_path);
    let tok = load_tokenizer(&model_path, engine.config.vocab_size);
    let mut crawler = Crawler::new();

    println!("[crawl] Fetching: {}", url);
    match crawler.fetch_url(url) {
        Ok(result) => {
            println!(
                "[crawl] Status: {} | Size: {} bytes | Time: {} ms",
                result.status, result.size_bytes, result.elapsed_ms
            );
            println!("\n--- Extracted Text ({} chars) ---", result.text.len());

            // Show first 500 chars of extracted text
            let preview: String = result.text.chars().take(500).collect();
            println!("{}", preview);
            if result.text.len() > 500 {
                println!("... (truncated)");
            }

            // Tokenize + inference
            let tokens = tok.encode(&result.text);
            println!("\n[crawl] Tokenized: {} tokens", tokens.len());

            run_inference_pipeline(&mut engine, &tokens, &tok, &SamplingConfig::default(), 128);
        }
        Err(e) => eprintln!("[crawl] Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Command: api
// ---------------------------------------------------------------------------
fn cmd_api(args: &[String]) {
    if args.len() < 4 {
        eprintln!("[error] Usage: api <model.myai> <URL> [-m METHOD] [-b BODY] [-H \"Key: Value\"]");
        return;
    }
    let model_path = PathBuf::from(&args[2]);
    let url = &args[3];

    // Parse optional flags
    let mut method = "GET".to_string();
    let mut body: Option<String> = None;
    let mut headers: Vec<(String, String)> = Vec::new();

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--method" => {
                if i + 1 < args.len() {
                    method = args[i + 1].to_uppercase();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-b" | "--body" => {
                if i + 1 < args.len() {
                    body = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-H" | "--header" => {
                if i + 1 < args.len() {
                    if let Some(colon) = args[i + 1].find(':') {
                        let key = args[i + 1][..colon].trim().to_string();
                        let val = args[i + 1][colon + 1..].trim().to_string();
                        headers.push((key, val));
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    let mut engine = load_engine(&model_path);
    let tok = load_tokenizer(&model_path, engine.config.vocab_size);
    let mut crawler = Crawler::new();

    let hdr_refs: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    println!("[api] {} {}", method, url);
    match crawler.fetch_api(url, &method, body.as_deref(), &hdr_refs) {
        Ok(result) => {
            println!(
                "[api] Status: {} | Size: {} bytes | Time: {} ms",
                result.status, result.size_bytes, result.elapsed_ms
            );
            println!("\n--- Response Text ---");

            let preview: String = result.text.chars().take(800).collect();
            println!("{}", preview);
            if result.text.len() > 800 {
                println!("... (truncated)");
            }

            let tokens = tok.encode(&result.text);
            println!("\n[api] Tokenized: {} tokens", tokens.len());

            run_inference_pipeline(&mut engine, &tokens, &tok, &SamplingConfig::default(), 128);
        }
        Err(e) => eprintln!("[api] Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Command: file
// ---------------------------------------------------------------------------
fn cmd_file(args: &[String]) {
    if args.len() < 4 {
        eprintln!("[error] Usage: file <model.myai> <PATH>");
        return;
    }
    let model_path = PathBuf::from(&args[2]);
    let file_path = &args[3];

    let mut engine = load_engine(&model_path);
    let tok = load_tokenizer(&model_path, engine.config.vocab_size);
    let crawler = Crawler::new();

    println!("[file] Reading: {}", file_path);
    match crawler.read_file(file_path) {
        Ok(result) => {
            println!(
                "[file] Size: {} bytes | Time: {} ms",
                result.size_bytes, result.elapsed_ms
            );
            let preview: String = result.text.chars().take(500).collect();
            println!("\n--- Content ---\n{}", preview);
            if result.text.len() > 500 {
                println!("... (truncated)");
            }

            let tokens = tok.encode(&result.text);
            println!("\n[file] Tokenized: {} tokens", tokens.len());

            run_inference_pipeline(&mut engine, &tokens, &tok, &SamplingConfig::default(), 128);
        }
        Err(e) => eprintln!("[file] Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Command: sys
// ---------------------------------------------------------------------------
fn cmd_sys(args: &[String]) {
    if args.len() < 4 {
        eprintln!("[error] Usage: sys <model.myai> <COMMAND>");
        return;
    }
    let model_path = PathBuf::from(&args[2]);
    let cmd = args[3..].join(" ");

    let mut engine = load_engine(&model_path);
    let tok = load_tokenizer(&model_path, engine.config.vocab_size);
    let crawler = Crawler::new();

    println!("[sys] Running: {}", cmd);
    match crawler.run_command(&cmd) {
        Ok(result) => {
            println!(
                "[sys] Exit: {} | Size: {} bytes | Time: {} ms",
                result.status, result.size_bytes, result.elapsed_ms
            );
            println!("\n--- Output ---\n{}", result.text);

            let tokens = tok.encode(&result.text);
            println!("[sys] Tokenized: {} tokens", tokens.len());

            run_inference_pipeline(&mut engine, &tokens, &tok, &SamplingConfig::default(), 128);
        }
        Err(e) => eprintln!("[sys] Error: {}", e),
    }
}

// ---------------------------------------------------------------------------
// Command: gen (generate from text)
// ---------------------------------------------------------------------------
fn cmd_gen(args: &[String]) {
    if args.len() < 4 {
        eprintln!("[error] Usage: gen <model.myai> <TEXT> [options]");
        eprintln!("  Options:");
        eprintln!("    -t <float>    Temperature (default: 0.7, 0=greedy)");
        eprintln!("    -p <float>    Top-p nucleus (default: 0.9)");
        eprintln!("    -k <int>      Top-k (if set, uses top-k instead of top-p)");
        eprintln!("    -n <int>      Max tokens to generate (default: 256)");
        eprintln!("    -r <float>    Repetition penalty (default: 1.15)");
        eprintln!("    --raw         Skip chat template, use raw text");
        return;
    }
    let model_path = PathBuf::from(&args[2]);

    // Separate text from flags: collect everything that doesn't start with '-'
    // until we hit a flag, then parse flags
    let mut text_parts: Vec<String> = Vec::new();
    let mut sampling = SamplingConfig::default();
    let mut max_tokens: usize = 256;
    let mut raw_mode = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-t" | "--temperature" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        sampling.temperature = v;
                        if v == 0.0 { sampling.strategy = SamplingStrategy::Greedy; }
                    }
                    i += 2;
                } else { i += 1; }
            }
            "-p" | "--top-p" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        sampling.top_p = v;
                        sampling.strategy = SamplingStrategy::TopP;
                    }
                    i += 2;
                } else { i += 1; }
            }
            "-k" | "--top-k" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        sampling.top_k = v;
                        sampling.strategy = SamplingStrategy::TopK;
                    }
                    i += 2;
                } else { i += 1; }
            }
            "-n" | "--max-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { max_tokens = v; }
                    i += 2;
                } else { i += 1; }
            }
            "-r" | "--rep-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() { sampling.repetition_penalty = v; }
                    i += 2;
                } else { i += 1; }
            }
            "--raw" => {
                raw_mode = true;
                i += 1;
            }
            _ => {
                text_parts.push(args[i].clone());
                i += 1;
            }
        }
    }

    let text = text_parts.join(" ");
    if text.is_empty() {
        eprintln!("[error] No input text provided");
        return;
    }

    let mut engine = load_engine(&model_path);
    let tok = load_tokenizer(&model_path, engine.config.vocab_size);

    // Discover stop tokens from tokenizer vocab
    let mut stop_tokens = Vec::new();
    for special in &["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
        if let Some(id) = tok.find_token_id(special) {
            stop_tokens.push(id as usize);
            println!("[gen] Stop token: '{}' → ID {}", special, id);
        }
    }
    sampling.stop_tokens = stop_tokens;

    // Format with Qwen chat template (unless --raw)
    let prompt = if raw_mode {
        text.clone()
    } else {
        format_qwen_chat(&text)
    };

    println!("[gen] Input: \"{}\"", &text);
    if !raw_mode {
        println!("[gen] Using Qwen chat template");
    }
    println!("[gen] Sampling: {:?}, temp={:.2}, top_p={:.2}, rep_penalty={:.2}",
        sampling.strategy, sampling.temperature, sampling.top_p, sampling.repetition_penalty);

    let tokens = tok.encode(&prompt);
    println!("[gen] Tokenized: {} tokens → {:?}...", tokens.len(), &tokens[..tokens.len().min(10)]);

    run_inference_pipeline(&mut engine, &tokens, &tok, &sampling, max_tokens);
}

// ---------------------------------------------------------------------------
// Command: train-tok (train BPE tokenizer)
// ---------------------------------------------------------------------------
fn cmd_train_tok(args: &[String]) {
    if args.len() < 3 {
        eprintln!("[error] Usage: train-tok <corpus_file> [num_merges]");
        return;
    }
    let corpus_path = &args[2];
    let num_merges: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4000);

    println!("[train-tok] Loading corpus: {}", corpus_path);
    let corpus = match std::fs::read_to_string(corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[error] Failed to read corpus: {}", e);
            return;
        }
    };

    println!(
        "[train-tok] Corpus: {} chars, training {} merges...",
        corpus.len(),
        num_merges
    );

    let mut tok = Tokenizer::new(8192);
    let t0 = Instant::now();
    tok.train(&corpus, num_merges);
    let elapsed = t0.elapsed();

    let out_path = Path::new("tokenizer.mytok");
    tok.save(out_path).expect("Failed to save tokenizer");

    println!(
        "[train-tok] Done in {:.1}s — {} merges learned",
        elapsed.as_secs_f64(),
        tok.num_merges()
    );
    println!("[train-tok] Saved to: {}", out_path.display());

    // Test encode/decode
    let sample = &corpus[..corpus.len().min(200)];
    let ids = tok.encode(sample);
    let ratio = sample.len() as f64 / ids.len() as f64;
    println!(
        "[train-tok] Compression: {} bytes → {} tokens ({:.1}x ratio)",
        sample.len(),
        ids.len(),
        ratio
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> ModelConfig {
    ModelConfig {
        dim: 512,
        hidden_dim: 1536,
        n_layers: 12,
        n_heads: 8,
        head_dim: 64,
        vocab_size: 32000,
        max_seq_len: MAX_SEQ_LEN,
        is_quantized: true,
        is_bitnet: false,
        n_kv_heads: 8,  // Same as n_heads for MHA
        n_experts: 8,  // 8 experts = 2× specialization capacity, same runtime compute (top-2)
        top_k: 2,
        int4_group_size: 64,
        depth_router_layer: DEPTH_ROUTER_AFTER_LAYER,
        tri_layer_mode: false,  // NLLM tri-layer dense mode (disabled by default)
        speculative_steps: 0,
        rope_theta: 10000.0,
    }
}

fn get_model_path(args: &[String], index: usize) -> Option<PathBuf> {
    args.get(index).map(PathBuf::from)
}

/// Try to load a HuggingFace tokenizer (.mytok) adjacent to the model file.
/// Falls back to byte-level tokenizer if not found.
fn load_tokenizer(model_path: &Path, vocab_size: usize) -> Tokenizer {
    // Try <model_name>.mytok
    let mytok_path = model_path.with_extension("mytok");
    if mytok_path.exists() {
        match Tokenizer::load_hf_vocab(&mytok_path) {
            Ok(tok) => {
                println!("[tokenizer] Loaded HF tokenizer from {:?}", mytok_path);
                return tok;
            }
            Err(e) => {
                eprintln!("[tokenizer] Warning: Failed to load {:?}: {}", mytok_path, e);
            }
        }
    }

    // Try tokenizer.mytok in same directory
    if let Some(parent) = model_path.parent() {
        let alt_path = parent.join("tokenizer.mytok");
        if alt_path.exists() {
            match Tokenizer::load_hf_vocab(&alt_path) {
                Ok(tok) => {
                    println!("[tokenizer] Loaded HF tokenizer from {:?}", alt_path);
                    return tok;
                }
                Err(e) => {
                    eprintln!("[tokenizer] Warning: Failed to load {:?}: {}", alt_path, e);
                }
            }
        }
    }

    println!("[tokenizer] Using byte-level fallback (no .mytok file found)");
    Tokenizer::new(vocab_size)
}

fn load_engine(path: &PathBuf) -> Engine {
    println!("[engine] Loading: {:?}", path);
    let t0 = Instant::now();
    let engine = Engine::load(path).unwrap_or_else(|e| {
        eprintln!("[error] Failed to load model: {}", e);
        std::process::exit(1);
    });
    println!(
        "[engine] Ready in {:.1} ms\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    engine
}

fn print_memory_report(engine: &Engine) {
    let backbone_mb = engine.weights.total_bytes as f64 / (1024.0 * 1024.0);
    let expert_mb = engine.weights.expert_weight_bytes() as f64 / (1024.0 * 1024.0);
    let dense_mb = engine.weights.dense_weight_bytes() as f64 / (1024.0 * 1024.0);
    let scratch_mb = engine.state.memory_bytes() as f64 / (1024.0 * 1024.0);
    let crawler_kb = Crawler::new().memory_bytes() as f64 / 1024.0;
    let tok_kb = Tokenizer::new(engine.config.vocab_size).memory_bytes() as f64 / 1024.0;
    let param_m = engine.config.param_count() as f64 / 1e6;

    // Estimate active RAM: dense weights + top_k/n_experts fraction of expert weights + scratchpad
    let expert_active_frac = if engine.config.n_experts > 0 {
        engine.config.top_k as f64 / engine.config.n_experts as f64
    } else { 1.0 };
    let active_expert_mb = expert_mb * expert_active_frac;
    let active_total = dense_mb + active_expert_mb + scratch_mb + (crawler_kb + tok_kb) / 1024.0;

    println!("+-------------------------------------------------+");
    println!("|       MEMORY REPORT  (Sparse MoE v2)            |");
    println!("+-------------------------------------------------+");
    println!("| Model: ~{:.1}M params                             |", param_m);
    println!("|   dim={}, hidden={}, layers={}               |", engine.config.dim, engine.config.hidden_dim, engine.config.n_layers);
    println!("|   heads={}, vocab={}, seq={}              |", engine.config.n_heads, engine.config.vocab_size, engine.config.max_seq_len);
    println!("|   experts={} × top-{}, int4_group={}            |", engine.config.n_experts, engine.config.top_k, engine.config.int4_group_size);
    if engine.config.depth_router_layer > 0 {
        println!("|   adaptive depth: router after layer {}          |", engine.config.depth_router_layer);
    }
    println!("+-------------------------------------------------+");
    println!("| Backbone (disk)    : {:>8.2} MB                |", backbone_mb);
    println!("|   Dense (always)   : {:>8.2} MB                |", dense_mb);
    println!("|   Expert (Int4)    : {:>8.2} MB ({:.0}% of file)  |", expert_mb, expert_mb / backbone_mb * 100.0);
    println!("| Scratchpad (heap)  : {:>8.2} MB                |", scratch_mb);
    println!("| Tokenizer          : {:>8.1} KB                |", tok_kb);
    println!("| Crawler            : {:>8.1} KB                |", crawler_kb);
    println!("+-------------------------------------------------+");
    println!("| Active RAM estimate: {:>8.2} MB                |", active_total);
    println!("|   (top-{}/{} experts, mmap sparse loading)      |", engine.config.top_k, engine.config.n_experts);
    println!("+-------------------------------------------------+\n");
}

/// Run raw inference: feed seed token and generate `gen_len` tokens.
fn run_inference(engine: &mut Engine, seed_token: usize, gen_len: usize) -> (Vec<usize>, f64) {
    let t0 = Instant::now();
    let mut token = seed_token;
    let mut tokens = Vec::with_capacity(gen_len);

    for pos in 0..gen_len {
        token = engine.forward(token, pos);
        tokens.push(token);
    }

    (tokens, t0.elapsed().as_secs_f64())
}

/// Format text using Qwen chat template.
fn format_qwen_chat(user_input: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        user_input
    )
}

/// Pipeline: take pre-tokenized input, run inference with smart sampling, stream output.
fn run_inference_pipeline(
    engine: &mut Engine,
    input_tokens: &[u32],
    tok: &Tokenizer,
    sampling: &SamplingConfig,
    max_gen_tokens: usize,
) {
    let vocab = engine.config.vocab_size;
    let max_seq = engine.config.max_seq_len;

    // Convert to usize and clamp to vocab range
    let tokens: Vec<usize> = input_tokens.iter()
        .map(|&t| (t as usize).min(vocab - 1))
        .collect();

    // Phase 1: PREFILL — process all input tokens to build KV cache
    let gen_budget = max_gen_tokens.min(max_seq.saturating_sub(tokens.len()));
    let n_prefill = tokens.len().min(max_seq.saturating_sub(gen_budget).max(1));

    println!(
        "\n[inference] Prefilling {} tokens (MoE top-{}/{})...",
        n_prefill, engine.config.top_k, engine.config.n_experts
    );

    let t0 = Instant::now();
    for (pos, &tok_id) in tokens[..n_prefill].iter().enumerate() {
        engine.forward(tok_id, pos);
    }
    let prefill_time = t0.elapsed();
    println!(
        "[inference] Prefill: {:.1} ms ({:.1} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        n_prefill as f64 / prefill_time.as_secs_f64().max(0.001)
    );

    // Phase 2: GENERATE — autoregressive decoding with streaming output
    let mut token_history = tokens.clone();
    let mut next_token = engine.sample_from_logits(sampling, &token_history);

    print!("\n");
    let _ = std::io::stdout().flush();

    let t1 = Instant::now();
    let mut gen_count = 0;

    for step in 0..gen_budget {
        let pos = n_prefill + step;
        if pos >= max_seq {
            break;
        }

        // Check for stop tokens
        if sampling.stop_tokens.contains(&next_token) {
            break;
        }

        // Decode and stream this token
        let token_str = tok.decode_token(next_token as u32);
        print!("{}", token_str);
        let _ = std::io::stdout().flush();

        token_history.push(next_token);
        gen_count += 1;

        // Forward pass for next token
        engine.forward(next_token, pos);
        next_token = engine.sample_from_logits(sampling, &token_history);
    }

    let gen_time = t1.elapsed();
    println!(
        "\n\n[inference] Generated {} tokens in {:.1} ms ({:.1} tok/s)",
        gen_count,
        gen_time.as_secs_f64() * 1000.0,
        gen_count as f64 / gen_time.as_secs_f64().max(0.001)
    );
    println!("[inference] {}", engine.sparse_stats_report());
    println!();
}
