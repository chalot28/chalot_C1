// =============================================================================
// crawler.rs — Multi-source data crawler (Web, API, File, System)
// =============================================================================
//
// Fetches data from diverse sources and extracts clean text for the AI model.
// Uses `ureq` (sync HTTP) — no async runtime overhead.
//
// Memory budget: ~1-2 MB max per crawl (configurable max_body_size).
// =============================================================================

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Describes where to crawl data from.
#[allow(dead_code)]
pub enum CrawlSource {
    /// Fetch a web page via HTTP GET.
    Web(String),
    /// Call an API endpoint with custom method/headers/body.
    Api {
        url: String,
        method: String,
        body: Option<String>,
        headers: HashMap<String, String>,
    },
    /// Read a local file.
    File(String),
    /// Execute a system command and capture stdout.
    System(String),
}

/// Result of a crawl operation.
pub struct CrawlResult {
    /// Source identifier (URL, file path, command).
    pub source: String,
    /// Raw response body.
    #[allow(dead_code)]
    pub raw: String,
    /// Extracted clean text (HTML tags stripped, JSON flattened, etc.).
    pub text: String,
    /// HTTP status code (0 for non-HTTP sources).
    pub status: u16,
    /// Time taken in milliseconds.
    pub elapsed_ms: u64,
    /// Response size in bytes.
    pub size_bytes: usize,
}

/// Crawler configuration and state.
pub struct Crawler {
    /// User-Agent header for HTTP requests.
    pub user_agent: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum response body size in bytes (prevents OOM).
    pub max_body_size: usize,
    /// Minimum delay between requests (rate limiting) in ms.
    pub rate_limit_ms: u64,
    /// Timestamp of last request (for rate limiting).
    last_request: Option<Instant>,
}

impl Crawler {
    /// Create a new crawler with sensible defaults.
    pub fn new() -> Self {
        Crawler {
            user_agent: "AI_chalot_C1/0.2 (Rust AI Engine)".to_string(),
            timeout_secs: 15,
            max_body_size: 1024 * 1024, // 1 MB max body
            rate_limit_ms: 200,          // 5 req/sec max
            last_request: None,
        }
    }

    /// Create a crawler with custom limits.
    #[allow(dead_code)]
    pub fn with_limits(timeout_secs: u64, max_body_kb: usize, rate_limit_ms: u64) -> Self {
        Crawler {
            timeout_secs,
            max_body_size: max_body_kb * 1024,
            rate_limit_ms,
            ..Self::new()
        }
    }

    // -- Rate limiting ------------------------------------------------------

    fn wait_rate_limit(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed().as_millis() as u64;
            if elapsed < self.rate_limit_ms {
                std::thread::sleep(Duration::from_millis(self.rate_limit_ms - elapsed));
            }
        }
        self.last_request = Some(Instant::now());
    }

    // -- Unified fetch ------------------------------------------------------

    /// Fetch data from any source.
    #[allow(dead_code)]
    pub fn fetch(&mut self, source: CrawlSource) -> Result<CrawlResult, String> {
        match source {
            CrawlSource::Web(url) => self.fetch_url(&url),
            CrawlSource::Api { url, method, body, headers } => {
                let hdr_vec: Vec<(&str, &str)> = headers
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                self.fetch_api(&url, &method, body.as_deref(), &hdr_vec)
            }
            CrawlSource::File(path) => self.read_file(&path),
            CrawlSource::System(cmd) => self.run_command(&cmd),
        }
    }

    // -- HTTP GET -----------------------------------------------------------

    /// Fetch a URL via HTTP GET, extract text from HTML.
    pub fn fetch_url(&mut self, url: &str) -> Result<CrawlResult, String> {
        self.wait_rate_limit();
        let t0 = Instant::now();

        let resp = ureq::get(url)
            .set("User-Agent", &self.user_agent)
            .timeout(Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| format!("HTTP GET failed: {}", e))?;

        let status = resp.status();
        let raw = self.read_body(resp)?;
        let size = raw.len();
        let text = Self::extract_text_auto(&raw);

        Ok(CrawlResult {
            source: url.to_string(),
            raw,
            text,
            status,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            size_bytes: size,
        })
    }

    // -- HTTP API call ------------------------------------------------------

    /// Call an API endpoint with custom method, body, headers.
    pub fn fetch_api(
        &mut self,
        url: &str,
        method: &str,
        body: Option<&str>,
        headers: &[(&str, &str)],
    ) -> Result<CrawlResult, String> {
        self.wait_rate_limit();
        let t0 = Instant::now();

        let mut req = match method.to_uppercase().as_str() {
            "GET" => ureq::get(url),
            "POST" => ureq::post(url),
            "PUT" => ureq::put(url),
            "DELETE" => ureq::delete(url),
            "PATCH" => ureq::patch(url),
            other => return Err(format!("Unsupported HTTP method: {}", other)),
        };

        req = req
            .set("User-Agent", &self.user_agent)
            .timeout(Duration::from_secs(self.timeout_secs));

        for &(k, v) in headers {
            req = req.set(k, v);
        }

        let resp = match body {
            Some(b) => req
                .set("Content-Type", "application/json")
                .send_string(b)
                .map_err(|e| format!("API call failed: {}", e))?,
            None => req.call().map_err(|e| format!("API call failed: {}", e))?,
        };

        let status = resp.status();
        let raw = self.read_body(resp)?;
        let size = raw.len();

        // For API responses, try JSON extraction first
        let text = if raw.trim_start().starts_with('{') || raw.trim_start().starts_with('[') {
            Self::flatten_json(&raw)
        } else {
            Self::extract_text_auto(&raw)
        };

        Ok(CrawlResult {
            source: format!("{} {}", method.to_uppercase(), url),
            raw,
            text,
            status,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            size_bytes: size,
        })
    }

    // -- File reading -------------------------------------------------------

    /// Read a local file and extract text.
    pub fn read_file(&self, path: &str) -> Result<CrawlResult, String> {
        let t0 = Instant::now();
        let p = Path::new(path);

        if !p.exists() {
            return Err(format!("File not found: {}", path));
        }

        let metadata = std::fs::metadata(p).map_err(|e| format!("metadata: {}", e))?;
        if metadata.len() as usize > self.max_body_size {
            return Err(format!(
                "File too large: {} bytes (max: {})",
                metadata.len(),
                self.max_body_size
            ));
        }

        let raw = std::fs::read_to_string(p)
            .or_else(|_| {
                // Binary file — read as lossy UTF-8
                let bytes = std::fs::read(p).map_err(|e| format!("read: {}", e))?;
                Ok::<String, String>(String::from_utf8_lossy(&bytes).into_owned())
            })
            .map_err(|e| format!("read file: {}", e))?;

        let size = raw.len();
        let text = Self::extract_text_auto(&raw);

        Ok(CrawlResult {
            source: path.to_string(),
            raw,
            text,
            status: 0,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            size_bytes: size,
        })
    }

    // -- System command execution -------------------------------------------

    /// Run a system command and capture stdout.
    pub fn run_command(&self, cmd: &str) -> Result<CrawlResult, String> {
        let t0 = Instant::now();

        let output = if cfg!(target_os = "windows") {
            std::process::Command::new("cmd")
                .args(["/C", cmd])
                .output()
        } else {
            std::process::Command::new("sh")
                .args(["-c", cmd])
                .output()
        };

        let output = output.map_err(|e| format!("Failed to execute command: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        let raw = if stderr.is_empty() {
            stdout.clone()
        } else {
            format!("{}\n[STDERR]\n{}", stdout, stderr)
        };

        let size = raw.len();
        let status = output.status.code().unwrap_or(-1) as u16;

        Ok(CrawlResult {
            source: format!("cmd: {}", cmd),
            raw: raw.clone(),
            text: raw,
            status,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            size_bytes: size,
        })
    }

    // -- Body reading with size limit ---------------------------------------

    fn read_body(&self, resp: ureq::Response) -> Result<String, String> {
        let reader = resp.into_reader();
        let mut buf = Vec::with_capacity(self.max_body_size.min(64 * 1024));
        reader
            .take(self.max_body_size as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("read body: {}", e))?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    // -- Text extraction (static methods) -----------------------------------

    /// Auto-detect content type and extract text.
    fn extract_text_auto(raw: &str) -> String {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("<!") || trimmed.starts_with("<html") || trimmed.starts_with("<HTML")
            || trimmed.starts_with("<head") || trimmed.starts_with("<body")
        {
            Self::strip_html(raw)
        } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
            Self::flatten_json(raw)
        } else {
            // Plain text — just trim excessive whitespace
            Self::normalize_whitespace(raw)
        }
    }

    /// Strip HTML tags and extract readable text.
    /// Simple state-machine approach — no regex, no external deps.
    pub fn strip_html(html: &str) -> String {
        let mut result = String::with_capacity(html.len() / 3);
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;
        let mut last_was_space = false;
        let mut tag_name = String::new();

        let chars: Vec<char> = html.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            if c == '<' {
                in_tag = true;
                tag_name.clear();
                i += 1;
                continue;
            }

            if in_tag {
                if c == '>' {
                    in_tag = false;
                    let tag_lower = tag_name.to_lowercase();
                    if tag_lower.starts_with("script") {
                        in_script = true;
                    } else if tag_lower.starts_with("/script") {
                        in_script = false;
                    } else if tag_lower.starts_with("style") {
                        in_style = true;
                    } else if tag_lower.starts_with("/style") {
                        in_style = false;
                    }
                    // Block elements → insert newline
                    if matches!(
                        tag_lower.trim_start_matches('/').split_whitespace().next(),
                        Some("p" | "div" | "br" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                            | "li" | "tr" | "td" | "th" | "blockquote" | "pre" | "hr"
                            | "section" | "article" | "header" | "footer" | "nav")
                    ) {
                        if !result.ends_with('\n') {
                            result.push('\n');
                        }
                        last_was_space = true;
                    }
                } else {
                    tag_name.push(c);
                }
                i += 1;
                continue;
            }

            if in_script || in_style {
                i += 1;
                continue;
            }

            // Decode common HTML entities
            if c == '&' {
                let rest: String = chars[i..].iter().take(10).collect();
                if rest.starts_with("&amp;") {
                    result.push('&');
                    i += 5;
                    last_was_space = false;
                    continue;
                } else if rest.starts_with("&lt;") {
                    result.push('<');
                    i += 4;
                    last_was_space = false;
                    continue;
                } else if rest.starts_with("&gt;") {
                    result.push('>');
                    i += 4;
                    last_was_space = false;
                    continue;
                } else if rest.starts_with("&quot;") {
                    result.push('"');
                    i += 6;
                    last_was_space = false;
                    continue;
                } else if rest.starts_with("&nbsp;") {
                    result.push(' ');
                    i += 6;
                    last_was_space = true;
                    continue;
                } else if rest.starts_with("&#") {
                    // Numeric entity
                    if let Some(semi_pos) = rest.find(';') {
                        let num_str = &rest[2..semi_pos];
                        let code = if num_str.starts_with('x') || num_str.starts_with('X') {
                            u32::from_str_radix(&num_str[1..], 16).ok()
                        } else {
                            num_str.parse::<u32>().ok()
                        };
                        if let Some(ch) = code.and_then(char::from_u32) {
                            result.push(ch);
                            i += semi_pos + 1;
                            last_was_space = false;
                            continue;
                        }
                    }
                }
            }

            // Normal character
            if c.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(c);
                last_was_space = false;
            }
            i += 1;
        }

        Self::normalize_whitespace(&result)
    }

    /// Flatten JSON into readable key=value text.
    pub fn flatten_json(json_str: &str) -> String {
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(val) => {
                let mut out = String::with_capacity(json_str.len());
                Self::flatten_value(&val, "", &mut out);
                out
            }
            Err(_) => json_str.to_string(),
        }
    }

    fn flatten_value(val: &serde_json::Value, prefix: &str, out: &mut String) {
        match val {
            serde_json::Value::Object(map) => {
                for (key, v) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };
                    Self::flatten_value(v, &path, out);
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    let path = format!("{}[{}]", prefix, i);
                    Self::flatten_value(v, &path, out);
                }
            }
            serde_json::Value::String(s) => {
                out.push_str(prefix);
                out.push_str(" = ");
                out.push_str(s);
                out.push('\n');
            }
            serde_json::Value::Number(n) => {
                out.push_str(prefix);
                out.push_str(" = ");
                out.push_str(&n.to_string());
                out.push('\n');
            }
            serde_json::Value::Bool(b) => {
                out.push_str(prefix);
                out.push_str(" = ");
                out.push_str(if *b { "true" } else { "false" });
                out.push('\n');
            }
            serde_json::Value::Null => {
                out.push_str(prefix);
                out.push_str(" = null\n");
            }
        }
    }

    /// Extract a specific path from JSON (e.g., "data.items[0].name").
    #[allow(dead_code)]
    pub fn json_extract(json_str: &str, path: &str) -> Option<String> {
        let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let mut current = &val;

        for part in path.split('.') {
            // Check for array index: "items[0]"
            if let Some(bracket_pos) = part.find('[') {
                let key = &part[..bracket_pos];
                let idx_str = &part[bracket_pos + 1..part.len() - 1];
                let idx: usize = idx_str.parse().ok()?;

                if !key.is_empty() {
                    current = current.get(key)?;
                }
                current = current.get(idx)?;
            } else {
                current = current.get(part)?;
            }
        }

        Some(match current {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }

    /// Normalize whitespace: collapse runs, trim lines.
    fn normalize_whitespace(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut prev_empty = false;

        for line in s.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !prev_empty && !result.is_empty() {
                    result.push('\n');
                    prev_empty = true;
                }
            } else {
                // Collapse internal whitespace
                let mut last_space = false;
                for c in trimmed.chars() {
                    if c.is_whitespace() {
                        if !last_space {
                            result.push(' ');
                            last_space = true;
                        }
                    } else {
                        result.push(c);
                        last_space = false;
                    }
                }
                result.push('\n');
                prev_empty = false;
            }
        }

        result.trim().to_string()
    }

    // -- Batch crawling -----------------------------------------------------

    /// Crawl multiple URLs and return results.
    #[allow(dead_code)]
    pub fn crawl_urls(&mut self, urls: &[&str]) -> Vec<Result<CrawlResult, String>> {
        urls.iter().map(|url| self.fetch_url(url)).collect()
    }

    // -- System info gathering ----------------------------------------------

    /// Gather basic system information.
    pub fn system_info(&self) -> Result<CrawlResult, String> {
        let t0 = Instant::now();
        let mut info = String::new();

        info.push_str(&format!("OS: {}\n", std::env::consts::OS));
        info.push_str(&format!("Arch: {}\n", std::env::consts::ARCH));

        if let Ok(cwd) = std::env::current_dir() {
            info.push_str(&format!("CWD: {}\n", cwd.display()));
        }

        // Environment variables of interest
        for var in &["PATH", "HOME", "USERPROFILE", "COMPUTERNAME", "HOSTNAME"] {
            if let Ok(val) = std::env::var(var) {
                info.push_str(&format!("{}={}\n", var, val));
            }
        }

        // Disk and time
        info.push_str(&format!(
            "Time: {:?}\n",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));

        let size = info.len();
        Ok(CrawlResult {
            source: "system_info".to_string(),
            raw: info.clone(),
            text: info,
            status: 0,
            elapsed_ms: t0.elapsed().as_millis() as u64,
            size_bytes: size,
        })
    }

    /// Estimated memory usage of the crawler itself.
    pub fn memory_bytes(&self) -> usize {
        self.user_agent.len() + 64 // struct fields + user agent string
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        let html = r#"<html><head><title>Test</title></head>
            <body><h1>Hello</h1><p>World &amp; friends</p>
            <script>var x = 1;</script></body></html>"#;
        let text = Crawler::strip_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World & friends"));
        assert!(!text.contains("var x"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_flatten_json() {
        let json = r#"{"name":"Alice","age":30,"items":[1,2,3]}"#;
        let flat = Crawler::flatten_json(json);
        assert!(flat.contains("name = Alice"));
        assert!(flat.contains("age = 30"));
        assert!(flat.contains("items[0] = 1"));
    }

    #[test]
    fn test_json_extract() {
        let json = r#"{"data":{"items":[{"name":"first"},{"name":"second"}]}}"#;
        let result = Crawler::json_extract(json, "data.items[0].name");
        assert_eq!(result, Some("first".to_string()));
    }

    #[test]
    fn test_read_file() {
        let crawler = Crawler::new();
        let result = crawler.read_file("Cargo.toml");
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.text.contains("AI_chalot_C1"));
    }

    #[test]
    fn test_system_info() {
        let crawler = Crawler::new();
        let result = crawler.system_info();
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.text.contains("OS:"));
    }

    #[test]
    fn test_run_command() {
        let crawler = Crawler::new();
        let cmd = if cfg!(target_os = "windows") {
            "echo hello"
        } else {
            "echo hello"
        };
        let result = crawler.run_command(cmd);
        assert!(result.is_ok());
        assert!(result.unwrap().text.contains("hello"));
    }

    #[test]
    fn test_normalize_whitespace() {
        let messy = "  hello   world  \n\n\n  foo   bar  ";
        let clean = Crawler::normalize_whitespace(messy);
        assert_eq!(clean, "hello world\n\nfoo bar");
    }
}
