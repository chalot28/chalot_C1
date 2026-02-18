// =============================================================================
// tokenizer/tests.rs — Tokenizer unit tests
// =============================================================================

#[allow(unused_imports)]
use super::core::Tokenizer;

#[test]
fn test_byte_level_encode_decode() {
    let tok = Tokenizer::new(512);
    let text = "Hello World!";
    let ids = tok.encode(text);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_empty_string() {
    let tok = Tokenizer::new(512);
    let ids = tok.encode("");
    assert!(ids.is_empty());
    let decoded = tok.decode(&ids);
    assert!(decoded.is_empty());
}
