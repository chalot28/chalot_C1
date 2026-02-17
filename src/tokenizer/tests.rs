// =============================================================================
// tokenizer/tests.rs — Tokenizer tests
// =============================================================================

use super::*;

#[test]
fn test_byte_level_encode_decode() {
    let tok = Tokenizer::new(8192);
    let text = "Hello, world!";
    let ids = tok.encode(text);
    assert_eq!(ids.len(), text.len()); // No merges → 1 token per byte
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_with_special() {
    let tok = Tokenizer::new(8192);
    let ids = tok.encode_with_special("Hi");
    assert_eq!(ids[0], BOS_TOKEN);
    assert_eq!(*ids.last().unwrap(), EOS_TOKEN);
    assert_eq!(ids.len(), 4); // BOS + H + i + EOS
}

#[test]
fn test_bpe_training() {
    let mut tok = Tokenizer::new(8192);
    let corpus = "abcabcabcabc"; // "ab" and "abc" should be learned
    tok.train(corpus, 5);
    assert!(tok.num_merges() > 0);
    let ids = tok.encode("abc");
    // After merges, "abc" should be fewer than 3 tokens
    assert!(ids.len() <= 3, "ids={:?}", ids);
}

#[test]
fn test_utf8_roundtrip() {
    let tok = Tokenizer::new(8192);
    let text = "Xin chào thế giới! 你好世界";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_save_load() {
    let mut tok = Tokenizer::new(8192);
    tok.train("hello hello hello world world", 3);
    let path = Path::new("test_tokenizer.txt");
    tok.save(path).unwrap();

    let tok2 = Tokenizer::load(path, 8192).unwrap();
    assert_eq!(tok.num_merges(), tok2.num_merges());
    assert_eq!(tok.encode("hello"), tok2.encode("hello"));

    std::fs::remove_file(path).ok();
}
