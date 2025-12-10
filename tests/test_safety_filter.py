from server import evaluate_safety, BLOCKLIST

def test_safety_detects_blocked_words():
    for word in BLOCKLIST:
        result = evaluate_safety(f"This contains {word}.")
        assert not result["safe"]
        assert word in result["hits"]

def test_safety_passes_clean_text():
    result = evaluate_safety("I love cookies and bears.")
    assert result["safe"]
    assert result["hits"] == []
