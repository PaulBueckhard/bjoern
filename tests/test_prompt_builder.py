from server import build_prompt

def test_prompt_includes_history_and_user():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    prompt = build_prompt(history, "How are you?", "en", "Lilly")

    assert "User: Hi" in prompt
    assert "Assistant: Hello!" in prompt
    assert "User: How are you?" in prompt
