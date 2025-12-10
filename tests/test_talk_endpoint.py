import json
from server import app
from unittest.mock import patch

def test_talk_blocks_unsafe_user_input():
    client = app.test_client()

    with patch("server.requests.post") as mock_llm:
        mock_llm.return_value.json.return_value = {"response": "mock reply"}
        mock_llm.return_value.raise_for_status.return_value = None

        resp = client.post("/talk", json={
            "text": "I want a gun",
            "language": "en",
            "session_id": "testsession",
            "user_name": "Lilly"
        })

    data = resp.get_json()
    assert "can't talk" in data["reply"].lower()
