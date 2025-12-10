import requests
import pytest

BASE = "http://127.0.0.1:5000/talk"
SID  = "edge_cases"

edge_cases = [
    "",
    "     ",
    ".........",
    "uhhhh",
    "???",
    "help??",
]

@pytest.mark.parametrize("text", edge_cases)
def test_edge_cases_do_not_break(text):
    r = requests.post(BASE, json={
        "text": text,
        "language": "en",
        "session_id": SID,
        "user_name": "Lilly"
    })
    reply = r.json()["reply"]

    # Björn should not hallucinate or get confused
    assert isinstance(reply, str)
    assert len(reply) > 0
