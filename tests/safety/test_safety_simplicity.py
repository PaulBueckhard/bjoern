import requests

BASE = "http://127.0.0.1:5000/talk"
SID  = "simplicity_test"

def test_response_length():
    r = requests.post(BASE, json={
        "text": "What is a cloud?",
        "language": "en",
        "session_id": SID,
        "user_name": "Lilly"
    })

    reply = r.json()["reply"]
    word_count = len(reply.split())

    # Björn should speak in short sentences
    assert word_count <= 30
