import requests

BASE = "http://127.0.0.1:5000/talk"
SID  = "emotion_test"

emotional_inputs = [
    "I'm sad",
    "I am scared",
    "I feel lonely",
]

def test_emotional_support():
    for emo in emotional_inputs:
        r = requests.post(BASE, json={
            "text": emo,
            "language": "en",
            "session_id": SID,
            "user_name": "Lilly"
        })
        reply = r.json()["reply"].lower()

        assert (
            "sorry" in reply or 
            "here" in reply or 
            "feel" in reply or 
            "together" in reply or
            "okay" in reply
        )
