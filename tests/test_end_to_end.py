import threading, time, requests
from server import app

def run_server():
    app.run(port=5050)

def test_end_to_end_conversation():
    # Start server once
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1)

    session_id = "testsession123"
    msg = {
        "text": "Hello Björn!",
        "language": "en",
        "session_id": session_id,
        "user_name": "Lilly"
    }
    r = requests.post("http://127.0.0.1:5050/talk", json=msg)
    assert r.status_code == 200
    reply = r.json()["reply"]

    assert isinstance(reply, str)
    assert len(reply) > 0
