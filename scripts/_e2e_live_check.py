import json
import uuid
import wave
import struct
from pathlib import Path

import requests

base = "http://127.0.0.1:8000"


def ensure_tiny_wav() -> Path:
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "data" / "demo" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wav_path = fixture_dir / "e2e_tiny_5s.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    sample_rate = 16000
    duration_s = 5
    total_frames = sample_rate * duration_s
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence = struct.pack("<h", 0)
        wf.writeframes(silence * total_frames)
    return wav_path

def main() -> None:
    email = f"e2e_{uuid.uuid4().hex[:8]}@gmail.com"
    password = "Test@12345"
    s = requests.Session()

    print("[1] signup")
    r = s.post(f"{base}/api/auth/signup", json={"username": email, "password": password}, timeout=40)
    print(r.status_code, r.text[:400])

    payload = r.json() if "application/json" in r.headers.get("content-type", "") else {}
    token = payload.get("access_token")

    if not token:
        print("[2] login fallback")
        r = s.post(f"{base}/api/auth/login", json={"username": email, "password": password}, timeout=40)
        print(r.status_code, r.text[:400])
        payload = r.json() if "application/json" in r.headers.get("content-type", "") else {}
        token = payload.get("access_token")

    print("token_present", bool(token))
    if not token:
        raise SystemExit(2)

    headers = {"Authorization": f"Bearer {token}"}

    print("[3] me")
    r = s.get(f"{base}/api/auth/me", headers=headers, timeout=40)
    print(r.status_code, r.text[:400])

    print("[4] chat")
    r = s.post(
        f"{base}/api/chat",
        headers=headers,
        json={"text": "I have debt pressure and salary comes monthly", "thread_id": ""},
        timeout=180,
    )
    print(r.status_code)
    print(r.text[:600])
    chat_data = r.json() if "application/json" in r.headers.get("content-type", "") else {}
    conversation_id = chat_data.get("conversation_id")
    thread_id = chat_data.get("chat_thread_id")

    print("[4a] analyze audio with tiny 5s wav")
    tiny_wav = ensure_tiny_wav()
    with open(tiny_wav, "rb") as af:
        r = s.post(
            f"{base}/api/analyze",
            headers=headers,
            files={"file": (tiny_wav.name, af, "audio/wav")},
            data={"thread_id": thread_id or ""},
            timeout=240,
        )
    print(r.status_code)
    print(r.text[:600])

    print("[5] results")
    r = s.get(f"{base}/api/results", headers=headers, timeout=40)
    print(r.status_code, r.text[:500])

    print("[6] threads")
    r = s.get(f"{base}/api/threads", headers=headers, timeout=40)
    print(r.status_code, r.text[:500])

    if thread_id:
        print("[7] thread messages")
        r = s.get(f"{base}/api/threads/{thread_id}/messages", headers=headers, timeout=40)
        print(r.status_code, r.text[:500])

    if conversation_id:
        print("[8] transcript update")
        r = s.put(
            f"{base}/api/conversations/{conversation_id}/transcript",
            headers={**headers, "Content-Type": "application/json"},
            data=json.dumps({"transcript": "Updated transcript for e2e flow", "reanalyze": False}),
            timeout=120,
        )
        print(r.status_code, r.text[:500])

    print("[9] purge")
    r = s.post(
        f"{base}/api/history/purge",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps({"confirm": True}),
        timeout=40,
    )
    print(r.status_code, r.text[:500])

    print("[10] results after purge")
    r = s.get(f"{base}/api/results", headers=headers, timeout=40)
    print(r.status_code, r.text[:500])


if __name__ == "__main__":
    main()
