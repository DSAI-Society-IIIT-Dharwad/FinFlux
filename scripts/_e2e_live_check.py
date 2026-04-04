import json
import uuid

import requests

base = "http://127.0.0.1:8000"
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
