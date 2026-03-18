import json
import base64
import urllib.parse
import urllib.request
import urllib.error
import time
import hmac
import hashlib
import os

import firebase_admin
from firebase_admin import firestore
from firebase_functions import https_fn, options

# ---------------------------------------------------------------------------
# Init — all clients are lazy to avoid deployment timeouts
# ---------------------------------------------------------------------------
_db = None
_kms_client = None

def get_db():
    global _db
    if _db is None:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _db = firestore.client()
    return _db

def get_kms():
    global _kms_client
    if _kms_client is None:
        from google.cloud import kms
        _kms_client = kms.KeyManagementServiceClient()
    return _kms_client

PROJECT_ID   = "arul-health"
KMS_LOCATION = "us-central1"
KMS_KEYRING  = "arul-health-keyring"
KMS_KEY      = "gmail-token-key"

# Read from environment variables (set in functions/.env)
CLIENT_ID     = os.environ.get("GMAIL_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GMAIL_CLIENT_SECRET", "")
REDIRECT_URI  = os.environ.get("GMAIL_REDIRECT_URI", "")
STATE_SECRET  = os.environ.get("GMAIL_STATE_SECRET", "")

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_SEND_URL   = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_SCOPE      = "https://www.googleapis.com/auth/gmail.send openid email"

# ---------------------------------------------------------------------------
# KMS helpers
# ---------------------------------------------------------------------------
def _kms_key_name() -> str:
    return get_kms().crypto_key_path(PROJECT_ID, KMS_LOCATION, KMS_KEYRING, KMS_KEY)

def encrypt_token(plaintext: str) -> str:
    resp = get_kms().encrypt(
        request={"name": _kms_key_name(), "plaintext": plaintext.encode()}
    )
    return base64.b64encode(resp.ciphertext).decode()

def decrypt_token(ciphertext_b64: str) -> str:
    ciphertext = base64.b64decode(ciphertext_b64)
    resp = get_kms().decrypt(
        request={"name": _kms_key_name(), "ciphertext": ciphertext}
    )
    return resp.plaintext.decode()

# ---------------------------------------------------------------------------
# State token helpers (HMAC-SHA256, no external deps)
# ---------------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def make_state(patient_id: str) -> str:
    payload = _b64url(json.dumps({"pid": patient_id, "t": int(time.time())}).encode())
    sig = _b64url(
        hmac.new(STATE_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    )
    return f"{payload}.{sig}"

def verify_state(state: str) -> str:
    try:
        payload, sig = state.rsplit(".", 1)
    except ValueError:
        raise ValueError("Malformed state")
    expected_sig = _b64url(
        hmac.new(STATE_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("Invalid state signature")
    data = json.loads(base64.urlsafe_b64decode(payload + "=="))
    if time.time() - data["t"] > 3600:
        raise ValueError("State token expired")
    return data["pid"]

# ---------------------------------------------------------------------------
# Cloud Function 1: generate_gmail_oauth_url
# POST body: { "patientId": "..." }
# ---------------------------------------------------------------------------
@https_fn.on_request(
    region="us-central1",
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def generate_gmail_oauth_url(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)

    body = req.get_json(silent=True) or {}
    patient_id = body.get("patientId", "").strip()
    if not patient_id:
        return https_fn.Response(
            json.dumps({"error": "patientId required"}),
            status=400, content_type="application/json",
        )

    state = make_state(patient_id)
    params = urllib.parse.urlencode({
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         GMAIL_SCOPE,
        "access_type":   "offline",
        "prompt":        "consent",
        "state":         state,
    })
    url = f"{GOOGLE_AUTH_URL}?{params}"

    return https_fn.Response(
        json.dumps({"url": url}),
        status=200, content_type="application/json",
    )

# ---------------------------------------------------------------------------
# Cloud Function 2: gmail_oauth_callback
# Google redirects here after patient consents.
# GET ?code=...&state=...
# ---------------------------------------------------------------------------
@https_fn.on_request(region="us-central1")
def gmail_oauth_callback(req: https_fn.Request) -> https_fn.Response:
    code  = req.args.get("code", "")
    state = req.args.get("state", "")
    error = req.args.get("error", "")

    if error:
        return _html_page("Connection cancelled", "You declined Gmail access. You can connect anytime from the app.")

    if not code or not state:
        return _html_page("Something went wrong", "Missing code or state. Please try the link again.")

    try:
        patient_id = verify_state(state)
    except ValueError:
        return _html_page("Link expired", "This link has expired or is invalid. Ask your navigator to send a new one.")

    token_data = _exchange_code(code)
    if "error" in token_data:
        return _html_page("Something went wrong", f"Could not connect Gmail: {token_data.get('error_description', token_data['error'])}")

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return _html_page("Something went wrong", "Google did not return a refresh token. Please try the link again.")

    patient_email = _get_email_from_id_token(token_data.get("id_token", ""))
    encrypted = encrypt_token(refresh_token)

    get_db().collection("patients").document(patient_id) \
      .collection("integrations").document("gmail") \
      .set({
          "connected":     True,
          "email":         patient_email,
          "refresh_token": encrypted,
          "granted_at":    firestore.SERVER_TIMESTAMP,
          "scope":         GMAIL_SCOPE,
      })

    return _html_page(
        "Gmail connected!",
        f"Your Gmail ({patient_email}) is now connected. Your navigator can send emails on your behalf.",
    )

# ---------------------------------------------------------------------------
# Cloud Function 3: send_gmail_on_behalf
# POST body: { "patientId", "navigatorId", "to", "subject", "body" }
# ---------------------------------------------------------------------------
@https_fn.on_request(
    region="us-central1",
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["POST"]),
)
def send_gmail_on_behalf(req: https_fn.Request) -> https_fn.Response:
    if req.method != "POST":
        return https_fn.Response("Method not allowed", status=405)

    body         = req.get_json(silent=True) or {}
    patient_id   = body.get("patientId", "").strip()
    to           = body.get("to", "").strip()
    subject      = body.get("subject", "").strip()
    email_body   = body.get("body", "").strip()
    navigator_id = body.get("navigatorId", "").strip()

    if not all([patient_id, to, subject, email_body]):
        return https_fn.Response(
            json.dumps({"error": "patientId, to, subject, and body are required"}),
            status=400, content_type="application/json",
        )

    gmail_doc = get_db().collection("patients").document(patient_id) \
                  .collection("integrations").document("gmail").get()

    if not gmail_doc.exists or not gmail_doc.get("connected"):
        return https_fn.Response(
            json.dumps({"error": "Gmail not connected for this patient"}),
            status=400, content_type="application/json",
        )

    refresh_token = decrypt_token(gmail_doc.get("refresh_token"))
    access_token  = _refresh_access_token(refresh_token)

    if not access_token:
        return https_fn.Response(
            json.dumps({"error": "Could not obtain access token"}),
            status=500, content_type="application/json",
        )

    patient_email = gmail_doc.get("email", "me")
    raw_message   = _build_raw_email(patient_email, to, subject, email_body)
    result        = _gmail_send(access_token, raw_message)

    if "error" in result:
        return https_fn.Response(
            json.dumps({"error": result["error"]}),
            status=500, content_type="application/json",
        )

    get_db().collection("patients").document(patient_id) \
      .collection("email_log").document() \
      .set({
          "to":                to,
          "subject":           subject,
          "body":              email_body,
          "sent_at":           firestore.SERVER_TIMESTAMP,
          "sent_by_navigator": navigator_id,
          "gmail_message_id":  result.get("id", ""),
          "patient_email":     patient_email,
      })

    return https_fn.Response(
        json.dumps({"success": True, "messageId": result.get("id")}),
        status=200, content_type="application/json",
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _exchange_code(code: str) -> dict:
    data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code",
    }).encode()
    req = urllib.request.Request(GOOGLE_TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

def _refresh_access_token(refresh_token: str):
    data = urllib.parse.urlencode({
        "refresh_token": refresh_token,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "refresh_token",
    }).encode()
    req = urllib.request.Request(GOOGLE_TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()).get("access_token")
    except Exception:
        return None

def _get_email_from_id_token(id_token: str) -> str:
    if not id_token:
        return ""
    try:
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("email", "")
    except Exception:
        return ""

def _build_raw_email(from_addr: str, to_addr: str, subject: str, body: str) -> str:
    message = (
        f"From: {from_addr}\r\n"
        f"To: {to_addr}\r\n"
        f"Subject: {subject}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{body}"
    )
    return base64.urlsafe_b64encode(message.encode()).decode().rstrip("=")

def _gmail_send(access_token: str, raw: str) -> dict:
    data = json.dumps({"raw": raw}).encode()
    req  = urllib.request.Request(GMAIL_SEND_URL, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}

def _html_page(title: str, message: str) -> https_fn.Response:
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Georgia, serif; background: #F5F0E8; color: #2C2416;
            display: flex; align-items: center; justify-content: center;
            min-height: 100vh; margin: 0; padding: 24px; box-sizing: border-box; }}
    .card {{ max-width: 480px; text-align: center; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 12px; }}
    p  {{ font-size: 1rem; line-height: 1.6; color: #5a4a3a; }}
  </style>
</head>
<body>
  <div class="card"><h1>{title}</h1><p>{message}</p></div>
</body>
</html>"""
    return https_fn.Response(html, status=200, content_type="text/html")