"""
End-to-end tests for the Arul orchestration layer.

Requires:
  - Firebase emulator running: firebase emulators:start --only functions
  - patient_001 seeded in Firestore with preferredName, cancerType, treatmentPhase

Usage:
  python test_e2e.py                          # run all tests
  python test_e2e.py TestChitchat             # run one test class
  python test_e2e.py TestCareNeedResolve      # run one test class
"""

import json
import time
import unittest
import urllib.request
import urllib.error

BASE = "http://localhost:5001/arul-development/us-central1"
PATIENT_ID = "patient_001"


def post(endpoint, body):
    """POST JSON to the emulator and return (status_code, parsed_json)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def fresh_conv_id(label):
    return f"test_{label}_{int(time.time() * 1000)}"


class TestChitchat(unittest.TestCase):
    """Patient sends casual message -> AI replies, no task created."""

    def test_chitchat_returns_ai_reply(self):
        conv_id = fresh_conv_id("chitchat")
        status, resp = post("message", {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "Hey, how are you doing today?",
        })
        print(f"\n  Chitchat response: {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertTrue(resp["isChitchat"], "Expected chitchat classification")
        self.assertTrue(len(resp["ackMessage"]) > 0, "Expected non-empty AI reply for chitchat")
        # supervisor_node always generates a task_id UUID, but chitchat never
        # writes it to Firestore (write_task_node is skipped), so it's harmless


class TestCareNeedResolve(unittest.TestCase):
    """Patient sends care need -> navigator resolves it."""

    def test_full_resolve_flow(self):
        conv_id = fresh_conv_id("resolve")

        # Step 1: Patient sends care need
        print("\n  Step 1: Patient sends care need...")
        status, resp = post("message", {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "I need to reschedule my chemo appointment from Tuesday to Friday",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertFalse(resp["isChitchat"], "Should be classified as care need, not chitchat")
        self.assertTrue(len(resp["taskId"]) > 0, "Should create a task")
        self.assertIn(resp["urgency"], ["urgent", "routine", "low"])
        self.assertTrue(len(resp["ackMessage"]) > 0, "First message should get an ack")

        # Step 2: Navigator resolves
        print("  Step 2: Navigator resolves...")
        status, resp = post("navigator_reply", {
            "conversationId": conv_id,
            "navigatorResponse": "Hi Maria! I moved your chemo to Friday at 2pm.",
            "action": "resolve",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertEqual(resp["status"], "completed")
        self.assertIn("Friday", resp["finalMessage"])


class TestCareNeedFollowup(unittest.TestCase):
    """Patient sends care need -> navigator asks followup -> patient replies -> navigator resolves."""

    def test_full_followup_flow(self):
        conv_id = fresh_conv_id("followup")

        # Step 1: Patient sends care need
        print("\n  Step 1: Patient sends care need...")
        status, resp = post("message", {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "I need help getting to my appointment next week",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertFalse(resp["isChitchat"])

        # Step 2: Navigator asks followup
        print("  Step 2: Navigator asks followup...")
        status, resp = post("navigator_reply", {
            "conversationId": conv_id,
            "navigatorResponse": "Which day is your appointment? And what's your address?",
            "action": "followup",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertEqual(resp["status"], "awaiting_patient")

        # Step 3: Patient replies to followup
        print("  Step 3: Patient replies to followup...")
        status, resp = post("patient_reply", {
            "conversationId": conv_id,
            "patientId": PATIENT_ID,
            "message": "It's on Thursday at 10am. I live at 123 Main St.",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200, f"patient_reply failed: {resp}")
        self.assertTrue(resp["success"])
        self.assertEqual(resp["status"], "awaiting_navigator")

        # Step 4: Navigator resolves
        print("  Step 4: Navigator resolves...")
        status, resp = post("navigator_reply", {
            "conversationId": conv_id,
            "navigatorResponse": "I've arranged a ride for Thursday at 9am from 123 Main St.",
            "action": "resolve",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertEqual(resp["status"], "completed")


class TestOutbound(unittest.TestCase):
    """Navigator sends proactive message to patient (new conversation)."""

    def test_outbound_new_conversation(self):
        conv_id = fresh_conv_id("outbound")
        print("\n  Outbound message (new conversation)...")
        status, resp = post("outbound_message", {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "Hi Maria, just checking in - how are you feeling after your last treatment?",
            "navigatorId": "navigator_001",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])
        self.assertTrue(len(resp["taskId"]) > 0)

    def test_outbound_existing_conversation(self):
        conv_id = fresh_conv_id("outbound_existing")
        print("\n  Outbound message (existing conversation)...")
        status, resp = post("outbound_message", {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "Just following up on your medication question from yesterday.",
            "navigatorId": "navigator_001",
            "isExistingConversation": True,
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertTrue(resp["success"])


class TestOnboard(unittest.TestCase):
    """Generate welcome message for new patient."""

    def test_onboard_patient(self):
        print("\n  Onboard patient...")
        status, resp = post("onboard_patient", {
            "patientId": PATIENT_ID,
            "navigatorName": "Sarah",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        # 200 = success, 409 = already onboarded (both acceptable)
        self.assertIn(status, [200, 409])
        if status == 200:
            self.assertTrue(resp["success"])
            self.assertTrue(len(resp["welcomeMessage"]) > 0)


class TestClassifyMessage(unittest.TestCase):
    """Classify whether a message belongs to an existing conversation."""

    def test_classify_new_message(self):
        print("\n  Classify message (no open conversations)...")
        status, resp = post("classify_message", {
            "patientId": "patient_no_open_tasks",
            "message": "I have a question about my medication",
        })
        print(f"  -> {json.dumps(resp, indent=2)}")
        self.assertEqual(status, 200)
        self.assertEqual(resp["decision"], "new")


class TestDedup(unittest.TestCase):
    """Duplicate messages should return cached response."""

    def test_duplicate_message_returns_cached(self):
        conv_id = fresh_conv_id("dedup")
        payload = {
            "patientId": PATIENT_ID,
            "conversationId": conv_id,
            "message": "I need to cancel my appointment tomorrow",
        }

        print("\n  First request...")
        status1, resp1 = post("message", payload)
        print(f"  -> {json.dumps(resp1, indent=2)}")
        self.assertEqual(status1, 200)
        self.assertTrue(resp1["success"])

        print("  Duplicate request (same message + conv)...")
        status2, resp2 = post("message", payload)
        print(f"  -> {json.dumps(resp2, indent=2)}")
        self.assertEqual(status2, 200)
        # Should return the same cached response
        self.assertEqual(resp1["messageId"], resp2["messageId"])
        self.assertEqual(resp1["taskId"], resp2["taskId"])


class TestEdgeCases(unittest.TestCase):
    """Validation and error handling."""

    def test_missing_fields(self):
        status, resp = post("message", {"patientId": PATIENT_ID})
        self.assertEqual(status, 400)

    def test_empty_message(self):
        status, resp = post("message", {
            "patientId": PATIENT_ID,
            "conversationId": "test",
            "message": "",
        })
        self.assertEqual(status, 400)

    def test_navigator_reply_missing_fields(self):
        status, resp = post("navigator_reply", {"conversationId": "test"})
        self.assertEqual(status, 400)

    def test_patient_reply_no_awaiting_task(self):
        status, resp = post("patient_reply", {
            "conversationId": "nonexistent_conv",
            "patientId": PATIENT_ID,
            "message": "hello",
        })
        self.assertEqual(status, 409)

    def test_errors_dont_leak_internals(self):
        """500 errors should not expose stack traces."""
        status, resp = post("navigator_reply", {
            "conversationId": "nonexistent_thread_xyz",
            "navigatorResponse": "test",
            "action": "resolve",
        })
        if status == 500:
            self.assertEqual(resp["error"], "Internal server error")


if __name__ == "__main__":
    print(f"Running E2E tests against {BASE}")
    print(f"Patient ID: {PATIENT_ID}\n")
    unittest.main(verbosity=2)
