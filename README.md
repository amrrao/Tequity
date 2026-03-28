# Arul Orchestration Layer

LangGraph-powered orchestration backend for the Arul care coordination platform. Receives patient messages from the Mac iMessage server, classifies them using Gemini 2.5 Flash, writes a structured task card to Firestore for the patient navigator dashboard, sends an immediate acknowledgement back to the patient, then waits. When the navigator replies via the dashboard, the graph resumes and returns the final message to send to the patient.

---

## How It Works

```
Mac iMessage Server
        │
        │ POST /message
        ▼
  LangGraph Orchestrator
        │
  ├── Loads patient context from Firestore
  ├── Gemini classifies: task type + urgency + summary
  ├── Persists the inbound patient message in `messages/{messageId}` (for threading)
  ├── Writes task card to Firestore (navigator dashboard reads this)
  ├── Returns ack message → Mac sends to patient immediately (includes `messageId`)
  └── PAUSES (state saved to Firestore)

Navigator Dashboard
        │
        │ Navigator sees task, types reply, clicks Send
        │ POST /navigator-reply
        ▼
  LangGraph Resumes
        │
  ├── Persists the navigator reply to `messages/{messageId}` with `replyToMessageId` set
  ├── Marks task as completed in Firestore (stores `replyMessageId`)
  └── Returns final message → Mac sends to patient (includes `replyToMessageId` so iMessage can thread)
```

---

## Live Endpoints

Both functions are deployed on Firebase Cloud Functions (Python 3.13, us-central1):

| Endpoint | URL |
|---|---|
| Receive patient message | `https://us-central1-pathway-care-39e7d.cloudfunctions.net/message` |
| Send navigator reply | `https://us-central1-pathway-care-39e7d.cloudfunctions.net/navigator_reply` |

---

## Endpoint 1: `POST /message`

Called by the **Mac iMessage server** every time a patient sends a message.

### Request Body
```json
{
  "patientId":      "patient_001",
  "conversationId": "conv_001",
  "message":        "Hi, can you help me reschedule my chemo from Tuesday to Friday?",
  "replyToMessageId": "optional-message-uuid"   // optional: present when the patient is replying to a specific iMessage bubble
}
```

| Field | Type | Description |
|---|---|---|
| `patientId` | string | The patient's document ID in Firestore `patients` collection |
| `conversationId` | string | Unique ID for this conversation thread. **Must be the same value passed to `/navigator_reply` later.** Acts as the LangGraph thread ID |
| `message` | string | Raw text of the patient's iMessage |

### Response Body
```json
{
  "success":    true,
  "messageId":  "uuid-of-this-inbound-message",
  "ackMessage": "Got it! I'm passing this to your care navigator right now. We'll get back to you shortly.",
  "taskId":     "uuid-of-the-created-task",
  "urgency":    "routine"
}
```

| Field | Description |
|---|---|
| `ackMessage` | **Send this to the patient via iMessage immediately.** This is the acknowledgement text. |
| `taskId` | ID of the task created in Firestore. You can use this to look up the task later. |
| `urgency` | `"urgent"` / `"routine"` / `"low"` — reflects how Gemini classified the message |

### What happens in Firestore
A new document is created in the `tasks` collection:
```json
{
  "taskId":              "uuid",
  "patientId":           "patient_001",
  "conversationId":      "conv_001",
  "patientName":         "Maria",
  "rawMessage":          "Hi, can you help me reschedule my chemo...",
  "taskType":            "appointment",
  "urgency":             "routine",
  "summary":             "Patient wants to reschedule chemo from Tuesday to Friday.",
  "navigatorNotes":      "Patient is in chemotherapy phase. Check with oncology clinic for availability.",
  "extractedEntities":   { "dates": ["Tuesday", "Friday"], "locations": [], ... },
  "status":              "pending",
  "assignedNavigatorId": "navigator_001",
  "navigatorResponse":   null,
  "createdAt":           "timestamp",
  "updatedAt":           "timestamp"
}
```

Additionally, the inbound iMessage is stored in a new `messages` collection for threading:

```json
{
  "messageId": "uuid-of-this-inbound-message",
  "conversationId": "conv_001",
  "patientId": "patient_001",
  "sender": "patient",
  "body": "Hi, can you help me reschedule my chemo from Tuesday to Friday?",
  "replyToMessageId": null, // or the messageId the patient replied to
  "taskId": "uuid-of-the-created-task",
  "createdAt": "timestamp"
}
```

---

## Endpoint 2: `POST /navigator_reply`

Called by the **navigator dashboard** when the navigator has handled the task and typed their reply.

### Request Body
```json
{
  "conversationId":    "conv_001",
  "navigatorResponse": "Hi Maria! I've moved your chemo appointment to Friday at 2pm at the oncology clinic. You'll get a confirmation from them shortly.",
  "replyToMessageId":  "optional-message-uuid" // optional: let the navigator explicitly thread to a specific patient message
}
```

| Field | Type | Description |
|---|---|---|
| `conversationId` | string | **Must match the `conversationId` used in the original `/message` call.** This is how the system resumes the correct paused graph. |
| `navigatorResponse` | string | The full text the navigator typed as their reply to the patient |

### Response Body
```json
{
  "success":      true,
  "finalMessage": "Hi Maria! I've moved your chemo appointment to Friday at 2pm at the oncology clinic. You'll get a confirmation from them shortly.",
  "replyToMessageId": "original-message-uuid"
}
```

| Field | Description |
|---|---|
| `finalMessage` | **Send this to the patient via iMessage.** This is the navigator's reply, cleaned up and ready to send. |

### What happens in Firestore
The task document is updated:
```json
{
  "status":            "completed",
  "navigatorResponse": "Hi Maria! I've moved your chemo appointment...",
  "updatedAt":         "timestamp"
}
```

The navigator's reply is also stored in `messages/{replyMessageId}` and the task gains a `replyMessageId` field pointing to that message. The `replyToMessageId` links the navigator's message back to the original patient message so the iMessage client can thread the reply to the correct bubble.

---

## The `conversationId` — Critical Link

The `conversationId` is the single piece of data that connects the two API calls together. The LangGraph graph pauses after `/message` and saves its entire state to Firestore under this ID. When `/navigator_reply` is called with the same `conversationId`, the graph resumes exactly where it left off.

**Rule:** One unique `conversationId` per patient conversation thread. If the same patient sends multiple messages in the same thread, reuse the same `conversationId`. If it's a brand new conversation, use a new one.

---

## Firestore Collections

| Collection | Written by | Read by |
|---|---|---|
| `patients` | Seeded manually / admin | `intake_node` on every message |
| `tasks` | Orchestrator on every message | Navigator Dashboard |
| `conversations` | Orchestrator | Navigator Dashboard |
| `lg_checkpoints` | LangGraph internals | LangGraph internals |
| `lg_writes` | LangGraph internals | LangGraph internals |

Do not manually edit `lg_checkpoints` or `lg_writes`.

---

## Task Types

Gemini classifies every message into one of these task types:

| Type | Description |
|---|---|
| `appointment` | Scheduling, rescheduling, cancelling medical appointments |
| `medication` | Questions or issues about medications |
| `insurance` | Insurance claims, authorizations, coverage questions |
| `transportation` | Getting to/from appointments |
| `meal` | Food delivery or meal assistance |
| `emotional_support` | Patient needs emotional support or is distressed |
| `general` | General questions that don't fit other categories |
| `unclear` | Message is too ambiguous to classify — navigator figures it out |

---

## Testing the Full Round-Trip

### Step 1 — Patient sends a message
```bash
curl -X POST https://us-central1-pathway-care-39e7d.cloudfunctions.net/message \
  -H "Content-Type: application/json" \
  -d '{
    "patientId": "patient_001",
    "conversationId": "conv_001",
    "message": "Hi, can you help me reschedule my chemo from Tuesday to Friday?"
  }'
```

Expected response:
```json
{
  "success": true,
  "ackMessage": "Got it! I'm passing this to your care navigator right now.",
  "taskId": "some-uuid",
  "urgency": "routine"
}
```

→ Send `ackMessage` to patient via iMessage  
→ Check Firestore `tasks` collection — new document should appear

### Step 2 — Navigator replies
```bash
curl -X POST https://us-central1-pathway-care-39e7d.cloudfunctions.net/navigator_reply \
  -H "Content-Type: application/json" \
  -d '{
    "conversationId": "conv_001",
    "navigatorResponse": "Hi Maria! I moved your chemo to Friday at 2pm. The clinic will confirm."
  }'
```

Expected response:
```json
{
  "success": true,
  "finalMessage": "Hi Maria! I moved your chemo to Friday at 2pm. The clinic will confirm."
}
```

→ Send `finalMessage` to patient via iMessage  
→ Check Firestore `tasks/[taskId]` — `status` should now be `"completed"`

---

## Seeding a Test Patient

In the Firebase console → Firestore → create:

**Collection:** `patients` → **Document ID:** `patient_001`
```
name:            "Maria Garcia"
preferredName:   "Maria"
cancerType:      "breast cancer"
treatmentPhase:  "chemotherapy"
navigatorId:     "navigator_001"
```

---

## Project Structure

```
functions/
  main.py          # HTTP handlers, dedup, request/response logic
  state.py         # ArulState TypedDict and type literals
  prompts.py       # All LLM prompt templates
  nodes.py         # Graph node functions (business logic)
  graph.py         # Graph builder, routing dicts, routing functions
  services.py      # Lazy singletons (DB, LLM, checkpointer, graph)
  gmail_oauth.py   # Gmail OAuth endpoints
  requirements.txt # Python dependencies
firebase.json
.firebaserc
ARCHITECTURE.md    # Detailed architecture docs
REFACTOR.md        # Sprint scope document
test_e2e.py        # End-to-end test suite
```

---

## Local Development

```bash
# Install dependencies
cd functions && pip install -r requirements.txt

# Start emulator (requires firebase-tools: npm install -g firebase-tools)
firebase emulators:start --only functions

# Run E2E tests (requires patient_001 seeded in Firestore)
python test_e2e.py
```

To redeploy after changes:
```bash
firebase deploy --only functions
```

See `ARCHITECTURE.md` for detailed architecture documentation.