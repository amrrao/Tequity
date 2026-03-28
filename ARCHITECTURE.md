# Arul Orchestration Layer - Architecture

## Overview

LangGraph-powered backend for a cancer care coordination platform. Receives patient messages via HTTP, classifies them with Gemini 2.5 Flash, manages conversation state through human-in-the-loop interrupts, and coordinates responses between patients and care navigators.

**Stack:** Firebase Cloud Functions (Python 3.13), LangGraph + Firestore checkpointing, Gemini 2.5 Flash, Firestore.

---

## File Structure

```
functions/
  main.py          # HTTP handlers, dedup, request/response logic
  state.py         # ArulState TypedDict and type literals
  prompts.py       # All LLM prompt templates
  nodes.py         # Graph node functions (business logic)
  graph.py         # Graph builder, routing dicts, routing functions
  services.py      # Lazy singletons (DB, LLM, checkpointer, graph)
  gmail_oauth.py   # Gmail OAuth endpoints (separate concern, imported by main.py)
  requirements.txt # Python dependencies
```

---

## Graph Flow

```
START --> intake --> supervisor
                       |
            +----------+----------+
            |          |          |
         chitchat   care_need  outbound
            |          |          |
        ai_reply   write_task  write_task
            |          |          |
           END    generate_ack  format_outbound
                       |          |
               wait_for_navigator END
                       |
                +------+------+
                |             |
             resolve       followup
                |             |
          format_reply   write_followup
                |             |
               END      wait_for_patient ---+
                              |             |
                              +-- (loops back to wait_for_navigator)
```

### Routing Decision Points

All routing is defined in `graph.py` with enumerated dicts:

| Decision Point | State Field | Routes |
|---|---|---|
| After supervisor | `state["route"]` | chitchat -> ai_reply, care_need -> write_task, outbound -> write_task |
| After write_task | `state["route"]` | care_need -> generate_ack, outbound -> format_outbound |
| After wait_for_navigator | `state["navigator_action"]` | resolve -> format_reply, followup -> write_followup |

---

## State Schema (`state.py`)

```python
class ArulState(TypedDict):
    raw_message: str                    # Patient or navigator message text
    patient_id: str                     # Firestore patient doc ID
    conversation_id: str                # Thread ID (LangGraph checkpoint key)
    patient_name: str                   # Preferred name, loaded by intake
    patient_context: dict               # cancer_type, treatment_phase, navigator_id
    task_id: str                        # UUID for the Firestore task document
    task_type: TaskType                 # Classification result
    urgency: Urgency                    # urgent | routine | low
    summary: str                        # One-sentence summary of patient need
    navigator_notes: str                # Context for the navigator
    extracted_entities: dict             # Dates, locations, people, etc.
    ack_message: str                    # Acknowledgment or chitchat reply sent to patient
    navigator_response: Optional[str]   # Navigator's reply text
    navigator_action: Optional[str]     # resolve | followup
    final_message: str                  # Final message to send to patient
    message_id: str                     # UUID for the current message document
    reply_to_message_id: Optional[str]  # For iMessage threading
    is_ambiguous: bool                  # Flag for ambiguous messages
    is_outbound: bool                   # Navigator-initiated message
    route: Route                        # chitchat | care_need | outbound
```

**Type literals:**
- `TaskType`: appointment, medication, insurance, transportation, meal, emotional_support, general, unclear, chitchat
- `Urgency`: urgent, routine, low
- `Route`: chitchat, care_need, outbound

---

## Node Descriptions (`nodes.py`)

| Node | Purpose | Side Effects |
|---|---|---|
| **intake** | Load patient profile from Firestore `patients` collection | Firestore read |
| **supervisor** | Classify message via Gemini (or skip for outbound). Sets `route`, `task_type`, `urgency`, `summary` | LLM call (unless outbound) |
| **ai_reply** | Generate warm chitchat response via Gemini | LLM call |
| **write_task** | Persist patient message, create task, update conversation in a single batch | Firestore batch write |
| **generate_ack** | Generate acknowledgment for first message in conversation. Returns `URGENT_ACK` constant for urgent, LLM-generated for others, empty for follow-up messages | Firestore read + LLM call |
| **format_outbound** | Persist navigator-initiated message and mark task completed | Firestore batch write |
| **wait_for_navigator** | `interrupt()` - pauses graph until navigator responds via `/navigator_reply` | LangGraph interrupt |
| **write_followup** | Persist navigator's followup question, set task to `awaiting_patient` | Firestore batch write |
| **wait_for_patient** | `interrupt()` - pauses graph until patient responds via `/patient_reply`. Writes patient reply to Firestore and clears stale loop state | LangGraph interrupt + Firestore batch write |
| **format_reply** | Persist final navigator response, mark task completed, resolve conversation | Firestore batch write |

---

## HTTP Endpoints (`main.py`)

| Endpoint | Method | Purpose | Graph Interaction |
|---|---|---|---|
| `/message` | POST | Patient sends inbound message | `graph.invoke(initial_state)` |
| `/navigator_reply` | POST | Navigator resolves or asks followup | `graph.invoke(Command(resume=...))` |
| `/patient_reply` | POST | Patient responds to followup question | `graph.invoke(Command(resume=...))` |
| `/onboard_patient` | POST | Generate welcome message for new patient | Direct LLM call (no graph) |
| `/classify_message` | POST | Route message to existing or new conversation | Direct LLM call (no graph) |
| `/outbound_message` | POST | Navigator sends proactive message | Graph invoke (new conv) or direct write (existing) |

**Deduplication:** All graph-invoking endpoints use content-based SHA256 dedup keys stored in a `dedup` Firestore collection. The key is set to "processing" before work begins and "completed" after, with the cached response stored for replay.

---

## Firestore Collections

| Collection | Written By | Purpose |
|---|---|---|
| `patients` | Admin / seeding | Patient profile data |
| `messages` | Graph nodes, main.py | Message history for iMessage threading |
| `tasks` | Graph nodes, main.py | Work items for navigator dashboard |
| `conversations` | Graph nodes, main.py | Conversation metadata and status tracking |
| `dedup` | main.py | At-least-once delivery deduplication |
| `lg_checkpoints` | LangGraph | Graph state persistence (do not edit manually) |
| `lg_writes` | LangGraph | Graph write log (do not edit manually) |

---

## Critical Constraints

**1. Sync Python only.** No `async def` anywhere. Use `llm.invoke()` and `graph.invoke()`. Async causes event loop errors under concurrent load on Firebase Cloud Functions.

**2. FirestoreSaver must be lazy.** Never instantiate at module load. Use `get_checkpointer()` in `services.py`, which initializes on first call. Module-load instantiation causes deployment timeouts.

**3. No in-memory state.** All conversation state and dedup keys live in Firestore. Module-level variables are only used for lazy singletons (DB, LLM, graph). In-memory state causes silent failures under concurrent requests.

**4. Dedup before work.** Set the dedup key to "processing" in Firestore before doing any graph invocation. Firestore delivers at-least-once, so the dedup write must happen first to prevent duplicate processing.

**5. IAM binding after deploy.** After every deploy, manually run:
```bash
gcloud run services add-iam-policy-binding <function-name> \
  --region=us-central1 --member=allUsers --role=roles/run.invoker
```
Do not add `invoker="public"` to function definitions.

---

## Extension Guide

### Adding a New Message Route

1. **`state.py`** - Add to `TaskType` literal if needed
2. **`prompts.py`** - Add the new type to `SUPERVISOR_PROMPT` taskType list and rules
3. **`nodes.py`** - Add handler node function: `def my_node(state: ArulState) -> dict`
4. **`graph.py`** - Add node to graph, add to `SUPERVISOR_ROUTES` dict, update `supervisor_node` to set the new route value

### Adding a New Endpoint

1. **`main.py`** - Add function with `@https_fn.on_request(**_FN_OPTS, timeout_sec=...)` decorator
2. If it invokes the graph, use `_initial_state()` helper and add dedup logic
3. Deploy and run IAM binding (Constraint #5)

---

## Testing

### E2E Tests

```bash
# Start emulator
firebase emulators:start --only functions

# Run full suite (needs patient_001 seeded in Firestore)
python test_e2e.py

# Run single test class
python test_e2e.py TestCareNeedFollowup
```

Tests cover: chitchat, care_need resolve, care_need followup loop, outbound (new + existing), onboard, classify, dedup, validation errors, error message safety.

### After Deploy

1. Send a chitchat message - verify AI reply, no task created
2. Send a care need - verify task creation + ack message
3. Send navigator reply (resolve) - verify task completion
4. Send navigator reply (followup) - verify loop back to patient
5. Monitor logs: `firebase functions:log`

---

## Known Limitations

- **No max iterations on followup loop.** If a navigator sends many followups, the graph loops indefinitely. Consider adding a counter if this becomes an issue.
- **Checkpoint migration.** Existing checkpoints from before the refactor don't have the `route` field. Routing functions use `.get("route", "care_need")` as a fallback.
- **Free-tier Gemini rate limits.** The `gemini-2.5-flash` free tier allows 20 requests/day. E2E tests consume ~15 requests per full run.
