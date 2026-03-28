# **Arul Health \- Orchestration Layer Sprint Scope**

**Engagement Type:** Project-based / Sprint **Duration:** 2-4 weeks **Engineer Level:** Mid-Level (Tequity) **Prepared by:** Arul Health **Date:** March 26, 2026

---

## **1\. Engagement Overview**

This sprint engages a mid-level Tequity engineer to overhaul the orchestration layer of the Arul Health platform. The engineer owns this end-to-end, from ramp to final handoff, within the boundaries defined below.

Arul Health is a chronic care coordination platform. We connect patients with human care navigators via iMessage, supported by an AI orchestration layer that handles message intake, classification, task creation, and conversation state management using LangGraph on Firebase Cloud Functions (Python).

This is not a greenfield engagement. The engineer is ramping into a live codebase with active patients on it.

---

## **2\. What's Broken and Why This Sprint Matters**

Three problems motivate this work:

**Fragile conversation looping.** The graph loops between `send_message_node` and `wait_for_navigator` without clean state boundaries. Edge cases \- duplicate messages, premature resolution, stale checkpoints, surface regularly.

**No real supervisor.** Routing logic is scattered across nodes. Adding new message types or routing paths requires touching multiple files with no clear contract.

**Reliability under load.** As patient volume grows, the current architecture shows race conditions and inconsistent behavior. The graph structure doesn't make concurrent execution safe by design.

---

## **3\. Scope of Work**

### **Workstream 1 \- LangGraph Graph Refactor**

Refactor the existing graph into a clean node structure. Each node should have a single responsibility, explicit typed state contracts, and no hidden side effects. The conversation looping flow must be stable with clear entry and exit conditions. Nodes should be unit-testable in isolation from Firebase where possible.

### **Workstream 2 \- Supervisor and Routing Logic**

Build a supervisor node that centralizes message classification and routing. The supervisor is the single place where a message's intent determines its path. Valid routing targets are enumerated. Adding a new route means touching only the supervisor and a new handler node \- nothing else. Classification prompt lives in a dedicated prompts file.

### **Workstream 3 \- Architecture Handoff Doc**

A document clear enough that another mid-level engineer can extend the system without a walkthrough. Covers: graph diagram, state schema, node descriptions, known constraints, and what requires senior review before changing.

---

## **4\. Out of Scope**

* Navigator portal (React frontend)  
* iMessage bridge (`index.ts`, Node.js client)  
* Firebase config, IAM, or Cloud Run settings  
* New agents or new message handling capabilities  
* Firestore collections structure or security rules

---

## **5\. Critical Constraints \- Read Before Writing Code**

**Sync Python only.** No `async def` anywhere. Use `llm.invoke` and `graph.invoke`. Async causes event loop errors under concurrent load on Firebase Functions.

**FirestoreSaver is a lazy singleton.** Instantiating `FirestoreSaver` at module load causes deployment timeouts. Must use a `get_checkpointer()` getter that initializes on first call.

**IAM workaround is permanent.** After every deploy, run `gcloud run services add-iam-policy-binding` manually. Omit `invoker="public"` from all function definitions.

**No in-memory state.** All conversation state and dedup keys live in Firestore — never module-level variables. In-memory state causes silent failures under concurrent requests.

**Dedup before async work.** Firestore delivers at-least-once. Set the dedup key in Firestore before doing any work, not after.

---

## **6\. Ramp Plan**

**Days 1-2:** Read the full orchestration layer. Set up local Firebase emulator. Do not deploy anything.

**Day 3:** Manually trace a single message end-to-end through the graph. Map actual behavior against intended flow. Note divergences.

**Day 4:** Alignment call with Ayush. Walk through findings, confirm refactored graph structure and supervisor routing targets, get sign-off before building.

---

## **7\. Deliverables and Acceptance Criteria**

| Deliverable | Acceptance Criteria |
| ----- | ----- |
| Refactored LangGraph graph | Clean node structure, typed state, stable looping flow, sync-only, deployable |
| Supervisor node | Single routing entrypoint, enumerated targets, prompt in dedicated file |
| Handoff document | Graph diagram, state schema, node descriptions, constraints, extension guidance |
| No regressions | Existing patient conversations unaffected |

---

## **8\. Checkpoints**

All production deploys require sign-off. The engineer does not push to production unilaterally.

* **Day 4** \- Architecture alignment before Workstream 1 begins  
* **End of Week 1** \- Graph refactor progress check  
* **Mid-sprint** \- Supervisor review before touching production  
* **Final day** \- Handoff walkthrough

