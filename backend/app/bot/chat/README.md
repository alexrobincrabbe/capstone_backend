# Chat Module (`app.bot.chat`)

This module handles bot chat behavior end-to-end:

- classify whether/how to respond
- gather context (memory/stats/history)
- generate the reply text
- optionally persist new memory
- emit structured traces per turn

## Broad Architecture

Main pieces:

- **Engine**: `engine.py`
- **Graph workflow**: `graph.py`
- **Router**: `routers.py`
- **Response generation**: `response_generator.py`
- **Prompt templates**: `prompts.py`
- **Memory services**: `memory/*`
- **Dependency assembly**: `deps.py`

The engine is the runtime orchestrator around a compiled LangGraph state machine.

## Engine Responsibilities

`BotChatEngine` is responsible for:

- Runtime state:
  - recent turns buffer
  - spam repeat state
  - last round outcome
  - deferred messages during active rounds
  - greet/session timing metadata
- Event entrypoints:
  - `on_player_joined`
  - `on_round_started`
  - `on_round_ended`
  - `on_chat_message`
- Pre-graph checks:
  - lightweight spam precheck
  - event normalization
- Graph invocation and post-processing:
  - invokes graph with state payload
  - sends reply via `send_chat`
  - emits structured traces
  - flushes deferred messages after round end

In short, the engine owns *runtime lifecycle and IO boundaries*.

## Graph Responsibilities

`build_chat_graph()` in `graph.py` defines deterministic turn flow and state transitions.

Core node sequence (chat path):

1. `event_gate` -> routes special events vs standard chat.
2. `spam_filter` -> blocks/replies for repeat/rapid spam.
3. `targeting` -> deterministic addressee hints.
4. `policy` -> probabilistic/deterministic response gate.
5. `decide` -> main route classification (via router), privacy checks, directedness resolution.
6. `plan_memory_retrieval` -> planner output (if memory route selected).
7. `gather_context` -> loads semantic memories, stats, and history slice.
8. `generate` -> LLM response generation from `ReplyContext`.
9. `decide_memory_write` -> decide if long-term memory should be written.
10. `persist_memory` -> writes memory when approved.
11. `sanitize` -> strips misleading callback phrasing in cold contexts.
12. `humanize` -> light style variance.

Special event paths:

- `player_join` event path for short welcomes.
- `round_end` event path for summary/outcome context and deferred flush.

The graph owns *decision logic and transformation of state into a reply*.

## Engine vs Graph (Practical Split)

- **Engine handles**
  - external inputs/outputs (chat send + trace emit)
  - runtime mutable buffers/counters
  - event ingestion and orchestration
- **Graph handles**
  - turn-level policy and routing decisions
  - context assembly rules
  - reply production pipeline
  - per-node tracing snapshots/transitions

## Router and Response Model Roles

- **Router (`routers.py`)**:
  - returns `RouteDecision` (`route`, privacy, stats/history need, directedness, etc.)
  - enforces a few deterministic safety/coercion rules around ambiguous outputs
- **Response generator (`response_generator.py`)**:
  - chooses prompt mode based on route/event (`join`, `simple`, `rich`)
  - generates final reply text
  - applies output sanitization (e.g., remove accidental speaker prefix)

## Memory and Context

- Semantic memory backend: `memory/semantic.py` (`PostgresSemanticMemoryService`).
- Retrieval planning: `memory/retrieval_plan_service.py`.
- Write decision: `memory/write_decision_service.py`.
- Context includes:
  - selected recent history window
  - relevant memories
  - player stats
  - last round outcome

## Tracing

- Graph records node before/after snapshots plus edge transitions in `trace`.
- Engine attaches source metadata and emits traces through `emit_trace` callback.
- `BOT_TRACE=true` enables additional server-side trace logs.

## Extending This Module

- Add/adjust deterministic turn behavior in `graph.py`.
- Change routing semantics in `routers.py` + `prompts.py`.
- Tune reply tone/format in response prompts.
- Add new context sources in `gather_context_node`.
- Keep deterministic safeguards in graph/router, and keep model-specific behavior in prompts/generator.
