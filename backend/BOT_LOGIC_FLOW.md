# Bot Logic Flow (Detailed)

This diagram focuses only on the bot-side runtime logic for chat handling.

## Scope

- Includes: `BotRoomAutomation` -> `BotManager` -> `BotController` -> `BotChatEngine`
- Includes router, response generation, memory retrieval/storage, and guardrails.
- Excludes broader websocket/game transport details unless directly needed by bot flow.

## Mermaid Diagram

```mermaid
flowchart TD
    A[Coordinator calls BotRoomAutomation.on_chat_message] --> B[BotManager.on_chat_message]
    B --> C[BotController.on_chat_message]
    C --> D[BotChatEngine.on_chat_message]

    D --> E[Record incoming turn in recent_turns]
    E --> F{Sender is bot/self?}
    F -- Yes --> Z1[Return]
    F -- No --> G{Round active?}
    G -- Yes --> G1[Defer message for post-round flush] --> Z1
    G -- No --> H{Spam repeat threshold hit?}
    H -- Yes --> H1[Optionally send anti-spam warning] --> Z1
    H -- No --> I{Goodbye intent?}
    I -- Yes --> I1[Deterministic goodbye reply + cooldown] --> Z1
    I -- No --> J{Third-party private question?}
    J -- Yes --> J1[Ignore for privacy] --> Z1
    J -- No --> K{Reply policy gate allows?}
    K -- No --> Z1
    K -- Yes --> L{LLM router available?}
    L -- No --> Z1
    L -- Yes --> M[Router classify route]
    M --> N{Route = ignore?}
    N -- Yes --> Z1
    N -- No --> O{Route needs memory?}
    O -- Yes --> O1[Retrieve relevant memories]
    O -- No --> P
    O1 --> P[Load optional stats/history by route]
    P --> Q{Special remember-me override?}
    Q -- Yes --> Q1[Deterministic remember reply] --> V
    Q -- No --> R{Response generator available?}
    R -- No --> Z1
    R -- Yes --> S[Generate LLM reply from context]
    S --> T[Apply false-familiarity sanitizer]
    T --> U[Humanize text: casing/punctuation/typos]
    U --> V[Send chat + mark reply policy + record bot turn]

    V --> W{Should store memory? route/store flag/heuristic}
    W -- No --> Z1
    W -- Yes --> X[_maybe_store_memory]
    X --> X1{No prior memories?}
    X1 -- Yes --> X2[Bootstrap: last_seen + profile/first_chat]
    X1 -- No --> X3[Skip bootstrap]
    X2 --> Y[Run memory extractor]
    X3 --> Y
    Y --> Y1{Extractor returned contextual memory?}
    Y1 -- Yes --> Y2[Store chat memory]
    Y1 -- No --> Y3[No chat memory write]
    Y2 --> Y4[Update rolling last_seen]
    Y3 --> Y4
    Y4 --> Z1

    AA[BotChatEngine.on_round_started] --> AB[Record EVENT: round_started]
    AC[BotChatEngine.on_round_ended] --> AD[Summarize outcome]
    AD --> AE[Update player_stats wins/losses]
    AE --> AF[Store rolling round_summary memory]
    AF --> AG[Record EVENT: round_ended]
    AG --> AH[Optional event comment via router+generator]
    AH --> AI[Flush deferred in-round messages]
```
