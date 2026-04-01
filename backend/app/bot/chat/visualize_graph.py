from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..models import ChatRoute, MemoryRetrievalPlan, MemoryWriteDecision, RouteDecision
from ..round_summary_service import RoundEndResult
from .graph import build_chat_graph, GraphDeps

CompiledGraph = Any


class _DummyRouter:
    async def classify(
        self,
        *,
        text: str,
        username: str,
        is_round_active: bool,
        recent_turns: list[str],
        last_round_outcome: object | None = None,
        participant_count: int = 0,
        bot_name: str = "",
        targeting_hint: str = "ambiguous",
        history: list[Any] | None = None,
        participant_names: list[str] | None = None,
        farewell_piggyback_likely: bool = False,
    ) -> RouteDecision:
        del (
            text,
            username,
            is_round_active,
            recent_turns,
            last_round_outcome,
            participant_count,
            bot_name,
            targeting_hint,
            history,
            participant_names,
            farewell_piggyback_likely,
        )
        return RouteDecision(
            route=ChatRoute.IGNORE,
            ignore_reason="visualize_dummy",
            directed_at_bot=None,
        )


class _DummyMemory:
    pass


class _DummyPlayerStats:
    pass


class _DummyResponder:
    pass


class _DummyRoundSummary:
    async def handle_round_end(self, room_state: object) -> RoundEndResult:
        return RoundEndResult(
            summary_text="gg",
            outcome={},
        )


class _DummyMemoryRetrievalPlan:
    async def plan(self, **kwargs: object) -> MemoryRetrievalPlan:
        del kwargs
        return MemoryRetrievalPlan(
            use_memory=False,
            query=None,
            mode="none",
            min_similarity=0.0,
            max_results=0,
        )


class _DummyMemoryWrite:
    async def decide(self, **kwargs: object) -> MemoryWriteDecision:
        del kwargs
        return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)


def build_compiled_graph() -> CompiledGraph:
    deps = GraphDeps(
        router=_DummyRouter(),
        memory=_DummyMemory(),
        player_stats=_DummyPlayerStats(),
        responder=_DummyResponder(),
        round_summary=_DummyRoundSummary(),
        memory_retrieval_plan=_DummyMemoryRetrievalPlan(),
        memory_write=_DummyMemoryWrite(),
    )
    return build_chat_graph(deps)


def _write_mermaid_file(graph_viz: object, out_path: Path) -> None:
    mermaid = graph_viz.draw_mermaid()  # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mermaid, encoding="utf-8")
    print(f"Wrote Mermaid graph to: {out_path}")


def _write_html_file(graph_viz: object, out_path: Path) -> None:
    mermaid = graph_viz.draw_mermaid()  # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LangGraph Visualization</title>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      background: #ffffff;
      color: #111111;
    }}
    .mermaid {{
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <h1>LangGraph Visualization</h1>
  <pre class="mermaid">
{mermaid}
  </pre>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML graph to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the bot chat LangGraph.")
    parser.add_argument(
        "--mermaid",
        type=str,
        default="",
        help="Optional output path for Mermaid text (for example: graph.mmd).",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="",
        help="Optional output path for an HTML file that renders the Mermaid graph.",
    )
    args = parser.parse_args()

    app = build_compiled_graph()
    graph_viz = app.get_graph()  # type: ignore[attr-defined]

    if args.mermaid:
        _write_mermaid_file(graph_viz, Path(args.mermaid).resolve())
        return

    if args.html:
        _write_html_file(graph_viz, Path(args.html).resolve())
        return

    ascii_map = graph_viz.draw_ascii()  # type: ignore[attr-defined]
    print(ascii_map)


if __name__ == "__main__":
    main()