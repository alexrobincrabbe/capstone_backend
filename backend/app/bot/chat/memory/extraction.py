from __future__ import annotations

import re
from typing import Any


class MemoryExtractionService:
    async def extract_memory(
        self,
        *,
        username: str,
        user_message: str,
        bot_reply: str | None = None,
        memories: list[Any] | None = None,
        stats: Any | None = None,
    ) -> str | None:
        raise NotImplementedError


class HeuristicMemoryExtractionService(MemoryExtractionService):
    async def extract_memory(
        self,
        *,
        username: str,
        user_message: str,
        bot_reply: str | None = None,
        memories: list[Any] | None = None,
        stats: Any | None = None,
    ) -> str | None:
        del bot_reply, memories, stats  # hooks for richer extractors; heuristic uses user text only
        text = user_message.strip()
        lowered = text.lower()
        if len(text) < 12:
            return None
        contextual = self._contextual_memory(username=username, text=text, lowered=lowered)
        if contextual:
            return contextual
        personal_detail = re.search(r"\bmy\b", lowered) or re.search(r"\bi (like|usually|prefer)\b", lowered)
        life_event = re.search(
            r"\b(i (had|have|am|was|got|did)|today|yesterday|tomorrow|exam|test|interview|job|work|school|stressed|stressful|sick|ill|tired|family)\b",
            lowered,
        )
        if personal_detail or life_event:
            return f"{username}: {text}"
        return None

    def _contextual_memory(self, *, username: str, text: str, lowered: str) -> str | None:
        # "I go there with my husband on wednesdays" -> stable preference/routine memory.
        outing_match = re.search(
            r"\bi go (?:there|to ([a-z][a-z\s]{1,40})) with my (husband|wife|partner|boyfriend|girlfriend)(?: on ([a-z]+))?\b",
            lowered,
        )
        if outing_match:
            place = (outing_match.group(1) or "that place").strip()
            relation = outing_match.group(2)
            day = (outing_match.group(3) or "").strip()
            day_part = f" on {day.title()}s" if day else ""
            if place == "that place":
                return f"{username} goes there with their {relation}{day_part}."
            return f"{username} goes to {place} with their {relation}{day_part}."

        location_match = re.search(
            r"\bi(?:\s*am|'m)?\s+(?:up in|in|from)\s+([a-z][a-z\s]{1,30})\b",
            lowered,
        )
        if location_match:
            place = location_match.group(1).strip().rstrip(".!?")
            return f"{username} is in {place.upper() if len(place) <= 3 else place.title()}."

        children_match = re.search(
            r"\b(?:i\s+)?have\s+(\d+)\s+(daughter|daughters|son|sons|kid|kids|children)\b",
            lowered,
        )
        if children_match:
            count = int(children_match.group(1))
            noun = children_match.group(2)
            if noun in {"daughter", "daughters"}:
                return f"{username} has {count} daughters."
            if noun in {"son", "sons"}:
                return f"{username} has {count} sons."
            return f"{username} has {count} children."

        # "I like ballet" / "I love chess"
        interest_match = re.search(r"\bi (like|love|enjoy|prefer)\s+([a-z][a-z\s]{1,40})\b", lowered)
        if interest_match:
            interest = interest_match.group(2).strip().rstrip(".!?")
            return f"{username} likes {interest}."

        # "my husband" / "my wife" style relationship hints.
        if re.search(r"\bmy (husband|wife|partner|boyfriend|girlfriend)\b", lowered):
            relation = re.search(r"\bmy (husband|wife|partner|boyfriend|girlfriend)\b", lowered)
            if relation:
                return f"{username} mentioned having a {relation.group(1)}."

        return None
