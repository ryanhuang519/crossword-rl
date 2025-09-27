#!/usr/bin/env python3
"""Automatically play a NYT Mini crossword using an LLM."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from litellm import completion
from litellm.types.utils import Choices, Message

from nyt import (
    MiniCrossword,
    build_mini_from_api,
    load_cached_mini,
    load_remote_json,
)

SYSTEM_PROMPT = """
You're a crossword expert.
I will provide you with a 5x5 mini crossword and you will solve it 1 clue at a time.
After each guess, I will validate your guess and send you the updated game board.
We will keep doing this until you have solved the crossword.

## Response format:
Guess in the format: "guess 1a=red"
You can also delete guesses you believe to be incorrect using "delete 1a"
You can only make 1 guess at a time. If you make more than 1 guess, really bad things will happen.

## Important:
Do not try solve the entire puzzle at once or reason about what the puzzle will look like.
After each guess, I will show you what the puzzle looks like.
"""
# you can make multiple moves in a single response by comma separating them, like "guess 1a=red, guess 1d=brown". if any of the moves are invalid, then all the moves after that will not be run.

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
DEFAULT_MAX_TURNS = 30

RESET_COLOR = "\033[0m"
ROLE_COLORS = {
    "system": "\033[95m",
    "user": "\033[94m",
    "assistant": "\033[92m",
}
ROLE_LABELS = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
}


def format_message(role: str, content: str, *, annotation: str | None = None) -> str:
    color = ROLE_COLORS.get(role, "")
    label = ROLE_LABELS.get(role, role.upper())
    if annotation:
        label = f"{label} {annotation}"
    header = f"{color}[{label}]{RESET_COLOR}" if color else f"[{label}]"
    return f"{header}\n{content}"


def print_message(role: str, content: str, *, annotation: str | None = None) -> None:
    print(format_message(role, content, annotation=annotation))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Have an LLM solve a NYT Mini crossword for a given date.",
    )
    parser.add_argument("date", help="Date of the puzzle in MM-DD-YYYY format.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="litellm model identifier to use (default: groq/openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum number of LLM turns before stopping.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached puzzles and download the puzzle again.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Display provider reasoning text if available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw LLM payloads for debugging.",
    )
    return parser.parse_args()


def log_debug(enabled: bool, title: str, payload: Any) -> None:
    if not enabled:
        return

    serializable = payload
    if hasattr(serializable, "model_dump"):
        try:
            serializable = serializable.model_dump()  # type: ignore[assignment]
        except TypeError:
            serializable = serializable.model_dump(mode="python")  # type: ignore[assignment]
    elif isinstance(serializable, list):
        serializable = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in serializable
        ]
    elif isinstance(serializable, dict):
        serializable = {
            key: value.model_dump() if hasattr(value, "model_dump") else value
            for key, value in serializable.items()
        }

    try:
        formatted = json.dumps(serializable, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        formatted = str(serializable)
    print_message("system", formatted, annotation=f"debug {title}")


def load_mini(date_text: str, *, force_download: bool = False) -> MiniCrossword:
    cache_path = Path("minis") / f"{date_text}.json"

    if cache_path.exists() and not force_download:
        print(f"Loaded cached puzzle for {date_text} from {cache_path}.")
        return load_cached_mini(cache_path)

    data = load_remote_json(date_text)
    mini = build_mini_from_api(date_text, data)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(mini), f, ensure_ascii=False, indent=2)
    print(f"Downloaded puzzle for {date_text} and cached to {cache_path}.")
    return mini


def render_start(mini: MiniCrossword) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        mini.render_starting_puzzle()
    return buffer.getvalue().strip()


def render_current(mini: MiniCrossword, moves: List[str]) -> Tuple[str, Tuple[int, int, int]]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        stats = mini.render(moves, show_progress=True)
    return buffer.getvalue().strip(), stats


def extract_segments(raw_response: str) -> List[str]:
    segments: List[str] = []
    for line in raw_response.splitlines():
        cleaned = line.strip().rstrip(".")
        if not cleaned:
            continue
        for part in cleaned.split(","):
            candidate = part.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered.startswith("guess") or lowered.startswith("delete") or "=" in candidate:
                segments.append(candidate)
    return segments


def apply_segments(mini: MiniCrossword, moves: List[str], segments: List[str]) -> Tuple[str, bool]:
    feedback_lines: List[str] = []
    puzzle_complete = False
    invalid_encountered = False

    for segment in segments:
        if invalid_encountered:
            break
        try:
            mini.play_move(segment, moves)
        except ValueError as exc:
            feedback_lines.append(f"Invalid move '{segment}': {exc}")
            feedback_lines.append("Remaining moves after this were not applied.")
            invalid_encountered = True
            break

        letters, filled, correct, total = mini._compute_board_state(moves)  # type: ignore[attr-defined]
        if correct == total:
            puzzle_complete = True
            break

    board_text, stats = render_current(mini, moves)
    filled, correct, total = stats

    if puzzle_complete:
        feedback_lines.append("Puzzle complete! 🎉")
    elif invalid_encountered:
        feedback_lines.append("Please submit another clue.")
    elif segments:
        feedback_lines.append("Valid guess!")
        feedback_lines.append("")
        feedback_lines.append("Please submit another clue.")

    feedback_lines.append("Grid layout:")
    feedback_lines.extend(board_text.splitlines())

    if not puzzle_complete:
        feedback_lines.insert(1, f"Progress: {filled}/{total} filled, {correct}/{total} correct.")

    return "\n".join(feedback_lines).strip(), puzzle_complete


def send_noop_feedback(mini: MiniCrossword, moves: List[str]) -> str:
    board_text, stats = render_current(mini, moves)
    filled, correct, total = stats

    lines = [
        "No valid moves detected. Respond with moves like 'guess 1a=word' or 'delete 1d'.",
        f"Progress: {filled}/{total} filled, {correct}/{total} correct.",
        "",
        "Grid layout:",
    ]
    lines.extend(board_text.splitlines())

    return "\n".join(lines)


def extract_assistant_content(choice: Any) -> tuple[str | None, Optional[str]]:
    reasoning_text: Optional[str] = None

    if isinstance(choice, Choices):
        message_obj = choice.message if isinstance(choice.message, Message) else None
        if message_obj is not None:
            content = getattr(message_obj, "content", None)
            if isinstance(content, str) and content:
                reasoning_text = getattr(message_obj, "reasoning_content", None) or getattr(message_obj, "reasoning", None)  # type: ignore[attr-defined]
                return content, reasoning_text

        text_value = getattr(choice, "text", None)
        if isinstance(text_value, str) and text_value:
            return text_value, reasoning_text

        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)  # type: ignore[attr-defined]
                return content, reasoning_text

        return None, reasoning_text

    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                reasoning_text = message.get("reasoning_content") or message.get("reasoning")
                return content, reasoning_text

        content = choice.get("content")
        if isinstance(content, str) and content:
            return content, None

        text_value = choice.get("text")
        if isinstance(text_value, str) and text_value:
            return text_value, None

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
                return content, reasoning_text

    return None, reasoning_text


def run_agent(
    mini: MiniCrossword,
    *,
    model: str,
    temperature: float,
    max_turns: int,
    debug: bool,
    reasoning: bool,
) -> None:
    moves: List[str] = []
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print_message("system", SYSTEM_PROMPT, annotation="prompt")

    starting_layout = render_start(mini)
    print_message("user", starting_layout, annotation="turn 0")

    history.append({"role": "user", "content": starting_layout})

    for turn in range(1, max_turns + 1):
        log_debug(debug, f"request turn {turn}", history)
        try:
            response = completion(
                model=model,
                messages=history,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            print_message("system", f"LLM request failed on turn {turn}: {exc}", annotation="error")
            return

        log_debug(debug, f"response turn {turn}", response)
        choices = list(response.choices or [])
        if not choices:
            log_debug(debug, f"empty-choices turn {turn}", response)
            print_message("system", "Model response did not include choices. Stopping.", annotation=f"turn {turn}")
            return

        choice_payload = choices[0]
        assistant_reply, reasoning_text = extract_assistant_content(choice_payload)
        assistant_reply = (assistant_reply or "").strip()
        if not assistant_reply:
            log_debug(debug, f"empty-choice turn {turn}", choice_payload)
            print_message("system", "Received empty response from model. Stopping.", annotation=f"turn {turn}")
            return

        if reasoning and reasoning_text:
            print_message("assistant", reasoning_text.strip(), annotation=f"turn {turn} reasoning")

        print_message("assistant", assistant_reply, annotation=f"turn {turn}")
        history.append({"role": "assistant", "content": assistant_reply})

        segments = extract_segments(assistant_reply)
        if not segments:
            feedback = send_noop_feedback(mini, moves)
            print_message("user", feedback, annotation=f"turn {turn}")
            history.append({"role": "user", "content": feedback})
            continue

        feedback, puzzle_complete = apply_segments(mini, moves, segments)
        print_message("user", feedback, annotation=f"turn {turn}")
        history.append({"role": "user", "content": feedback})

        if puzzle_complete:
            print_message("system", "Crossword solved by the agent!", annotation="complete")
            return

    print_message("system", f"Reached max turns ({max_turns}) without completing the puzzle.", annotation="status")


def main() -> None:
    load_dotenv()
    args = parse_args()
    mini = load_mini(args.date, force_download=args.force_download)
    run_agent(
        mini,
        model=args.model,
        temperature=args.temperature,
        max_turns=args.max_turns,
        debug=args.debug,
        reasoning=args.reasoning,
    )


if __name__ == "__main__":
    main()
