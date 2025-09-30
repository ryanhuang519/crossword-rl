#!/usr/bin/env python3
"""Automatically play a NYT Mini crossword using an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from litellm import Choices, completion

from nyt import MiniCrossword, load_from_remote_or_cache

MULTI_TURN_PROMPT = """
You're a crossword expert.
I will provide you with a 5x5 mini crossword and you will solve it 1 clue at a time.
After each guess, I will validate your guess and send you the updated game board.
We will keep doing this until you have solved the crossword.

## Response format:
Guess in the format: "guess 1a=red"
You can also delete guesses you believe to be incorrect using "delete 1a"
You can only make 1 guess at a time. If you make more than 1 guess, really bad things will happen.

DO NOT try to call a tool, simply respond with the response format. This is a multi-turn conversation.

## Important:
Do not try solve the entire puzzle at once or reason about what the puzzle will look like.
After each guess, I will show you what the puzzle looks like.
"""

SINGLE_TURN_PROMPT = """
You're a crossword expert.
I will provide you with a 5x5 mini crossword and you should solve the entire puzzle in one go.

## Response format:
Provide all your guesses in a single message using the format: "guess 1a=red"
You can provide multiple guesses separated by commas, like: "guess 1a=red, guess 2d=blue, guess 3a=green"
You can also delete guesses you believe to be incorrect using "delete 1a"

DO NOT try to call a tool, simply respond with the response format. This is a multi-turn conversation.

## Important:
Try to solve the entire puzzle at once. Analyze all the clues together and provide your complete solution.
Think about how the across and down clues intersect and use those intersections to validate your answers.
Provide ALL your answers in ONE message to solve the puzzle as efficiently as possible.
"""
# you can make multiple moves in a single response by comma separating them, like "guess 1a=red, guess 1d=brown". if any of the moves are invalid, then all the moves after that will not be run.

# DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
DEFAULT_MODEL = "groq/openai/gpt-oss-20b"
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
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Use multi-turn mode where the assistant solves 1 clue at a time.",
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


def run_agent(
    mini: MiniCrossword,
    *,
    model: str,
    temperature: float,
    max_turns: int,
    debug: bool,
    reasoning: bool,
    multi_turn: bool,
) -> None:
    moves: List[str] = []
    system_prompt = MULTI_TURN_PROMPT if multi_turn else SINGLE_TURN_PROMPT
    history = [{"role": "system", "content": system_prompt}]

    print_message("system", system_prompt, annotation="prompt")

    starting_message = mini.render(moves)
    print_message("user", starting_message, annotation="turn 0")
    history.append({"role": "user", "content": starting_message})

    turn = 1
    while turn <= max_turns:
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
            print_message("system", f"Model response did not include choices. Stopping. {response}", annotation=f"turn {turn}")
            return

        choice = choices[0]
        assert isinstance(choice, Choices)
        assistant_reply, reasoning_text = choice.message.content, getattr(choice.message, "reasoning", None)
        assistant_reply = (assistant_reply or "").strip()
        if not assistant_reply:
            log_debug(debug, f"empty-choice turn {turn}", choice)
            print_message("system", f"Received empty response from model. Stopping. {response}", annotation=f"turn {turn}")
            return

        if reasoning and reasoning_text:
            print_message("assistant", reasoning_text.strip(), annotation=f"turn {turn} reasoning")

        print_message("assistant", assistant_reply, annotation=f"turn {turn}")
        history.append({"role": "assistant", "content": f"<reasoning>{reasoning_text}</reasoning>\n{assistant_reply}"})

        feedback = mini.play_move(assistant_reply, moves)

        print_message("user", feedback, annotation=f"turn {turn}")
        history.append({"role": "user", "content": feedback})

        if "Puzzle Complete" in feedback:
            print_message("system", "Puzzle completed!", annotation="status")
            break

        turn += 1
    else:
        print_message("system", f"Reached max turns ({max_turns}) without completing the puzzle.", annotation="status")


def main() -> None:
    load_dotenv()
    args = parse_args()
    mini = load_from_remote_or_cache(args.date)
    mini.render_type = "coordinate"
    run_agent(
        mini,
        model=args.model,
        temperature=args.temperature,
        max_turns=args.max_turns,
        debug=args.debug,
        reasoning=args.reasoning,
        multi_turn=args.multi_turn,
    )


if __name__ == "__main__":
    main()
