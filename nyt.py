#!/usr/bin/env python3
"""
nyt.py – download and render a NYT Mini crossword in the
“Crossword Puzzle Serialization” markdown format used in our chat.
"""

import argparse
import json
from dataclasses import asdict, field
from dataclasses import dataclass as std_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic.dataclasses import dataclass as pydantic_dataclass

API_ENDPOINT = "https://www.nytimes.com/svc/crosswords/v6/puzzle/mini/{iso_date}.json"
API_HEADERS = {
    "x-games-auth-bypass": "true",
    "user-agent": "Mozilla/5.0 (compatible; CrosswordFetcher/1.0)"
}


@std_dataclass
class ParsedMove:
    action: str
    clue_key: str
    guess: str | None = None


@pydantic_dataclass
class MiniCell:
    answer: str | None = None
    label: str | None = None
    clues: list[int] = field(default_factory=list)

    def is_black(self) -> bool:
        return self.answer is None and not self.label and not self.clues


@pydantic_dataclass
class MiniClue:
    label: str
    direction: str
    cells: list[int]
    hint: str


@pydantic_dataclass
class MiniCrossword:
    date: str
    width: int
    height: int
    cells: list[MiniCell]
    clues: list[MiniClue]
    def __post_init_post_parse__(self) -> None:
        if self.width is None or self.height is None:
            raise ValueError("Puzzle dimensions are required.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Puzzle dimensions must be positive integers.")

        expected_cells = self.width * self.height
        if len(self.cells) != expected_cells:
            raise ValueError(
                f"Puzzle cell count mismatch: expected {expected_cells}, got {len(self.cells)}."
            )

    @staticmethod
    def _format_label(label: str) -> str:
        if label.isdigit() and int(label) < 10:
            return f"0{label}"
        return label

    @staticmethod
    def _normalize_clue_key(raw: str) -> str:
        key = raw.replace(" ", "").upper()
        if len(key) < 2 or not key[:-1].isdigit() or key[-1] not in {"A", "D"}:
            raise ValueError("Clue id must be like '1A' or '12D'.")
        return key

    def _clue_key(self, clue: MiniClue) -> str:
        return f"{clue.label}{clue.direction[0].upper()}"

    def _clue_map(self) -> dict[str, MiniClue]:
        return {self._clue_key(clue): clue for clue in self.clues}

    def _parse_move(self, raw_move: str) -> ParsedMove:
        move = raw_move.strip()
        if not move:
            raise ValueError("Move cannot be empty.")

        parts = move.split(None, 1)

        if "=" in move and (len(parts) == 1 or parts[0].lower() not in {"guess", "delete"}):
            clue_part, _, guess_part = move.partition("=")
            clue_key = self._normalize_clue_key(clue_part)
            guess = guess_part.strip().replace(" ", "").upper()
            if not guess:
                raise ValueError("Guess must include letters.")
            return ParsedMove("guess", clue_key, guess)

        if len(parts) == 2 and parts[0].lower() not in {"guess", "delete", "del"}:
            clue_key = self._normalize_clue_key(parts[0])
            guess = parts[1].strip().replace(" ", "").replace("=", "").upper()
            if not guess:
                raise ValueError("Guess must include letters.")
            return ParsedMove("guess", clue_key, guess)

        if len(parts) != 2:
            raise ValueError("Move must include an action and a clue, e.g. 'guess 1A=WORD'.")

        action, rest = parts[0].lower(), parts[1]

        if action == "del":
            action = "delete"

        if action == "guess":
            if "=" in rest:
                clue_part, _, guess_part = rest.partition("=")
            else:
                sub_parts = rest.split(None, 1)
                if len(sub_parts) != 2:
                    raise ValueError(
                        "Guess must be in the form 'guess <clue>=<answer>' or 'guess <clue> <answer>'."
                    )
                clue_part, guess_part = sub_parts
            clue_key = self._normalize_clue_key(clue_part)
            guess = guess_part.strip().replace(" ", "").upper()
            if not guess:
                raise ValueError("Guess must include letters.")
            return ParsedMove("guess", clue_key, guess)

        if action == "delete":
            clue_key = self._normalize_clue_key(rest)
            return ParsedMove("delete", clue_key)

        raise ValueError("Unknown move action. Use 'guess' or 'delete'.")

    def _grid_layout(self, letters: list[str | None] | None) -> list[str]:
        rows: list[list[str]] = [[] for _ in range(self.height)]

        for idx, cell in enumerate(self.cells):
            r, _ = divmod(idx, self.width)
            if cell.is_black():
                rows[r].append("##")
                continue

            letter = None if letters is None else letters[idx]
            if letter:
                rows[r].append(f" {letter}")
            elif cell.label:
                rows[r].append(self._format_label(cell.label))
            else:
                rows[r].append("..")

        return ["".join(f" {token}" for token in row) for row in rows]

    def _format_clue_lines(self) -> tuple[list[str], list[str]]:
        def format_direction(direction: str) -> list[str]:
            clues = [c for c in self.clues if c.direction == direction]
            clues.sort(key=lambda c: int(c.label))
            lines: list[str] = []
            for clue in clues:
                length = len(clue.cells)
                noun = "letter" if length == 1 else "letters"
                key = self._clue_key(clue)
                lines.append(f"{key}: {clue.hint} ({length} {noun})")
            return lines

        return format_direction("Across"), format_direction("Down")

    def _print_grid_and_clues(
        self,
        grid_lines: list[str],
        across: list[str],
        down: list[str],
        blank_lines: list[str] | None = None,
    ) -> None:
        print("# Crossword Puzzle Serialization\n")
        print(f"## Grid ({self.height}x{self.width})")
        print("Legend:")
        print("- `##` = black square")
        print("- `..` = empty white square")
        print("- Number = empty white square that is the start of a clue, either down or across\n")
        print("Grid layout:")
        for row in grid_lines:
            print(row)
        print("\n## Clues\n")
        print("### Across")
        for clue in across:
            print(clue + "  ")
        print("\n### Down")
        for clue in down:
            print(clue + "  ")

    def _compute_board_state(self, moves: list[str]) -> tuple[list[str | None], int, int, int]:
        letters: list[str | None] = [None] * len(self.cells)
        clue_map = self._clue_map()

        for move in moves:
            try:
                parsed = self._parse_move(move)
            except ValueError:
                continue

            clue = clue_map.get(parsed.clue_key)
            if clue is None:
                continue

            if parsed.action == "guess" and parsed.guess is not None:
                guess_letters = list(parsed.guess)
                if len(guess_letters) != len(clue.cells):
                    continue
                for offset, cell_idx in enumerate(clue.cells):
                    letters[cell_idx] = guess_letters[offset]
            elif parsed.action == "delete":
                for cell_idx in clue.cells:
                    letters[cell_idx] = None

        total = sum(1 for cell in self.cells if not cell.is_black())
        filled = sum(
            1
            for idx, cell in enumerate(self.cells)
            if not cell.is_black() and letters[idx] is not None
        )
        correct = sum(
            1
            for idx, cell in enumerate(self.cells)
            if not cell.is_black()
            and letters[idx] is not None
            and cell.answer is not None
            and letters[idx] == cell.answer.upper()
        )

        return letters, filled, correct, total

    def render_starting_puzzle(self) -> None:
        grid_lines = self._grid_layout(letters=None)
        across, down = self._format_clue_lines()
        self._print_grid_and_clues(grid_lines, across, down)

    def render(self, moves: list[str], *, show_progress: bool = True) -> tuple[int, int, int]:
        letters, filled, correct, total = self._compute_board_state(moves)
        grid_lines = self._grid_layout(letters)
        blank_lines = self._grid_layout(letters=None) if any(letters) else None
        across, down = self._format_clue_lines()
        self._print_grid_and_clues(grid_lines, across, down, blank_lines=blank_lines)
        if show_progress:
            print(f"\nProgress: {filled}/{total} filled, {correct}/{total} correct")
        return filled, correct, total

    def play_move(self, raw_move: str, moves: list[str]) -> None:
        parsed = self._parse_move(raw_move)
        clue_map = self._clue_map()
        clue = clue_map.get(parsed.clue_key)

        if clue is None:
            raise ValueError(f"Unknown clue '{parsed.clue_key}'.")

        length = len(clue.cells)

        if parsed.action == "guess":
            if parsed.guess is None:
                raise ValueError("Guess requires an answer.")
            guess = parsed.guess
            if len(guess) != length:
                noun = "letter" if length == 1 else "letters"
                raise ValueError(f"{parsed.clue_key} is {length} {noun} long.")
            if not guess.isalpha():
                raise ValueError("Guesses must contain letters only.")
            canonical = f"guess {parsed.clue_key}={guess}"
        else:
            canonical = f"delete {parsed.clue_key}"

        moves.append(canonical)



def load_remote_json(date_text: str) -> dict:
    """Fetch the NYT Mini JSON for the provided MM-DD-YYYY date."""
    try:
        parsed_date = datetime.strptime(date_text, "%m-%d-%Y").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{date_text}'. Expected MM-DD-YYYY.") from exc

    url = API_ENDPOINT.format(iso_date=parsed_date.isoformat())
    request = Request(url, headers=API_HEADERS)

    try:
        with urlopen(request) as response:
            return json.load(response)
    except HTTPError as err:
        raise SystemExit(
            f"Failed to download puzzle for {parsed_date:%B %d, %Y}: HTTP {err.code}"
        ) from err
    except URLError as err:
        raise SystemExit(
            f"Failed to download puzzle for {parsed_date:%B %d, %Y}: {err.reason}"
        ) from err

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a NYT Mini crossword for a given date.")
    parser.add_argument("date", help="Date of the puzzle in MM-DD-YYYY format.")
    return parser.parse_args()


def build_mini_from_api(date_text: str, api_payload: dict[str, Any]) -> MiniCrossword:
    try:
        puzzle = api_payload["body"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit("Unexpected NYT API response; missing mini puzzle data.") from exc

    dimensions = puzzle.get("dimensions")
    if not dimensions:
        raise SystemExit("Unexpected NYT API response; missing dimensions for puzzle.")
    width = dimensions.get("width")
    height = dimensions.get("height")
    if width is None or height is None:
        raise SystemExit("Unexpected NYT API response; incomplete puzzle dimensions.")

    cells_raw = puzzle.get("cells")
    clues_raw = puzzle.get("clues")
    if cells_raw is None or clues_raw is None:
        raise SystemExit("Unexpected NYT API response; puzzle cells or clues missing.")

    cells = []
    for cell in cells_raw:
        if not cell:
            cells.append(MiniCell())
            continue
        answer = cell.get("answer")
        answer = answer.upper() if isinstance(answer, str) else None
        cells.append(
            MiniCell(
                answer=answer,
                label=cell.get("label"),
                clues=cell.get("clues", []),
            )
        )

    clues = [
        MiniClue(
            label=clue["label"],
            direction=clue["direction"],
            cells=clue.get("cells", []),
            hint="".join(segment.get("plain", "") for segment in clue.get("text", [])),
        )
        for clue in clues_raw
    ]

    return MiniCrossword(
        date=date_text,
        width=width,
        height=height,
        cells=cells,
        clues=clues,
    )


def load_cached_mini(path: Path) -> MiniCrossword:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        cells_data = payload["cells"]
        clues_data = payload["clues"]
    except KeyError as exc:
        raise SystemExit(f"Cached puzzle at {path} is missing required field: {exc}") from exc

    converted = {
        "date": payload["date"],
        "width": payload["width"],
        "height": payload["height"],
        "cells": [
            MiniCell(
                answer=(cell.get("answer").upper() if isinstance(cell.get("answer"), str) else None),
                label=cell.get("label"),
                clues=cell.get("clues", []),
            )
            if cell
            else MiniCell()
            for cell in cells_data
        ],
        "clues": [
            MiniClue(
                label=clue["label"],
                direction=clue["direction"],
                cells=clue.get("cells", []),
                hint=clue.get("hint")
                if isinstance(clue.get("hint"), str)
                else "".join(
                    segment.get("plain", "") for segment in clue.get("text", [])
                ),
            )
            for clue in clues_data
        ],
    }

    return MiniCrossword(**converted)


def main():
    args = parse_args()
    cache_path = Path("minis") / f"{args.date}.json"

    moves: list[str] = []

    if cache_path.exists():
        mini = load_cached_mini(cache_path)
        print(f"Loaded cached puzzle for {args.date} from {cache_path}.")
    else:
        data = load_remote_json(args.date)
        mini = build_mini_from_api(args.date, data)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(mini), f, ensure_ascii=False, indent=2)
        print(f"Downloaded puzzle for {args.date} and cached to {cache_path}.")

    print("\nStarting layout:\n")
    try:
        filled, correct, total = mini.render(moves, show_progress=False)
    except KeyboardInterrupt:
        print("\nExiting. Progress saved.")
        return

    if correct == total:
        print("\nPuzzle complete! 🎉")
        return

    prompt = "\033[94mEnter move (guess <clue>=<answer> / delete <clue>) or Ctrl-D to exit: \033[0m"

    while True:
        try:
            move_input = input(prompt)
        except EOFError:
            print("\nExiting. Progress saved.")
            break
        except KeyboardInterrupt:
            print("\nExiting. Progress saved.")
            break

        move_input = move_input.strip()
        if not move_input:
            continue

        segments = [part.strip() for part in move_input.split(",") if part.strip()]
        if not segments:
            continue

        puzzle_complete = False
        invalid_move = False

        for segment in segments:
            try:
                mini.play_move(segment, moves)
            except ValueError as exc:
                print(f"Invalid move '{segment}': {exc}")
                invalid_move = True
                break

            letters, filled, correct, total = mini._compute_board_state(moves)
            grid_lines = mini._grid_layout(letters)

            try:
                last = mini._parse_move(moves[-1])
            except ValueError:
                last = ParsedMove("guess", "", None)

            if last.action == "delete":
                print("Clue cleared!\n")
            else:
                print("Valid guess!\n")

            print("Grid layout:")
            for row in grid_lines:
                print(row)

            if correct == total:
                print("\nPuzzle complete! 🎉")
                puzzle_complete = True
                break

            print("\nPlease submit another clue.")

        if puzzle_complete:
            break
        if invalid_move:
            continue

if __name__ == "__main__":
    main()
