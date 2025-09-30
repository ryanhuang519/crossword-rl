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
    render_type: str | None = None
    def __post_init_post_parse__(self) -> None:
        if self.width is None or self.height is None:
            raise ValueError("Puzzle dimensions are required.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Puzzle dimensions must be positive integers.")

        if self.render_type is not None:
            normalized = self.render_type.lower()
            if normalized not in {"coordinate"}:
                raise ValueError(
                    "render_type must be one of: 'coordinate'."
                )
            self.render_type = normalized

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
        if self.render_type == "coordinate":
            return self._coordinate_layout(letters)

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

    def _coordinate_layout(self, letters: list[str | None] | None) -> list[str]:
        lines: list[str] = []
        for row_idx in range(self.height):
            parts: list[str] = []
            for col_idx in range(self.width):
                cell_index = row_idx * self.width + col_idx
                cell = self.cells[cell_index]
                label = cell.label
                if cell.is_black():
                    value = "black"
                else:
                    letter = None if letters is None else letters[cell_index]
                    if letter:
                        value = letter.upper()
                    elif label:
                        value = self._format_label(label)
                    else:
                        value = "."
                parts.append(f"col{col_idx + 1} {value}")
            row_line = f"Row{row_idx + 1}: {', '.join(parts)}"
            if not row_line.endswith("."):
                row_line += "."
            lines.append(row_line)
        return lines

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

    def _build_grid_and_clues_lines(
        self,
        grid_lines: list[str],
        across: list[str],
        down: list[str],
    ) -> list[str]:
        lines = [
            "# Crossword Puzzle Serialization",
            f"## Grid ({self.height}x{self.width})",
            "Legend:",
        ]

        if self.render_type == "coordinate":
            lines.extend(
                [
                    "- `black` = black square",
                    "- Number = clue label for the cell",
                    "- `.` = empty white square without a label",
                    "- Letter = filled entry",
                ]
            )
        else:
            lines.extend(
                [
                    "- `##` = black square",
                    "- `..` = empty white square",
                    "- Number = empty white square that is the start of a clue, either down or across",
                ]
            )

        lines.append("Grid layout:")

        lines.extend(grid_lines)
        lines.extend(["", "## Clues", "", "### Across"])

        for clue in across:
            lines.append(f"{clue}  ")

        lines.extend(["", "### Down"])

        for clue in down:
            lines.append(f"{clue}  ")

        return lines

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


    def render(self, moves: list[str]) -> str:
        letters, filled, correct, total = self._compute_board_state(moves)
        grid_lines = self._grid_layout(letters)
        across, down = self._format_clue_lines()

        lines = self._build_grid_and_clues_lines(grid_lines, across, down)

        return "\n".join(lines)

    def play_move(self, move_input: str, moves: list[str]) -> str:
        stripped = move_input.strip()
        if not stripped:
            return ""

        segments = [part.strip() for part in stripped.split(",") if part.strip()]
        if not segments:
            letters, filled, correct, total = self._compute_board_state(moves)
            grid_lines = self._grid_layout(letters)
            lines: list[str] = [
                "No valid moves detected. Respond with moves like 'guess 1A=word' or 'delete 1D'.",
                "",
                "Grid layout:",
            ]
            lines.extend(grid_lines)
            lines.append("")
            if correct == total and total > 0:
                lines.append("Puzzle Complete! 🎉")
            return "\n".join(lines)

        clue_map = self._clue_map()
        failures: list[str] = []

        def record_failure(segment: str, message: str) -> None:
            failures.append(f"Invalid Move '{segment}': {message}")

        one_success = False
        for segment in segments:
            try:
                parsed = self._parse_move(segment)
                one_success = True
            except ValueError as exc:
                record_failure(segment, str(exc))
                continue

            clue = clue_map.get(parsed.clue_key)
            if clue is None:
                record_failure(segment, f"Unknown clue '{parsed.clue_key}'.")
                continue

            length = len(clue.cells)

            if parsed.action == "guess":
                if parsed.guess is None:
                    record_failure(segment, "Guess requires an answer.")
                    continue
                guess = parsed.guess
                if len(guess) != length:
                    noun = "letter" if length == 1 else "letters"
                    record_failure(segment, f"{parsed.clue_key} is {length} {noun} long.")
                    continue
                if not guess.isalpha():
                    record_failure(segment, "Guesses must contain letters only.")
                    continue
                canonical = f"guess {parsed.clue_key}={guess}"
            else:
                canonical = f"delete {parsed.clue_key}"

            moves.append(canonical)

        letters, filled, correct, total = self._compute_board_state(moves)
        grid_lines = self._grid_layout(letters)

        final_lines: list[str] = []
        if failures:
            final_lines.append("Failed move(s):")
            final_lines.extend(failures)
            final_lines.append("")

        if one_success:
            final_lines.append("Valid guess!\n")
        final_lines.append("Grid layout:")
        final_lines.extend(grid_lines)

        if correct == total:
            final_lines.append("")
            final_lines.append("Puzzle Complete! 🎉")
        else:
            final_lines.append("")
            final_lines.append("Please submit another guess.")

        return "\n".join(final_lines)



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
        "render_type": payload.get("render_type"),
    }

    return MiniCrossword(**converted)

def load_from_remote_or_cache(date_text: str) -> MiniCrossword:
    cache_path = Path("minis") / f"{date_text}.json"

    if cache_path.exists():
        mini = load_cached_mini(cache_path)
        print(f"Loaded cached puzzle for {date_text} from {cache_path}.")
    else:
        data = load_remote_json(date_text)
        mini = build_mini_from_api(date_text, data)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(mini), f, ensure_ascii=False, indent=2)
        print(f"Downloaded puzzle for {date_text} and cached to {cache_path}.")
    return mini

def main():
    args = parse_args()
    mini = load_from_remote_or_cache(args.date)
    mini.render_type = "coordinate"

    moves: list[str] = []
    starting_view = mini.render(moves)
    print(starting_view)

    prompt = "\033[94mEnter move (guess <clue>=<answer> / delete <clue>) or Ctrl-D to exit: \033[0m"

    while True:
        try:
            move_input = input(prompt)
        except EOFError:
            print("\nExiting.")
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        response = mini.play_move(move_input, moves)
        if not response:
            continue

        print(response)

        if "Puzzle Complete" in response:
            break

if __name__ == "__main__":
    main()
