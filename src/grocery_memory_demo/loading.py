from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from time import monotonic
from typing import TextIO

from grocery_memory_demo import evaluate
from grocery_memory_demo.models import Arm

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class LoadingState:
    completed: int = 0
    scenario_number: int = 0
    scenario_title: str = "Preparing data"
    arm: Arm | None = None
    turn: int = 0


class LoadingScreen:
    def __init__(self, total: int, stream: TextIO = sys.stdout) -> None:
        self.total = total
        self.stream = stream
        self.interactive = bool(getattr(stream, "isatty", lambda: False)())
        self.state = LoadingState()
        self.started_at = monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> LoadingScreen:
        if self.interactive:
            print("\033[?1049h\033[?25l", end="", file=self.stream, flush=True)
            self._draw(self.state, SPINNER[0])
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            print(
                f"Loading demo: 0/{self.total} comparisons complete",
                file=self.stream,
                flush=True,
            )
        return self

    def __exit__(self, error_type: object, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.interactive:
            print("\033[?25h\033[?1049l", end="", file=self.stream, flush=True)
        elif error_type is None:
            print(
                f"Demo ready: {self.state.completed}/{self.total} comparisons complete",
                file=self.stream,
                flush=True,
            )

    def waiting(self, scenario_number: int, scenario_title: str, arm: Arm, turn: int) -> None:
        with self._lock:
            self.state.scenario_number = scenario_number
            self.state.scenario_title = scenario_title
            self.state.arm = arm
            self.state.turn = turn
        if not self.interactive:
            label = evaluate.ARM_LABELS[arm]
            print(
                f"  [{self.state.completed + 1}/{self.total}] "
                f"{arm.value} {label}: waiting for Claude (turn {turn})",
                file=self.stream,
                flush=True,
            )

    def completed(self) -> None:
        with self._lock:
            self.state.completed += 1

    def _animate(self) -> None:
        frame = 0
        while not self._stop.wait(0.1):
            with self._lock:
                state = LoadingState(**vars(self.state))
            self._draw(state, SPINNER[frame % len(SPINNER)])
            frame += 1

    def _draw(self, state: LoadingState, spinner: str) -> None:
        width = 24
        filled = round(width * state.completed / self.total)
        bar = "█" * filled + "░" * (width - filled)
        elapsed = int(monotonic() - self.started_at)
        arm = (
            "Preparing"
            if state.arm is None
            else (f"{state.arm.value} · {evaluate.ARM_LABELS[state.arm]}")
        )
        status = "Loading files" if not state.turn else f"Waiting for Claude · turn {state.turn}"
        scenario = state.scenario_title.split(". ", 1)[-1]
        lines = [
            "GROCERY MEMORY DEMO",
            "",
            f"{spinner} Running live agent comparisons  {elapsed}s",
            f"  [{bar}]  {state.completed}/{self.total}",
            f"  Scenario  {state.scenario_number or '-'} · {scenario}",
            f"  Arm       {arm}",
            f"  Status    {status}",
            "",
            "Live model calls can take a little while.",
        ]
        print("\033[2J\033[H" + "\n".join(lines), end="", file=self.stream, flush=True)
