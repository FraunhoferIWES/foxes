"""Parent-owned calculation progress for terminals, notebooks and text logs."""

from __future__ import annotations

import sys
from time import monotonic
from typing import Any, TextIO


def _in_notebook() -> bool:
    """Detect an active kernel without importing IPython in ordinary scripts."""
    # A running kernel, unlike an installed notebook package, implies notebook IO.
    module = sys.modules.get("IPython")
    shell = module.get_ipython() if module is not None else None
    return shell is not None and getattr(shell, "kernel", None) is not None


class CalculationProgress:
    """One parent-side progress display whose lifetime includes finalization."""

    def __init__(
        self,
        name: str,
        total: int,
        mode: bool | None,
        verbosity: int,
        stream: TextIO | None = None,
    ) -> None:
        """Configure one calculation display.

        Parameters
        ----------
        name
            Engine/model label shown alongside counts and elapsed time.
        total
            Number of state/target chunk tasks.
        mode
            True selects a terminal/notebook bar or bounded log lines when
            redirected; false always selects log lines; ``None`` disables output.
        verbosity
            Zero disables every progress display.
        stream
            Text stream for terminal bars and log lines; ``None`` uses stderr.
        """
        # Keep UI state in the parent, never on a model serialized to workers.
        self.name = name
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.enabled = verbosity > 0 and mode is not None
        self.bar: Any = None
        self.completed = 0
        self._percent = 0
        self._started = monotonic()

        # Let tqdm choose Unicode blocks or ASCII based on the stream's encoding.
        if self.enabled and mode and (_in_notebook() or self.stream.isatty()):
            from tqdm.auto import tqdm

            self.bar = tqdm(
                total=total,
                desc=name,
                unit="chunk",
                file=self.stream,
                dynamic_ncols=True,
                ascii=None,
                mininterval=0.2,
                leave=True,
            )
        elif self.enabled:
            self._line("Starting")

    def _line(self, status: str) -> None:
        """Emit a single newline-terminated, machine-friendly progress snapshot."""
        # Avoid ANSI escapes and carriage-return redraws in redirected output.
        elapsed = monotonic() - self._started
        print(
            f"{self.name}: {status} {self.completed}/{self.total} chunks "
            f"({self._percent}%), elapsed {elapsed:.1f}s",
            file=self.stream,
            flush=True,
        )

    def update(self) -> None:
        """Record one successfully completed chunk, not merely a submitted task."""
        # Update the interactive display or emit at most ten intermediate log lines.
        self.completed += 1
        percent = int(100 * self.completed / max(self.total, 1))
        if self.bar is not None:
            self.bar.update(1)
        elif self.enabled and percent < 100 and percent // 10 > self._percent // 10:
            self._percent = percent
            self._line("Calculated")
        self._percent = percent

    def close(self, error: bool = False) -> None:
        """Close the display after output finalization or exceptional termination.

        Parameters
        ----------
        error
            Mark failure rather than claiming successful output completion.
        """
        # Distinguish chunk calculation from successful result/file finalization.
        if self.bar is not None:
            self.bar.set_description(f"{self.name} - {'failed' if error else 'done'}")
            self.bar.close()
        elif self.enabled:
            self._line("Failed" if error else "Finished")

    def finalizing(self) -> None:
        """Indicate that output assembly and submitted file writes are pending."""
        # A full chunk counter does not imply that file output has succeeded.
        if self.bar is not None:
            self.bar.set_description(f"{self.name} - finalizing")
