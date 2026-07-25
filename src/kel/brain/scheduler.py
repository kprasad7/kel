"""Parallel-to-finish scheduling (DESIGN.md 3.14): run redundant branches
concurrently, stop waiting as soon as one produces a sufficient result.

Honest limitation: Python cannot forcibly interrupt a running thread.
"Cancelling" a losing branch here means the caller stops waiting on it and
its result is discarded — already-started branches keep running to
completion in the background rather than being killed. Real cancellation
would need cooperative checkpoints inside each branch or a process-based
(not thread-based) executor; that's future work, not something to claim
this does today.
"""

from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RaceResult:
    winner: str | None
    result: Any


def race_to_finish(
    branches: dict[str, Callable[[], Any]], *, is_sufficient: Callable[[Any], bool] | None = None
) -> RaceResult:
    is_sufficient = is_sufficient or (lambda _result: True)
    pool = cf.ThreadPoolExecutor(max_workers=max(1, len(branches)))
    futures = {pool.submit(fn): name for name, fn in branches.items()}
    winner_name: str | None = None
    winner_result: Any = None
    try:
        for future in cf.as_completed(futures):
            name = futures[future]
            result = future.result()
            if is_sufficient(result):
                winner_name, winner_result = name, result
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return RaceResult(winner=winner_name, result=winner_result)
