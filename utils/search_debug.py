"""
Tracing for the question "why is this phase not in my results?".

A search drops candidates at a dozen independent gates — condition filters,
screening floors, pool truncation, score thresholds, name dedupe — and every
one of them is silent. When a phase the user knows is present fails to appear,
there is no way to tell which gate rejected it, or whether it was ever scored
at all.

A trace records what each stage did and follows named phases through every
gate, so a search can report the one line that matters: the gate that rejected
the phase, and the setting that controls it.

Tracing is off unless a caller opens one, and the module-level helpers are
no-ops in that case, so instrumented code paths cost nothing in normal use.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from typing import Dict, Iterable, List, Optional


# Every way a candidate can leave the pipeline, in the order it is applied,
# paired with the control that decides it. Reporting the control by the name
# on its widget is the point: a diagnosis the user cannot act on is useless.
GATES: Dict[str, str] = {
    "not_indexed": "no diffraction pattern in the search index",
    "ambient": "excluded by the 'Ambient only' filter",
    "screen_coverage": "screened out by 'Screen min coverage'",
    "pool_size": "screened in, but ranked below the 'Pool size' cut",
    "no_reference": "no reference pattern could be loaded",
    "no_lines_in_range": "no lines above 'Min line intensity' inside the scan range",
    "shifted_out": "every line moved outside the scan range by the 2θ shift",
    "min_found": "too few lines found for 'Min lines found'",
    "min_score": "scored below 'Min fingerprint'",
    "require_top": "strongest line missing, with 'Require strongest line' on",
    "duplicate_record": "outscored by another record of the same mineral",
    "max_results": "scored above the threshold but ranked below 'Max results'",
    "kept": "kept",
}


# The same fates as whole sentences, for reporting one phase's outcome. The
# labels above are written to be read down a column of counts, which makes them
# unreadable in a sentence: "quartz was strongest line missing".
GATE_REASONS: Dict[str, str] = {
    "not_indexed": "{phase} is in the database, but no diffraction pattern has "
                   "ever been calculated for it, so the search had nothing to "
                   "compare against your peaks.",
    "ambient": "{phase} is in the database, but the 'Ambient only' filter set "
               "it aside before scoring as a high-pressure or high-temperature "
               "structure.",
    "screen_coverage": "{phase} reached the first quick screening pass and was "
                       "dropped there: too few of its lines landed on your "
                       "peaks to clear 'Screen min coverage'.",
    "pool_size": "{phase} cleared screening but never made the shortlist that "
                 "gets scored in full — other candidates screened better and "
                 "filled 'Pool size' first.",
    "no_reference": "{phase} was picked for scoring, but its reference pattern "
                    "could not be loaded from the database.",
    "no_lines_in_range": "{phase} was scored, but none of its lines above "
                         "'Min line intensity' fall inside the 2θ range you "
                         "measured.",
    "shifted_out": "{phase} was scored, but the '2θ shift' in use moved every "
                   "one of its lines outside your scan range.",
    "min_found": "{phase} was scored, and too few of its lines found a peak to "
                 "satisfy 'Min lines found'.",
    "min_score": "{phase} was scored against your peaks and came out below "
                 "'Min fingerprint'.",
    "require_top": "{phase} was scored, but its strongest line found no peak "
                   "and 'Require strongest line present' is switched on.",
    "duplicate_record": "{phase} was found and scored, but another database "
                        "record of the same mineral scored higher and took its "
                        "place in the list.",
    "max_results": "{phase} passed every threshold and was then cut for list "
                   "length alone: 'Max results' filled up before it.",
    "kept": "{phase} passed every gate in the search.",
}


# The control to reach for once a gate has been named. A user who has just been
# told which gate rejected their phase still has to guess what to do about it,
# and guessing at a dozen interacting thresholds is exactly the problem this
# module exists to remove.
GATE_FIXES: Dict[str, str] = {
    "not_indexed": "No search setting can bring it back — the record needs a "
                   "pattern calculated for it first.",
    "ambient": "Uncheck 'Ambient only' if that is really how your sample was "
               "measured.",
    "screen_coverage": "Lower 'Screen min coverage' in Options.",
    "pool_size": "Raise 'Pool size' in Options.",
    "no_reference": "Try another database record of the same mineral.",
    "no_lines_in_range": "Lower 'Min line intensity' in Options, or collect a "
                         "wider 2θ range.",
    "shifted_out": "Reduce '2θ shift' towards zero.",
    "min_found": "Lower 'Min lines found' in Options, or widen '2θ tol' so "
                 "more reference lines reach a peak.",
    "min_score": "Either lower 'Min fingerprint' or improve the fit: widen "
                 "'2θ tol', or set 'Auto fit ±' to 0.30° if the pattern is "
                 "displaced.",
    "require_top": "Turn that setting off, or check whether peak finding "
                   "missed the peak it wanted.",
    "duplicate_record": "Nothing to change — look for that other record in the "
                        "results instead.",
    "max_results": "Raise 'Max results' and search again.",
}


def phase_key(name) -> str:
    """Normalized mineral name, matching the dedupe key used when ranking."""
    return str(name or "").strip().lower()


class SearchTrace:
    """
    Record of one search: what each stage did, and where phases were dropped.

    `watch` names are followed individually; everything else only contributes
    to the per-gate totals, which is what keeps a 3000-candidate search cheap
    to trace.
    """

    def __init__(self, watch: Iterable[str] = ()):
        self.watch = {phase_key(n) for n in watch if phase_key(n)}
        self.stages: List[str] = []
        self.gates: Counter = Counter()
        self.followed: Dict[str, List[dict]] = {}

    def watching(self, name) -> bool:
        """Whether this phase is followed individually."""
        if not self.watch:
            return False
        key = phase_key(name)
        return any(key == w or w in key for w in self.watch)

    def stage(self, text: str) -> None:
        self.stages.append(text)

    def gate(self, gate: str, name=None, **detail) -> None:
        """Record a candidate's fate; `detail` is kept only for watched names."""
        self.gates[gate] += 1
        self.follow(gate, name, **detail)

    def bulk(self, gate: str, count: int) -> None:
        """Add to a gate's total without per-candidate detail.

        Stages that reject thousands of phases at once count them this way, so
        the fate table stays a true funnel over the whole database rather than
        a tally of the few phases being followed.
        """
        if count:
            self.gates[gate] += int(count)

    def follow(self, gate: str, name=None, **detail) -> None:
        """Record detail for a watched phase without touching the totals."""
        if name is not None and self.watching(name):
            self.followed.setdefault(phase_key(name), []).append(
                {"name": str(name), "gate": gate, **detail}
            )

    # --- reporting ---

    def gate_summary(self) -> List[str]:
        """Per-gate counts in pipeline order, skipping gates nothing hit."""
        lines = []
        for gate, why in GATES.items():
            n = self.gates.get(gate, 0)
            if n:
                lines.append(f"    {n:6d}  {why}")
        return lines

    def verdict(self, name) -> Optional[dict]:
        """The best fate recorded for a watched phase, or None if never seen."""
        records = self.followed.get(phase_key(name))
        if not records:
            return None
        order = list(GATES)
        # The record that got furthest is the one that explains the outcome
        return max(records, key=lambda r: order.index(r["gate"])
                   if r["gate"] in order else -1)

    def report(self, name=None, heading: str = "SEARCH INTERNALS") -> str:
        """Raw stage log and per-gate funnel, for checking the summary above."""
        lines = [heading, "-" * 60,
                 "  What the search did, stage by stage:"]
        lines += [f"    {s}" for s in self.stages]
        if self.gates:
            total = sum(self.gates.values())
            lines += ["",
                      f"  Where all {total:,} candidates considered ended up:"]
            lines += self.gate_summary()
        for key in sorted(self.followed):
            records = self.followed[key]
            best = self.verdict(key)
            lines += ["",
                      f"  Database records named {records[0]['name']} that the "
                      f"search touched: {len(records)}"]
            lines.append("    furthest any of them got: "
                         f"{GATES.get(best['gate'], best['gate'])}")
            for r in sorted(records, key=lambda r: -(r.get("score") or 0))[:8]:
                bits = [f"id={r.get('mineral_id', '?')}"]
                for field in ("coverage", "score", "n_found", "n_expected"):
                    value = r.get(field)
                    if value is None:
                        continue
                    shown = f"{value:.3f}" if isinstance(value, float) else value
                    bits.append(f"{field}={shown}")
                bits.append(GATES.get(r["gate"], r["gate"]))
                lines.append("      " + "  ".join(bits))
        if name and not self.followed.get(phase_key(name)):
            lines += ["",
                      f"  No record named {name} ever entered the search index, "
                      "so no gate had a chance to reject it."]
        return "\n".join(lines)


_active: Optional[SearchTrace] = None


def active() -> Optional[SearchTrace]:
    return _active


@contextlib.contextmanager
def tracing(watch: Iterable[str] = ()):
    """Collect a trace for the search running inside this block."""
    global _active
    previous = _active
    _active = SearchTrace(watch)
    try:
        yield _active
    finally:
        _active = previous


# --- no-op helpers so instrumented code stays a single line -----------------

def stage(text: str) -> None:
    if _active is not None:
        _active.stage(text)


def gate(gate_name: str, name=None, **detail) -> None:
    if _active is not None:
        _active.gate(gate_name, name, **detail)


def bulk(gate_name: str, count: int) -> None:
    if _active is not None:
        _active.bulk(gate_name, count)


def follow(gate_name: str, name=None, **detail) -> None:
    if _active is not None:
        _active.follow(gate_name, name, **detail)


def watching(name) -> bool:
    return _active is not None and _active.watching(name)


def enabled() -> bool:
    return _active is not None
