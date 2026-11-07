"""Run the evals against the real agent and model, k times per case, and write a pass^k report.

make eval                            # every scenario, k=5
make eval ARGS="--k 2 --only pick"   # some scenarios, fewer runs
make eval ARGS="--simulate"          # whole conversations with simulated listeners
make eval ARGS="--scripted"          # no API calls: checks the harness itself
"""

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from selecta.infrastructure import config
from selecta.infrastructure.integrations import mistral
from selecta.infrastructure.integrations.mistral import MistralChat
from tests.evals.checks import invariants
from tests.evals.scenarios import LANGUAGES, SCENARIOS, Outcome, Scenario
from tests.evals.simulate import PERSONAS, converse
from tests.evals.world import Player, Scripted, build_agent

OUT_DIR = Path("data/evals")
WORKERS = 8
# A run the model didn't answer (a rate limit, an outage) is tried again before it's reported.
ATTEMPTS = 3
RETRY_WAIT_SECONDS = 20


def run_once(model: Any, scenario: Scenario, language: str, says: str) -> dict[str, Any]:
    """Run one listener turn in a fresh copy of the scenario and return what happened."""
    agent, world = build_agent(model)
    player = Player(works=scenario.player_works)
    session = scenario.setup(language)
    pick_before = session.pick
    started = time.monotonic()
    agent.reply(session, says, play=player)  # type: ignore[arg-type]
    turn = session.turns[-1]
    if turn.error:
        return {"error": turn.error}
    outcome = Outcome(language, session, turn, world, player, pick_before)
    return {
        "failures": scenario.check(outcome) + invariants(outcome),
        "reply": turn.reply,
        "tools": [{"name": r.call.name, "args": r.call.arguments} for r in turn.tool_runs],
        "rounds": turn.model_rounds,
        "tokens": turn.input_tokens + turn.output_tokens,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    """Run the selected scenarios or conversations and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="runs per case")
    parser.add_argument("--only", default="", help="only names that contain this text")
    parser.add_argument("--simulate", action="store_true", help="whole simulated conversations")
    parser.add_argument("--scripted", action="store_true", help="no API calls, harness only")
    args = parser.parse_args()
    if args.scripted:
        model: Any = Scripted("Sounds good. What else?")
        listener: Any = Scripted("Something calm, please.")
    else:
        if not config.MISTRAL.API_KEY:
            raise SystemExit("MISTRAL_API_KEY is not set: evals call the real model")
        model = MistralChat(mistral.connect(config.MISTRAL.API_KEY), config.MISTRAL.AGENT_MODEL)
        listener = model
    if args.simulate:
        personas = {p.name: p for p in PERSONAS}
        jobs = [
            {"case": name, "language": lang}
            for name in personas
            if args.only in name
            for lang in LANGUAGES
            for _ in range(args.k)
        ]
        runs = _execute(
            jobs, lambda j: converse(model, listener, personas[j["case"]], j["language"])
        )
        _write("simulated", _simulated_report(runs, model.model, args.k), runs)
    else:
        scenarios = {s.name: s for s in SCENARIOS}
        jobs = [
            {"case": s.name, "capability": s.capability, "language": lang, "says": says}
            for s in SCENARIOS
            if args.only in s.name
            for lang in LANGUAGES
            for says in s.says[lang]
            for _ in range(args.k)
        ]
        runs = _execute(
            jobs, lambda j: run_once(model, scenarios[j["case"]], j["language"], j["says"])
        )
        _write("turns", _report(runs, model.model, args.k), runs)


def _execute(
    jobs: list[dict[str, Any]], run: Callable[[dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run every job in parallel, retrying runs the model didn't answer, and merge the results."""

    def attempt(job: dict[str, Any]) -> dict[str, Any]:
        for n in range(ATTEMPTS):
            try:
                result = run(job)
            except Exception as e:
                # One crashed run is a finding, not a reason to lose the others.
                result = {"failures": [f"crashed: {type(e).__name__}: {e}"]}
            if "error" not in result:
                break
            time.sleep(RETRY_WAIT_SECONDS * (n + 1))
        return {**job, **result}

    print(f"{len(jobs)} runs")
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(attempt, jobs))


def _write(kind: str, report: str, runs: list[dict[str, Any]]) -> None:
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs))
    (out / "report.md").write_text(report)
    print(report)
    print(f"written to {out}")


def _passed(run: dict[str, Any]) -> bool:
    return "error" not in run and not run["failures"]


def _report(runs: list[dict[str, Any]], model: str, k: int) -> str:
    cases: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for r in runs:
        cases.setdefault((r["case"], r["language"]), {}).setdefault(r["says"], []).append(r)
    rows, total, at_k = [], 0, 0
    for scenario in dict.fromkeys(r["case"] for r in runs):
        capability = next(r["capability"] for r in runs if r["case"] == scenario)
        cells, ok_here, count_here = [], 0, 0
        for lang in LANGUAGES:
            said = list(cases.get((scenario, lang), {}).values())
            ok = sum(all(_passed(r) for r in group) for group in said)
            good = sum(_passed(r) for group in said for r in group)
            cells.append(f"{ok}/{len(said)} · {good}/{sum(len(g) for g in said)}")
            ok_here, count_here = ok_here + ok, count_here + len(said)
        at_k, total = at_k + ok_here, total + count_here
        rows.append(f"| {capability} | {scenario} | {' | '.join(cells)} | {ok_here}/{count_here} |")
    lines = [
        f"# Eval report: {model}, k={k}",
        "",
        "A case is one scenario, said one way, in one language. pass^k: the case passed in all "
        f"{k} runs. Each cell shows cases at pass^k / cases, then runs passed / runs.",
        "",
        "| capability | scenario | en | es | fr | pass^k |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"**pass^{k}: {at_k}/{total} cases.**",
        *_failures(runs),
    ]
    if answered := [r for r in runs if "error" not in r]:
        seconds = sorted(r["seconds"] for r in answered)
        p90 = statistics.quantiles(seconds, n=10)[-1] if len(seconds) > 1 else seconds[0]
        lines += [
            "",
            "## Latency and tokens",
            "",
            f"- turn latency: median {statistics.median(seconds):.1f} s, p90 {p90:.1f} s",
            f"- model rounds per turn: {statistics.mean(r['rounds'] for r in answered):.1f}",
            f"- tokens per turn: {statistics.mean(r['tokens'] for r in answered):,.0f}",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        f"- {len(rows)} scenarios, one turn each, on a synthetic 24-record collection; k={k}.",
        "- Checks look at tools, arguments and session state. The reply's language and a claim "
        "that something is playing are read with simple word rules, so they can miss or misread "
        "a reply.",
        "- Not measured: whether facts the agent says are true beyond the checked ones, tone, "
        "and anything about speech (transcription, voices, timing).",
    ]
    return "\n".join(lines) + "\n"


def _simulated_report(runs: list[dict[str, Any]], model: str, k: int) -> str:
    rows = []
    for persona in dict.fromkeys(r["case"] for r in runs):
        mine = [r for r in runs if r["case"] == persona]
        cells = []
        for lang in LANGUAGES:
            group = [r for r in mine if r["language"] == lang]
            ok = sum(_passed(r) for r in group)
            cells.append(f"{ok}/{len(group)}{' pass^k' if group and ok == len(group) else ''}")
        answered = [r for r in mine if "error" not in r] or [{"turns": 0, "tokens": 0}]
        turns = statistics.mean(r["turns"] for r in answered)
        tokens = statistics.mean(r["tokens"] for r in answered)
        rows.append(f"| {persona} | {' | '.join(cells)} | {turns:.1f} | {tokens:,.0f} |")
    lines = [
        f"# Simulated conversations: {model}, k={k}",
        "",
        "A model plays the listener. A conversation passes when a record plays within 10 turns, "
        "the persona's own goal is met and no turn breaks an invariant. Cells: conversations "
        "passed / run, and pass^k when all of them passed.",
        "",
        "| persona | en | es | fr | turns to music | tokens per conversation |",
        "|---|---|---|---|---|---|",
        *rows,
        *_failures(runs),
        "",
        "## Limitations",
        "",
        "- The listener is a model following a short persona, not a person: real listeners are "
        "messier, and the same model plays both sides.",
        "- Tokens are the agent's only, not the simulated listener's.",
    ]
    return "\n".join(lines) + "\n"


def _failures(runs: list[dict[str, Any]]) -> list[str]:
    reasons = Counter((r["case"], f) for r in runs for f in r.get("failures", []))
    errors = sum("error" in r for r in runs)
    lines = ["", "## Failures", ""]
    lines += [f"- {case}: {reason} ({n} runs)" for (case, reason), n in reasons.most_common()]
    if errors:
        lines.append(
            f"- {errors} runs got no answer from the model after {ATTEMPTS} tries: they count as "
            "not passed."
        )
    return lines


if __name__ == "__main__":
    main()
