"""Run the Gemini investigation pipeline once, with full token accounting.

This is the first script in the project that spends money. It therefore reports token
usage and an estimated cost for every run, so budget is measured rather than assumed.

    uv run python scripts/run_agent.py --incident INC-APP-ROKU-820

Deterministic DETECT runs first (no model), then the Workflow graph
INVESTIGATE -> CORRELATE -> QUANTIFY -> BRIEF drives Gemini over the tool layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from continuity.agent.agents import (
    build_investigation_pipeline,
    extract_pipeline_result,
    verify_brief_citations,
)
from continuity.agent.tools import build_function_tools
from continuity.analysis.detect import detect
from continuity.analysis.slices import Slice
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

GROUND_TRUTH = Path("data/ground_truth.json")
APP_NAME = "continuity"
SEARCH_PADDING = timedelta(hours=6)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def load_incident(incident_id: str) -> dict:
    if not GROUND_TRUTH.exists():
        raise SystemExit(f"{GROUND_TRUTH} not found. Run the loader first.")
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    for incident in truth["incidents"]:
        if incident["incident_id"] == incident_id:
            return incident
    ids = [i["incident_id"] for i in truth["incidents"]]
    raise SystemExit(f"unknown incident {incident_id!r}. Known: {ids}")


class Usage:
    """Accumulates token counts across every model turn in a run."""

    def __init__(self) -> None:
        self.prompt = 0
        self.output = 0
        self.turns = 0

    def add(self, event) -> None:
        meta = getattr(event, "usage_metadata", None)
        if meta is None:
            return
        self.turns += 1
        self.prompt += getattr(meta, "prompt_token_count", 0) or 0
        self.output += getattr(meta, "candidates_token_count", 0) or 0

    def report(self) -> str:
        total = self.prompt + self.output
        return (
            f"model turns {self.turns} | prompt tokens {self.prompt:,} | "
            f"output tokens {self.output:,} | total {total:,}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident", default="INC-APP-ROKU-820")
    parser.add_argument("--metric", default=None)
    args = parser.parse_args()

    load_dotenv(override=False)
    incident = load_incident(args.incident)
    metric = args.metric or incident["effects"][0]["metric"]
    true_start, true_end = _parse(incident["start"]), _parse(incident["end"])
    search_start, search_end = true_start - SEARCH_PADDING, true_end + SEARCH_PADDING

    print(f"\nincident : {args.incident}  (ground truth {true_start} .. {true_end})")
    print(f"metric   : {metric}")
    print(f"predicate: {incident['predicate']}   <-- the agent must DISCOVER this\n")

    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gateway:
        # --- DETECT: deterministic, no model ---------------------------------------
        t0 = time.perf_counter()
        detection = await detect(gateway, Slice(), metric, search_start, search_end)
        detect_ms = (time.perf_counter() - t0) * 1000
        if not detection.windows:
            print(f"DETECT found no anomaly windows in {detect_ms:.0f}ms -- nothing to do.")
            return 1
        worst = max(detection.windows, key=lambda w: abs(w.peak_z))
        print(
            f"DETECT ({detect_ms:.0f}ms, no model): {len(detection.windows)} window(s); "
            f"worst {worst.start} .. {worst.end}, peak z {worst.peak_z:.1f}"
        )

        # --- AGENT: Gemini drives the rest ----------------------------------------
        tools = build_function_tools(gateway)
        pipeline, audit_log = build_investigation_pipeline(tools)
        runner = InMemoryRunner(agent=pipeline, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="cli")

        brief_window = (worst.start.isoformat(), worst.end.isoformat())
        prompt = (
            f"A population-level anomaly was detected on metric '{metric}'.\n"
            f"Anomaly window: {brief_window[0]} to {brief_window[1]}.\n"
            f"Peak robust z-score at population level: {worst.peak_z:.1f}.\n"
            f"Investigate where this is concentrated and produce the full brief."
        )

        usage = Usage()
        t1 = time.perf_counter()
        async for event in runner.run_async(
            user_id="cli",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            usage.add(event)
        agent_ms = (time.perf_counter() - t1) * 1000

        final_session = await runner.session_service.get_session(
            app_name=APP_NAME, user_id="cli", session_id=session.id
        )
        result = extract_pipeline_result(final_session.state)

    # --- report ------------------------------------------------------------------
    print(f"\nAGENT ({agent_ms / 1000:.1f}s): {usage.report()}")
    print(f"tool calls logged: {len(audit_log.entries)}")

    inv = result.investigation
    found = {d.dimension: d.value for d in inv.final_slice} if inv else {}
    expected = incident["predicate"]
    print(f"\nBLAST RADIUS   agent found : {found}")
    print(f"               ground truth: {expected}")
    print(f"               match       : {found == expected}")
    if inv:
        print(f"               stop reason : {inv.stop_reason}   lift {inv.final_lift}")

    if result.correlation:
        c = result.correlation
        print(f"\nCAUSE          {c.top_candidate_change_id}")
        print(f"               confidence  : {c.confidence}")
        print(f"               disconfirming: {c.disconfirming_evidence[:100]}")

    if result.quantify:
        q = result.quantify
        print(
            f"\nIMPACT         subscribers {q.affected_subscribers:,}  "
            f"ARR {q.arr_at_risk_expected}"
        )

    if result.brief:
        try:
            verify_brief_citations(result.brief, audit_log)
            print("\nCITATIONS      every brief claim resolves to a logged tool call")
        except Exception as exc:
            print(f"\nCITATIONS      FAILED: {exc}")

    print("\n--- tool calls the model chose to make ---")
    for i, entry in enumerate(audit_log.entries):
        print(f"  {i:>2}. {entry.tool_name}({json.dumps(entry.arguments)[:110]})")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
