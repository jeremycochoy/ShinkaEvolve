"""
Tests that the async runner keeps the meta summary across a resume.

Each launch runs the real `run_async()` on one run directory. A stub replaces
the meta LLM and the evaluation of a job. `launch()` completes the jobs of a
launch with a stub job monitor. `search()` keeps the real job monitor and the
real proposal coordinator, with a stub in place of the proposal LLM. Thus no
test starts a search or calls an LLM.
"""

import asyncio
import itertools
import json
import logging
import math
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from shinka.core import (
    EvolutionConfig,
    ShinkaEvolveRunner,
    async_runner,
    async_summarizer,
)
from shinka.core.async_runner import AsyncRunningJob
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.dbase import meta_cost_with_live_costs
from shinka.launch import LocalJobConfig
from shinka.pricing.catalog import PricingConfig, PricingMode, refresh_model_catalog
from shinka.prompts import (
    META_STEP1_SYSTEM_MSG,
    META_STEP2_SYSTEM_MSG,
    META_STEP3_SYSTEM_MSG,
)
from shinka.wandb_logging import PROGRAM_TABLE_KEY, ShinkaWandbLogger, is_island_copy

STATE_FILE = "meta_memory.json"
EVAL_RESULTS = {
    "correct": {"correct": True},
    "metrics": {"combined_score": 1.0, "public": {}, "private": {}},
}
# The evaluation of an incorrect program whose score is not a number.
NAN_RESULTS = {
    "correct": {"correct": False},
    "metrics": {"combined_score": float("nan"), "public": {}, "private": {}},
}
PROGRAM = """# EVOLVE-BLOCK-START
# GEN_{generation}
value = {generation}
# EVOLVE-BLOCK-END
"""
PROPOSAL = """<NAME>patch_{generation}</NAME>
<DESCRIPTION>The program of generation {generation}.</DESCRIPTION>
```python
{program}```
"""


@pytest.fixture(autouse=True)
def offline_runner(monkeypatch, tmp_path):
    """Build each runner with the packaged prices and no check of the models."""
    config = PricingConfig(mode=PricingMode.OFFLINE, cache_dir=tmp_path / "prices")
    monkeypatch.setattr(
        async_runner, "refresh_model_catalog", lambda: refresh_model_catalog(config)
    )
    monkeypatch.setattr(
        async_runner, "_validate_evo_config_model_env_access", lambda evo_config: None
    )
    yield
    refresh_model_catalog(config)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    """Try a failed cost write again with no delay."""
    monkeypatch.setattr(async_summarizer, "META_COST_WRITE_RETRY_DELAY", 0.0)


@dataclass
class StubResponse:
    content: str
    cost: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


class StubMetaLLM:
    """Answer in place of the meta LLM and record each meta update."""

    def __init__(
        self,
        name: str,
        failing_updates=(),
        missing_answers=(),
        blank_answers=(),
        error_answers=(),
        blank_generations=(),
        slow_updates=(),
    ):
        self.name = name
        # The numbers of the updates whose step 3 gives no recommendations.
        self.failing_updates = set(failing_updates)
        # The (update, generation) pairs whose step 1 query gives no answer, a
        # blank answer, and an error.
        self.missing_answers = set(missing_answers)
        self.blank_answers = set(blank_answers)
        self.error_answers = set(error_answers)
        # The generations whose step 1 answer is blank in each update.
        self.blank_generations = set(blank_generations)
        # The numbers of the updates whose step 1 takes 5 seconds.
        self.slow_updates = set(slow_updates)
        # The generations of the step 1 queries of each update.
        self.batches: list[list[int]] = []
        self.step2_messages: list[str] = []
        self.step3_messages: list[str] = []
        self.queries_in_progress = 0

    async def query(self, msg, system_msg):
        if system_msg == META_STEP1_SYSTEM_MSG:
            return await self.summarize(msg)
        update = len(self.batches)
        if system_msg == META_STEP2_SYSTEM_MSG:
            self.step2_messages.append(msg)
            return StubResponse(f"Insights of {self.name} update {update}.")
        self.step3_messages.append(msg)
        if update in self.failing_updates:
            return StubResponse("")
        return StubResponse(f"1. Recommendation of {self.name} update {update}.")

    async def summarize(self, msg):
        """Answer the step 1 query of one program.

        The step 1 queries of one update are in progress together. Thus a
        query that starts when no other is in progress starts a new update.
        """
        generation = int(re.search(r"GEN_(\d+)", msg).group(1))
        if self.queries_in_progress == 0:
            self.batches.append([])
        self.batches[-1].append(generation)
        update = len(self.batches)
        self.queries_in_progress += 1
        try:
            await asyncio.sleep(5 if update in self.slow_updates else 0)
        finally:
            self.queries_in_progress -= 1
        if (update, generation) in self.error_answers:
            raise RuntimeError(f"The query for GEN_{generation} failed.")
        if (update, generation) in self.missing_answers:
            # The real client gives None after its last try.
            return None
        if (update, generation) in self.blank_answers:
            return StubResponse("")
        if generation in self.blank_generations:
            return StubResponse("")
        return StubResponse(f"Summary of GEN_{generation}.")


class StubProposalLLM:
    """Answer in place of the proposal LLM: the program of each next generation."""

    def __init__(self, first_generation: int):
        self.generation = first_generation
        # The system message and the user message of each proposal.
        self.prompts: list[str] = []

    def get_kwargs(self, model_sample_probs=None):
        return {"model_name": "stub-model"}

    async def query(self, msg, system_msg, **kwargs):
        self.prompts.append(f"{system_msg}\n{msg}")
        program = PROGRAM.format(generation=self.generation)
        proposal = PROPOSAL.format(generation=self.generation, program=program)
        self.generation += 1
        return StubResponse(proposal)


def make_runner(
    run_dir: Path,
    meta_llm: StubMetaLLM | None,
    meta_rec_interval: int,
    num_generations: int = 50,
    nan_generations=(),
    max_db_workers: int = 2,
    max_api_costs: float | None = None,
    scores: dict[int, float] | None = None,
    db_options: dict | None = None,
) -> ShinkaEvolveRunner:
    """Build a runner on run_dir. Without a meta LLM, the meta summary is off.

    The evaluation of each generation of nan_generations gives a NaN score,
    and scores gives the score of a generation (1.0 by default).
    max_db_workers gives the count of workers of the write queue of the
    database (2 is the default of the runner). max_api_costs gives the cost
    budget of the search. db_options gives more fields of the database config.
    """
    init_path = run_dir.parent / "initial.py"
    init_path.write_text(PROGRAM.format(generation=0), encoding="utf-8")
    evo_config = EvolutionConfig(
        patch_types=["full"],
        patch_type_probs=[1.0],
        num_generations=num_generations,
        llm_models=["stub-model"],
        llm_dynamic_selection=None,
        meta_rec_interval=meta_rec_interval if meta_llm else None,
        meta_llm_models=["stub-meta-model"],
        embedding_model=None,
        init_program_path=str(init_path),
        results_dir=str(run_dir),
        max_api_costs=max_api_costs,
    )
    runner = ShinkaEvolveRunner(
        evo_config,
        LocalJobConfig(),
        DatabaseConfig(**{"num_islands": 2, **(db_options or {})}),
        verbose=False,
        max_evaluation_jobs=1,
        max_db_workers=max_db_workers,
    )
    score_of = scores or {}

    def results_of(generation: int) -> dict:
        if generation in nan_generations:
            return NAN_RESULTS
        if generation in score_of:
            metrics = {
                "combined_score": score_of[generation],
                "public": {},
                "private": {},
            }
            return {"correct": {"correct": True}, "metrics": metrics}
        return EVAL_RESULTS

    async def get_job_results(job_id, results_dir):
        # The results directory of a job is <run_dir>/gen_<generation>/results.
        return results_of(int(Path(results_dir).parent.name[4:]))

    runner.scheduler.run = lambda exec_fname, results_dir: (results_of(0), 0.0)
    runner.scheduler.get_job_results_async = get_job_results
    if meta_llm:
        runner.meta_summarizer.async_llm_client = meta_llm
    return runner


async def complete_job(runner: ShinkaEvolveRunner, generation: int) -> None:
    """Give the runner the completed evaluation job of one generation."""
    gen_dir = Path(runner.results_dir) / f"gen_{generation}"
    (gen_dir / "results").mkdir(parents=True)
    exec_fname = gen_dir / "main.py"
    exec_fname.write_text(
        f"# GEN_{generation}\nvalue = {generation}\n", encoding="utf-8"
    )
    job = AsyncRunningJob(
        job_id=f"job-{generation}",
        exec_fname=str(exec_fname),
        results_dir=str(gen_dir / "results"),
        start_time=time.time(),
        proposal_started_at=time.time(),
        evaluation_submitted_at=time.time(),
        generation=generation,
        meta_patch_data={"patch_name": f"patch_{generation}"},
    )
    assert await runner._process_single_job_safely(job)


def read_state(run_dir: Path) -> dict | None:
    """Read the meta state file of a run directory."""
    state_path = run_dir / STATE_FILE
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def read_memory(runner: ShinkaEvolveRunner) -> dict:
    """Give the state that the runner holds in memory, as a state file gives it."""
    memory_path = Path(runner.results_dir).parent / "memory.json"
    runner.meta_summarizer.sync_summarizer.save_meta_state(str(memory_path))
    return json.loads(memory_path.read_text(encoding="utf-8"))


async def read_state_after_the_cost_writes(runner: ShinkaEvolveRunner) -> dict | None:
    """Read the state file after the cost writes of the meta updates.

    A cost write runs in its own task, after the update. The state file
    gets an update after its cost write.
    """
    if runner.meta_summarizer:
        await runner.meta_summarizer.wait_for_cost_writes()
    return read_state(Path(runner.results_dir))


def waiting_generations(state: dict) -> list[int]:
    return [program["generation"] for program in state["unprocessed_programs"]]


def summary_labels(meta_summary: str) -> list[tuple[int, int]]:
    """Give the generation of the text and of the identifier of each summary."""
    labels = re.findall(
        r"Summary of GEN_(\d+)\.\n\*\*Program Identifier:\*\* Generation (\d+) ",
        meta_summary,
    )
    assert len(labels) == meta_summary.count("**Program Identifier:**"), meta_summary
    return [(int(text), int(identifier)) for text, identifier in labels]


def launch(
    run_dir: Path,
    meta_llm: StubMetaLLM | None,
    generations: list[int],
    meta_rec_interval: int = 3,
    nan_generations=(),
    max_db_workers: int = 2,
    scores: dict[int, float] | None = None,
    db_options: dict | None = None,
) -> SimpleNamespace:
    """Run one launch of a search on run_dir, from the setup to the final summary.

    The launch completes one job for each generation of the list, in their
    order (see make_runner() for the other arguments). The result
    holds what the first proposal of the launch sees (the recommendation, the
    API cost of the search, the count of the meta updates of the launch, the
    state file after the cost writes, and the seconds from the last progress
    that the runner recorded). It also holds the state file after the cost
    writes and the state in memory after each job, and the API cost of the
    search at the end of the launch.
    """
    runner = make_runner(
        run_dir,
        meta_llm,
        meta_rec_interval,
        nan_generations=nan_generations,
        max_db_workers=max_db_workers,
        scores=scores,
        db_options=db_options,
    )
    seen = SimpleNamespace(
        first_recommendation=None,
        api_cost=None,
        updates=None,
        start_state=None,
        progress_age=None,
        states=[],
        memory=[],
        final_api_cost=None,
    )

    async def complete_jobs():
        try:
            seen.api_cost = runner.total_api_cost
            seen.updates = len(meta_llm.batches) if meta_llm else 0
            seen.progress_age = time.time() - runner.last_progress_time
            if runner.meta_summarizer:
                seen.first_recommendation = (
                    runner.meta_summarizer.get_sampled_recommendation()
                )
            seen.start_state = await read_state_after_the_cost_writes(runner)
            for generation in generations:
                await complete_job(runner, generation)
                if runner.meta_summarizer:
                    seen.memory.append(read_memory(runner))
                seen.states.append(await read_state_after_the_cost_writes(runner))
        finally:
            runner.should_stop.set()
            runner.finalization_complete.set()

    async def no_proposals():
        return None

    runner._job_monitor_task = complete_jobs
    runner._proposal_coordinator_task = no_proposals
    asyncio.run(runner.run_async())
    seen.final_api_cost = runner.total_api_cost
    assert len(seen.states) == len(generations), "a job of the launch failed"
    return seen


def search(
    run_dir: Path,
    meta_llm: StubMetaLLM,
    meta_rec_interval: int,
    num_generations: int,
    stop_at: tuple[str, int] | None = None,
    job_ids: dict[int, str] | None = None,
    totals: list[float] | None = None,
    nan_generations=(),
    max_api_costs: float | None = None,
) -> list[str]:
    """Run one launch with the real job monitor and the real proposal coordinator.

    stop_at gives a method of the runner or of its meta summarizer, and a call
    number. The launch then ends at that call as a stop of the process does:
    the call does not run, and no later step of the launch runs.

    job_ids gives the identifier of the job of a generation. By default, each
    job has its own identifier.

    totals gets the API cost of the search at each proposal, then at the end
    of the launch.

    The evaluation of each generation of nan_generations gives a NaN score.
    max_api_costs gives the cost budget of the search.

    The result is the prompt of each proposal of the launch.
    """
    runner = make_runner(
        run_dir,
        meta_llm,
        meta_rec_interval,
        num_generations,
        nan_generations=nan_generations,
        max_api_costs=max_api_costs,
    )
    last_generation = max(
        (int(path.name[4:]) for path in run_dir.glob("gen_*")), default=0
    )
    runner.llm = StubProposalLLM(first_generation=last_generation + 1)
    if totals is not None:
        query = runner.llm.query

        async def record_then_query(msg, system_msg, **kwargs):
            totals.append(runner.total_api_cost)
            return await query(msg, system_msg, **kwargs)

        runner.llm.query = record_then_query

    async def submit_job(exec_fname, results_dir):
        generation_dir = Path(exec_fname).parent.name
        return (job_ids or {}).get(int(generation_dir[4:]), f"job-{generation_dir}")

    async def completed(jobs):
        return [False] * len(jobs)

    runner.scheduler.submit_async_nonblocking = submit_job
    runner.scheduler.batch_check_status_async = completed

    async def run_search():
        if stop_at:
            method_name, stop_call = stop_at
            is_runner_method = hasattr(ShinkaEvolveRunner, method_name)
            owner = runner if is_runner_method else runner.meta_summarizer
            method = getattr(owner, method_name)
            launch_task = asyncio.current_task()
            calls = []

            def stop_or_call(*args):
                calls.append(args)
                if len(calls) == stop_call:
                    launch_task.cancel()
                    raise asyncio.CancelledError
                return method(*args)

            setattr(owner, method_name, stop_or_call)
        await asyncio.wait_for(runner.run_async(), timeout=60)

    try:
        if stop_at:
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(run_search())
        else:
            asyncio.run(run_search())
    finally:
        if totals is not None:
            totals.append(runner.total_api_cost)
    return runner.llm.prompts


def evaluated_program_ids(run_dir: Path) -> dict[str, int]:
    """Give the generation of each evaluated program of the database, by id."""
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        return {p.id: p.generation for p in db.get_evaluated_programs()}
    finally:
        db.close()


def meta_files(run_dir: Path) -> list[str]:
    return sorted(path.name for path in (run_dir / "meta").iterdir())


def test_fresh_search_writes_the_meta_state(tmp_path):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1")

    seen = launch(run_dir, meta_llm, generations=[1, 2, 3])

    # The search itself does not change.
    assert seen.first_recommendation is None
    assert meta_llm.batches == [[0, 1, 2], [3]]
    assert meta_files(run_dir) == ["meta_3.txt", "meta_4.txt"]

    # Generation 1: 2 programs wait for a summary, and there is no state file.
    assert seen.start_state is None
    assert waiting_generations(seen.memory[0]) == [0, 1]
    assert seen.memory[0]["total_programs_meta_processed"] == 0
    assert seen.states[0] is None

    # Generation 2: the first meta update gives recommendations and the state file.
    assert waiting_generations(seen.states[1]) == []
    assert seen.states[1]["total_programs_meta_processed"] == 3
    assert seen.states[1]["meta_summary"].count("Summary of GEN_") == 3
    assert seen.states[1]["meta_scratch_pad"] == "Insights of launch 1 update 1."
    assert (
        seen.states[1]["meta_recommendations"]
        == "1. Recommendation of launch 1 update 1."
    )

    # Generation 3 waits, then the final summary takes it.
    assert waiting_generations(seen.memory[2]) == [3]
    final_state = read_state(run_dir)
    assert waiting_generations(final_state) == []
    assert final_state["total_programs_meta_processed"] == 4
    assert (
        final_state["meta_recommendations"] == "1. Recommendation of launch 1 update 2."
    )


def test_update_orders_the_summaries_by_generation(tmp_path):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1")

    # The job of generation 2 ends before the job of generation 1.
    seen = launch(run_dir, meta_llm, generations=[2, 1])

    assert meta_llm.batches == [[0, 2, 1]]
    summary = seen.states[1]["meta_summary"]
    assert summary_labels(summary) == [(0, 0), (1, 1), (2, 2)]


def test_update_of_the_initial_program_writes_the_meta_state(tmp_path):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1")

    seen = launch(run_dir, meta_llm, generations=[], meta_rec_interval=1)

    assert meta_llm.batches == [[0]]
    assert waiting_generations(seen.start_state) == []
    assert seen.start_state["total_programs_meta_processed"] == 1
    assert (
        seen.start_state["meta_recommendations"]
        == "1. Recommendation of launch 1 update 1."
    )
    assert seen.api_cost == pytest.approx(0.03)

    # The next launch finds generation 0 only. Its cost total holds the update
    # of launch 1, and the update of its own program of generation 0.
    next_launch = launch(run_dir, StubMetaLLM("launch 2"), [], meta_rec_interval=1)
    assert next_launch.api_cost == pytest.approx(0.06)


def test_launch_on_generation_0_only_adds_the_meta_costs_only(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"

    async def paid_embedding(self, exec_fname):
        return None, 0.5

    # The embedding of each program costs 0.5. The island copy of the first
    # program of generation 0 holds a copy of that cost.
    monkeypatch.setattr(ShinkaEvolveRunner, "_get_code_embedding_async", paid_embedding)
    first = launch(run_dir, StubMetaLLM("launch 1"), [], meta_rec_interval=1)
    second = launch(run_dir, StubMetaLLM("launch 2"), [], meta_rec_interval=1)
    third = launch(run_dir, None, generations=[])

    # Launch 1 pays 1 embedding and the update of 1 program.
    assert first.api_cost == pytest.approx(0.5 + 0.03)
    # Launch 2 is not a resume, thus its other costs start at 0: it pays 1
    # embedding. Its state holds 2 updates, thus its total holds their cost.
    assert second.api_cost == pytest.approx(0.5 + 0.03 + 0.03)
    # With no meta summary, such a launch has no state: its total is its own cost.
    assert third.api_cost == pytest.approx(0.5)


def test_launch_on_generation_0_only_continues_the_meta_state(tmp_path, caplog):
    run_dir = tmp_path / "run"
    # Launch 1 has generation 0 only: it ends after the update of that
    # generation, with no proposal.
    search(run_dir, StubMetaLLM("launch 1"), meta_rec_interval=1, num_generations=1)
    assert list(evaluated_program_ids(run_dir).values()) == [0]
    first_output = (run_dir / "meta" / "meta_1.txt").read_text(encoding="utf-8")

    meta_llm = StubMetaLLM("launch 2")
    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        prompts = search(run_dir, meta_llm, meta_rec_interval=1, num_generations=2)

    # A database with generation 0 only is not a resume: the runner evaluates
    # the initial program again. That program is a new row, thus it gets its
    # own summary.
    programs = evaluated_program_ids(run_dir)
    assert sorted(programs.values()) == [0, 0, 1]
    assert meta_llm.batches == [[0], [1]]
    # The launch loads the state of launch 1, and a log line says what it loaded.
    assert (
        "Meta state of the launch: state file loaded, 1 processed programs, "
        "0 waiting programs (0 from the database), recommendations: Yes"
    ) in caplog.messages
    # The update of the new program continues the insights and the
    # recommendations of launch 1. It comes before the first proposal, thus
    # that proposal gets the recommendation of this update.
    assert "Insights of launch 1 update 1." in meta_llm.step2_messages[0]
    assert "1. Recommendation of launch 1 update 1." in meta_llm.step3_messages[0]
    assert "Recommendation of launch 2 update 1." in prompts[0]
    # The count of the processed programs continues, and the output file of
    # launch 1 stays.
    assert meta_files(run_dir) == ["meta_1.txt", "meta_2.txt", "meta_3.txt"]
    assert (run_dir / "meta" / "meta_1.txt").read_text(encoding="utf-8") == first_output
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(programs)
    assert summary_labels(state["meta_summary"]) == [(0, 0), (0, 0), (1, 1)]


def test_first_proposal_after_generation_0_only_gets_the_saved_recommendation(tmp_path):
    run_dir = tmp_path / "run"
    search(run_dir, StubMetaLLM("launch 1"), meta_rec_interval=1, num_generations=1)

    # With an interval of 2, the new program of generation 0 waits for the
    # program of generation 1: no update comes before the first proposal.
    meta_llm = StubMetaLLM("launch 2")
    prompts = search(run_dir, meta_llm, meta_rec_interval=2, num_generations=2)

    # The first proposal prompt of launch 2 holds the recommendation of launch 1.
    assert "Recommendation of launch 1 update 1." in prompts[0]
    # The count continues: 1 program of launch 1, then 2 programs of launch 2.
    assert meta_llm.batches == [[0, 1]]
    assert meta_files(run_dir) == ["meta_1.txt", "meta_3.txt"]
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )


def test_fresh_search_does_not_load_a_state_file_with_no_database(tmp_path):
    run_dir = tmp_path / "run"
    # A state file of another search, in a results directory with no database.
    other_dir = tmp_path / "other" / "run"
    other_dir.parent.mkdir()
    launch(other_dir, StubMetaLLM("other search"), generations=[1, 2])
    run_dir.mkdir()
    (run_dir / STATE_FILE).write_bytes((other_dir / STATE_FILE).read_bytes())

    meta_llm = StubMetaLLM("launch 1")
    seen = launch(run_dir, meta_llm, generations=[1, 2])

    # The meta state follows the database: the search starts with no state.
    assert seen.first_recommendation is None
    assert meta_llm.batches == [[0, 1, 2]]
    assert "other search" not in meta_llm.step3_messages[0]
    assert meta_files(run_dir) == ["meta_3.txt"]


def test_log_line_says_what_the_launch_loaded(tmp_path, caplog):
    run_dir = tmp_path / "run"
    caplog.set_level(logging.INFO, logger="shinka.core.async_runner")
    # A launch with no meta summary gives 2 programs and no state file.
    launch(run_dir, None, generations=[1])
    # The final summary of launch 2 gives no recommendations: generation 3
    # waits in the state file.
    launch(run_dir, StubMetaLLM("launch 2", failing_updates={2}), generations=[2, 3])
    # A launch with no meta summary gives generation 4 to the database only.
    launch(run_dir, None, generations=[4])
    launch(run_dir, StubMetaLLM("launch 4"), generations=[])

    start = "Meta state of the launch: "
    assert [line for line in caplog.messages if line.startswith(start)] == [
        start + "state file not loaded, 0 processed programs, "
        "2 waiting programs (2 from the database), recommendations: No",
        start + "state file loaded, 3 processed programs, "
        "2 waiting programs (1 from the database), recommendations: Yes",
    ]


def test_resume_continues_the_meta_state(tmp_path):
    run_dir = tmp_path / "run"
    # The final summary of launch 1 gives no recommendations: generation 3 waits.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2}), [1, 2, 3])
    first_output = (run_dir / "meta" / "meta_3.txt").read_text(encoding="utf-8")

    meta_llm = StubMetaLLM("launch 2")
    seen = launch(run_dir, meta_llm, generations=[4, 5])

    # The first proposal of the resume gets a recommendation.
    assert seen.first_recommendation == "Recommendation of launch 1 update 1."
    # The program that waits continues: generations 3, 4 and 5 make the update.
    assert meta_llm.batches == [[3, 4, 5]]
    # The scratchpad and the recommendations continue.
    assert "Insights of launch 1 update 1." in meta_llm.step2_messages[0]
    assert "1. Recommendation of launch 1 update 1." in meta_llm.step3_messages[0]
    # The count continues: the next output file has the next number.
    assert meta_files(run_dir) == ["meta_3.txt", "meta_6.txt"]
    assert (run_dir / "meta" / "meta_3.txt").read_text(encoding="utf-8") == first_output
    # The summary continues: it holds each generation of the 2 launches.
    second_output = (run_dir / "meta" / "meta_6.txt").read_text(encoding="utf-8")
    for generation in range(6):
        assert f"Summary of GEN_{generation}." in second_output
    assert read_state(run_dir)["total_programs_meta_processed"] == 6


def test_resume_makes_the_due_update_before_the_first_proposal(tmp_path):
    run_dir = tmp_path / "run"
    # Update 2 and the final summary of launch 1 give no recommendations.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1, 2, 3, 4, 5])
    assert waiting_generations(read_state(run_dir)) == [3, 4, 5]

    meta_llm = StubMetaLLM("launch 2")
    seen = launch(run_dir, meta_llm, generations=[])

    # The waiting list holds a full interval, thus the resume starts with an update.
    assert meta_llm.batches == [[3, 4, 5]]
    assert seen.first_recommendation == "Recommendation of launch 2 update 1."
    assert meta_files(run_dir) == ["meta_3.txt", "meta_6.txt"]


def test_update_without_recommendations_keeps_the_waiting_programs(tmp_path):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1", failing_updates={2})

    seen = launch(run_dir, meta_llm, generations=[1, 2, 3, 4, 5])

    # Generation 5 starts update 2, and its step 3 gives no recommendations.
    assert meta_llm.batches[:2] == [[0, 1, 2], [3, 4, 5]]
    state = seen.memory[4]
    assert waiting_generations(state) == [3, 4, 5]
    assert state["total_programs_meta_processed"] == 3
    assert state["meta_summary"].count("Summary of GEN_") == 3
    assert state["meta_scratch_pad"] == "Insights of launch 1 update 1."
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 1."
    # The state file stays the file of update 1: it holds no result of update 2.
    assert seen.states[4] == seen.states[1]
    assert seen.states[4]["total_programs_meta_processed"] == 3
    # The final summary then takes the same 3 programs.
    assert meta_llm.batches[2] == [3, 4, 5]
    assert meta_files(run_dir) == ["meta_3.txt", "meta_6.txt"]


def test_resume_without_state_file_summarizes_the_full_history(tmp_path):
    run_dir = tmp_path / "run"
    # A search before the state file: 6 programs, and jobs that end out of order.
    launch(run_dir, None, generations=[2, 1, 4, 3, 5])
    assert read_state(run_dir) is None

    meta_llm = StubMetaLLM("resume")
    seen = launch(run_dir, meta_llm, generations=[])

    # One update before the first proposal, on each program one time, in the
    # order of the generations. The island copy of generation 0 is not there.
    assert meta_llm.batches == [[0, 1, 2, 3, 4, 5]]
    assert seen.first_recommendation == "Recommendation of resume update 1."
    assert meta_files(run_dir) == ["meta_6.txt"]
    state = seen.start_state
    assert waiting_generations(state) == []
    assert state["total_programs_meta_processed"] == 6
    # A program from the database gets the same label as in a fresh search.
    assert state["meta_summary"].count("Correct Program: True") == 6
    # The cost of the update (3 LLM steps on 6 programs) is in the total of
    # this launch, and the next launch reads the same total from the database.
    assert seen.api_cost == pytest.approx(0.08)
    assert launch(run_dir, StubMetaLLM("launch 3"), []).api_cost == pytest.approx(0.08)


def test_resume_without_state_file_below_the_interval_makes_no_update(tmp_path):
    run_dir = tmp_path / "run"
    launch(run_dir, None, generations=[1])

    meta_llm = StubMetaLLM("resume")
    seen = launch(run_dir, meta_llm, generations=[2])

    # Generations 0 and 1 wait at the start. Generation 2 completes the interval.
    assert seen.first_recommendation is None
    assert meta_llm.batches == [[0, 1, 2]]
    assert meta_files(run_dir) == ["meta_3.txt"]


def test_search_without_meta_summary_writes_no_state_file(tmp_path):
    run_dir = tmp_path / "run"

    launch(run_dir, None, generations=[1, 2, 3])

    assert read_state(run_dir) is None


def test_resume_adds_the_programs_that_the_state_file_does_not_hold(tmp_path):
    run_dir = tmp_path / "run"
    # The update after generation 1 writes the state file. Generation 2 goes
    # into the waiting list. The launch stops after the write of generation 3
    # to the database, before the program goes into the waiting list: the
    # fourth call.
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=2,
        num_generations=5,
        stop_at=("add_evaluated_program", 4),
    )
    assert sorted(evaluated_program_ids(run_dir).values()) == [0, 1, 2, 3]
    state = read_state(run_dir)
    assert state["total_programs_meta_processed"] == 2
    assert waiting_generations(state) == []

    meta_llm = StubMetaLLM("launch 2")
    search(run_dir, meta_llm, meta_rec_interval=2, num_generations=5)

    # The resume adds generations 2 and 3 from the database, thus the update
    # that is due comes before the proposal of generation 4. The final summary
    # takes generation 4.
    assert meta_llm.batches == [[2, 3], [4]]
    assert meta_files(run_dir) == ["meta_2.txt", "meta_4.txt", "meta_5.txt"]
    # The state holds each program of the database one time, by its id.
    state = read_state(run_dir)
    programs = evaluated_program_ids(run_dir)
    assert sorted(programs.values()) == [0, 1, 2, 3, 4]
    assert sorted(state["processed_program_ids"]) == sorted(programs)
    assert state["unprocessed_programs"] == []
    assert summary_labels(state["meta_summary"]) == [
        (generation, generation) for generation in range(5)
    ]


def test_stop_of_a_resume_before_the_load_keeps_the_state_file(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    launch(run_dir, StubMetaLLM("launch 1"), generations=[1, 2])
    state = (run_dir / STATE_FILE).read_text(encoding="utf-8")

    # A stop of the process (one Ctrl-C) comes in the setup of a resume,
    # before the load of the meta state.
    async def stop(self):
        raise asyncio.CancelledError

    monkeypatch.setattr(ShinkaEvolveRunner, "_get_total_api_costs", stop)
    with pytest.raises(asyncio.CancelledError):
        launch(run_dir, StubMetaLLM("launch 2"), generations=[])

    # The end of the launch does not write its empty state over the state file.
    assert (run_dir / STATE_FILE).read_text(encoding="utf-8") == state


def test_stop_before_the_first_state_file_keeps_each_program(tmp_path):
    run_dir = tmp_path / "run"
    # The launch stops after the write of generation 2 to the database, before
    # the first update: the third call, after generations 0 and 1.
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=3,
        num_generations=4,
        stop_at=("add_evaluated_program", 3),
    )
    assert sorted(evaluated_program_ids(run_dir).values()) == [0, 1, 2]
    assert read_state(run_dir) is None

    meta_llm = StubMetaLLM("launch 2")
    search(run_dir, meta_llm, meta_rec_interval=3, num_generations=4)

    # The resume adds the 3 generations, thus the update that is due comes
    # before the proposal of generation 3. The final summary takes generation 3.
    assert meta_llm.batches == [[0, 1, 2], [3]]
    assert meta_files(run_dir) == ["meta_3.txt", "meta_4.txt"]
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_stop_before_the_output_file_does_not_repeat_the_summary(tmp_path):
    run_dir = tmp_path / "run"
    # Generation 1 completes the interval. The launch stops at the first write
    # of an output file: the 3 steps of the update of generations 0 and 1 are
    # complete.
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=2,
        num_generations=3,
        stop_at=("write_meta_output_async", 1),
    )
    assert not (run_dir / "meta").exists()

    meta_llm = StubMetaLLM("launch 2")
    search(run_dir, meta_llm, meta_rec_interval=2, num_generations=3)

    # The resume does not summarize generations 0 and 1 again. The final
    # summary takes generation 2.
    assert meta_llm.batches == [[2]]
    assert "Insights of launch 1 update 1." in meta_llm.step2_messages[0]
    # The resume writes the output file of the update of launch 1 from the state.
    assert meta_files(run_dir) == ["meta_2.txt", "meta_3.txt"]
    first_output = (run_dir / "meta" / "meta_2.txt").read_text(encoding="utf-8")
    assert summary_labels(first_output) == [(0, 0), (1, 1)]
    assert "1. Recommendation of launch 1 update 1." in first_output
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2)]


def test_new_job_with_the_identifier_of_a_stored_job_gets_its_own_summary(tmp_path):
    run_dir = tmp_path / "run"
    # The identifier of a local job holds the PID of its process, and the
    # system can give that PID to a new process. Here the job of generation 2
    # gets the identifier of the job of generation 1.
    job_ids = {1: "ProcessWithLogging(PID: 4242)", 2: "ProcessWithLogging(PID: 4242)"}
    # Launch 1 stops after the state save of the update of generation 1, thus
    # before the end of the side effects of the job of generation 1.
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=2,
        num_generations=2,
        stop_at=("write_meta_output_async", 1),
        job_ids=job_ids,
    )
    assert summary_labels(read_state(run_dir)["meta_summary"]) == [(0, 0), (1, 1)]

    meta_llm = StubMetaLLM("launch 2")
    search(run_dir, meta_llm, meta_rec_interval=2, num_generations=3, job_ids=job_ids)

    # The job of generation 2 is a new job: its program gets its own row and
    # its own summary. The program of generation 1 keeps its one summary.
    programs = evaluated_program_ids(run_dir)
    assert sorted(programs.values()) == [0, 1, 2]
    assert meta_llm.batches == [[2]]
    state = read_state(run_dir)
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2)]
    assert sorted(state["processed_program_ids"]) == sorted(programs)


# The interval and the count of generations of a search whose first meta
# update is on each path of the runner. For the resume path, a launch with no
# meta summary first gives the 2 programs of the update.
UPDATE_PATHS = {
    "initial program": (1, 2),
    "completed job": (2, 2),
    "resume": (2, 2),
    "final summary": (5, 2),
}


def update_costs(batches: list[list[int]]) -> float:
    """Give the cost of meta updates: 0.01 for each answer of the 3 steps."""
    return sum(0.01 * (len(batch) + 2) for batch in batches)


def stored_meta_costs(run_dir: Path) -> float:
    """Give the sum of the meta costs of the evaluated programs of the database."""
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        return sum(
            p.metadata.get("meta_cost") or 0.0 for p in db.get_evaluated_programs()
        )
    finally:
        db.close()


def stored_update_ids(run_dir: Path) -> list[str]:
    """Give the id of each meta update whose cost the database holds."""
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        return [
            update_id
            for p in db.get_evaluated_programs()
            for update_id in p.metadata.get("meta_cost_update_ids") or []
        ]
    finally:
        db.close()


def output_texts(run_dir: Path) -> dict[str, str]:
    """Give the text of each output file of the meta updates, by name."""
    meta_dir = run_dir / "meta"
    if not meta_dir.exists():
        return {}
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(meta_dir.glob("*.txt"))
    }


def prompt_marks(prompts: list[str]) -> list[list[str]]:
    """Give the meta recommendations and insights in each proposal prompt."""
    return [
        re.findall(r"(?:Recommendation|Insights) of launch \d+ update \d+", prompt)
        for prompt in prompts
    ]


def slow_cost_write(db, program_id, meta_cost, update_id):
    """Take the place of the cost write of a meta update when it is slow.

    Each meta update writes its cost with `ProgramDatabase.add_meta_cost()`
    in a worker thread, and no other write of a launch uses that method. Here
    the write takes 0.5 seconds, then it fails.
    """
    time.sleep(0.5)
    raise RuntimeError("The write of the cost failed.")


def locked_cost_write(db, program_id, meta_cost, update_id):
    """Take the place of the same write when the database is locked."""
    raise sqlite3.OperationalError("database is locked")


def fail_cost_writes(patch) -> None:
    """Make each write of a meta cost to the database fail."""
    patch.setattr(ProgramDatabase, "add_meta_cost", locked_cost_write)


def fail_the_first_cost_writes(patch, count: int) -> list:
    """Make the first count writes of a meta cost fail, and the later succeed.

    Returns the list of the arguments of each write.
    """
    add_meta_cost = ProgramDatabase.add_meta_cost
    calls = []

    def fail_then_write(db, *args):
        calls.append(args)
        if len(calls) <= count:
            raise sqlite3.OperationalError("database is locked")
        return add_meta_cost(db, *args)

    patch.setattr(ProgramDatabase, "add_meta_cost", fail_then_write)
    return calls


def commit_each_cost_write_late(patch, seconds: float = 0.5) -> None:
    """Make each write of a meta cost commit after some seconds."""
    add_meta_cost = ProgramDatabase.add_meta_cost

    def write_late(db, *args):
        time.sleep(seconds)
        return add_meta_cost(db, *args)

    patch.setattr(ProgramDatabase, "add_meta_cost", write_late)


async def wait_before_the_write_queue(event: threading.Event) -> bool:
    """Wait for an event (30 seconds at most) with no worker thread of the database.

    The cost writes of 2 updates can wait at the same time. A wait in a
    worker thread holds that worker, and the search needs a worker for its
    other writes. A real database has no such wait: when it hangs, each write
    waits, and when it continues, each write continues.
    """
    deadline = time.monotonic() + 30
    while not event.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return event.is_set()


def hold_cost_writes_until_the_final_summary(patch) -> None:
    """Make each write of a meta cost wait until the final summary starts.

    Thus each cost write of the search waits while the later proposals and
    updates occur, as with a slow database.
    """
    final_summary_started = threading.Event()
    add_meta_cost_async = AsyncProgramDatabase.add_meta_cost_async

    async def wait_then_write(self, *args):
        assert await wait_before_the_write_queue(final_summary_started)
        return await add_meta_cost_async(self, *args)

    summarize = AsyncMetaSummarizer.perform_final_summary_async

    async def start_then_summarize(self, *args, **kwargs):
        final_summary_started.set()
        return await summarize(self, *args, **kwargs)

    patch.setattr(AsyncProgramDatabase, "add_meta_cost_async", wait_then_write)
    patch.setattr(
        AsyncMetaSummarizer, "perform_final_summary_async", start_then_summarize
    )


class FakeWandbRun:
    """Take the place of a W&B run, and record what the runner logs to it."""

    def __init__(self):
        self.payloads: list[dict] = []
        self.summary: dict = {}
        self.logged = threading.Condition()

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, payload: dict) -> None:
        with self.logged:
            self.payloads.append(payload)
            self.logged.notify_all()

    def finish(self):
        pass

    def program_payloads(self) -> list[dict]:
        """Give the payload of each program that the runner logged."""
        return [payload for payload in self.payloads if "individual/id" in payload]

    def wait_for_the_log_of(self, program_id: str) -> None:
        """Wait in a worker thread until the runner logs a program."""
        with self.logged:
            assert self.logged.wait_for(
                lambda: any(
                    payload["individual/id"] == program_id
                    for payload in self.program_payloads()
                ),
                timeout=30,
            )


def record_wandb(patch) -> FakeWandbRun:
    """Turn on the W&B logging of each runner, with a fake wandb package."""
    run = FakeWandbRun()
    wandb: Any = ModuleType("wandb")
    wandb.init = lambda **kwargs: run
    wandb.Table = SimpleNamespace
    wandb.Settings = SimpleNamespace
    patch.setitem(sys.modules, "wandb", wandb)
    patch.setattr(
        async_runner, "ShinkaWandbLogger", lambda enabled: ShinkaWandbLogger(True)
    )
    return run


def wandb_costs(run: FakeWandbRun) -> dict[int, dict[str, float]]:
    """Give the costs of the W&B payload of each program, by generation."""
    return {
        payload["generation"]: {
            key: round(value, 6)
            for key, value in payload.items()
            if key.startswith("individual/cost/")
        }
        for payload in run.program_payloads()
    }


def retry_the_first_cost_write_after_the_wandb_log(patch, run, run_dir) -> list:
    """Make the first cost write fail, and its retry wait for the W&B log.

    Thus the side effects of the program of the update end before the retry:
    the metadata write, then the W&B log. A retry delay of 1 second gives the
    same order. Returns what W&B, the database and the state file hold at the
    retry, before its write.
    """
    add_meta_cost = ProgramDatabase.add_meta_cost
    calls = itertools.count(1)
    at_the_retry: list[SimpleNamespace] = []

    def fail_then_write_after_the_log(db, program_id, *args):
        if next(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        if not at_the_retry:
            run.wait_for_the_log_of(program_id)
            at_the_retry.append(
                SimpleNamespace(
                    wandb=wandb_costs(run),
                    stored=stored_meta_costs(run_dir),
                    state=read_state(run_dir),
                )
            )
        return add_meta_cost(db, program_id, *args)

    patch.setattr(ProgramDatabase, "add_meta_cost", fail_then_write_after_the_log)
    return at_the_retry


# The failure points of the cost write of a meta update.
COST_WRITE_FAILURES = {
    "fails 1 time": partial(fail_the_first_cost_writes, count=1),
    "fails 3 times": lambda patch: fail_the_first_cost_writes(
        patch, async_summarizer.META_COST_WRITE_TRIES
    ),
    "fails each time": fail_cost_writes,
    "is slow": commit_each_cost_write_late,
    "waits for the final summary": hold_cost_writes_until_the_final_summary,
}
# The interval of a fresh search whose first meta update is on each path.
FRESH_PATHS = {"initial program": 1, "completed job": 2, "final summary": 5}
# The result of a fresh search with no failure point, by interval and budget.
REFERENCE_SEARCHES: dict = {}


def fresh_search_result(run_dir, monkeypatch, interval, max_api_costs, failure=None):
    """Run a fresh search of 3 generations, with the failure point of a cost write.

    Return what the search gives: the programs of each meta update, the meta
    marks of each proposal prompt, the output files, the cost total at the
    end of the search, and the costs of the W&B payload of each program.
    """
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    with monkeypatch.context() as patch:
        run = record_wandb(patch)
        if failure:
            COST_WRITE_FAILURES[failure](patch)
        prompts = search(
            run_dir, meta_llm, interval, 3, totals=totals, max_api_costs=max_api_costs
        )
    return {
        "updates": meta_llm.batches,
        "prompts": prompt_marks(prompts),
        "output files": output_texts(run_dir),
        "cost total": round(totals[-1], 6),
        "W&B costs": wandb_costs(run),
    }


@pytest.mark.parametrize("max_api_costs", [None, 0.045])
@pytest.mark.parametrize("path", FRESH_PATHS)
@pytest.mark.parametrize("failure", COST_WRITE_FAILURES)
def test_cost_write_does_not_change_a_fresh_search(
    tmp_path, monkeypatch, failure, path, max_api_costs
):
    interval = FRESH_PATHS[path]
    key = (interval, max_api_costs)
    if key not in REFERENCE_SEARCHES:
        REFERENCE_SEARCHES[key] = fresh_search_result(
            tmp_path / "reference", monkeypatch, interval, max_api_costs
        )

    result = fresh_search_result(
        tmp_path / "run", monkeypatch, interval, max_api_costs, failure
    )

    # As on main, the state in memory takes an update after its step 3, then
    # the output file comes, then the cost total counts the update. The
    # program of the update gets its cost before its W&B log. The cost write
    # comes after, in its own task. Thus a cost write that fails, or that
    # waits, changes no proposal prompt, no output file, no cost total, no W&B
    # payload of a program, and no stop of the cost budget (0.045 stops the
    # search after its first update at the intervals 1 and 2).
    assert result == REFERENCE_SEARCHES[key]
    assert result["updates"]


# The paths whose meta update writes its cost to the program of the update,
# then logs that program to W&B. Each path gives the interval, the generations
# of the jobs, and the generation of the program of the update.
WANDB_PATHS = {"initial program": (1, [], 0), "completed job": (2, [1], 1)}


@pytest.mark.parametrize("path", WANDB_PATHS)
def test_wandb_gets_the_cost_of_an_update_whose_cost_write_tries_again(
    tmp_path, monkeypatch, path
):
    run_dir = tmp_path / "run"
    interval, generations, update_generation = WANDB_PATHS[path]
    meta_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        run = record_wandb(patch)
        at_the_retry = retry_the_first_cost_write_after_the_wandb_log(
            patch, run, run_dir
        )
        result = launch(run_dir, meta_llm, generations, meta_rec_interval=interval)

    assert len(meta_llm.batches) == 1
    paid = round(update_costs(meta_llm.batches), 6)
    expected = {generation: 0.0 for generation in [0, *generations]}
    expected[update_generation] = paid
    # Before the retry: as on main, W&B has the cost of the update with its
    # program. The metadata write of that program keeps the meta cost of the
    # database. Thus the database holds no cost, and no save holds the update.
    (before,) = at_the_retry
    assert {
        g: costs["individual/cost/meta"] for g, costs in before.wandb.items()
    } == expected
    assert before.stored == 0
    assert before.state is None or before.state["meta_recommendations_history"] == []
    # After the retry: W&B logged each program one time, with the same cost.
    # The database holds the cost one time, and the state file holds the
    # update.
    assert {
        g: costs["individual/cost/meta"] for g, costs in wandb_costs(run).items()
    } == expected
    assert len(run.program_payloads()) == len(expected)
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    assert len(stored_update_ids(run_dir)) == 1
    state = read_state(run_dir)
    assert state["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 1."
    ]
    assert waiting_generations(state) == []
    # The final table merges the cost of the update with the costs of the
    # database by update id, and the summary of W&B reads the cost total: each
    # holds the cost one time.
    table = run.payloads[-1][PROGRAM_TABLE_KEY]
    assert sum(row[-1] for row in table.data) == pytest.approx(paid)
    assert run.summary["run/total_api_cost"] == pytest.approx(paid)
    assert result.final_api_cost == pytest.approx(paid)


@pytest.mark.parametrize("db_path", ["programs.sqlite", None], ids=["file", "memory"])
def test_final_wandb_snapshot_merges_the_live_meta_costs(tmp_path, monkeypatch, db_path):
    run = record_wandb(monkeypatch)
    runner = make_runner(tmp_path / "run", None, 1)
    runner.wandb_logger.start(
        evo_config=runner.evo_config,
        db_config=runner.db_config,
        job_config=runner.job_config,
        results_dir=tmp_path,
    )
    # The database holds the cost of update 1. The cost write of update 2
    # failed, thus only the live costs hold that cost. The island copy of the
    # program repeats both costs.
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / db_path) if db_path else None)
    )
    metadata = {"api_costs": 0.01, "meta_cost": 0.03, "meta_cost_update_ids": ["1"]}
    db.add(Program(id="program", code="", metadata=metadata), defer_maintenance=True)
    db.add(
        Program(id="copy", code="", metadata={**metadata, "_is_island_copy": True}),
        defer_maintenance=True,
    )
    runner.db = db
    live_costs = {"1": 0.03, "2": 0.05}
    runner.live_meta_costs = {"program": dict(live_costs), "copy": dict(live_costs)}

    asyncio.run(runner._finish_wandb_logging())
    db.close()

    # With a database file, a worker thread reads the programs from the file.
    # With a database in memory, the runner reads them. In each case, the
    # final W&B row and the final table merge the live costs with the costs of
    # the database by update id: each cost counts 1 time. A copy shows no
    # cost, as on SakanaAI main.
    final_row, final_table = run.payloads[-2:]
    assert final_row["cost/meta"] == pytest.approx(0.08)
    assert final_row["cost/total"] == pytest.approx(0.09)
    rows = final_table[PROGRAM_TABLE_KEY].data
    assert {row[0]: round(row[-1], 6) for row in rows} == {"program": 0.09, "copy": 0}


def record_final_console_summaries(patch) -> list[str]:
    """Record the final console summary of each launch.

    The runners of the tests have no verbose output, thus they print no final
    summary. Here each launch prints its final summary to a console that
    records it. Returns the text of each final summary.
    """
    texts: list[str] = []
    print_final_summary = ShinkaEvolveRunner._print_final_summary

    async def print_and_record(self):
        self.verbose, self.console = True, Console(record=True, width=200)
        await print_final_summary(self)
        texts.append(self.console.export_text())

    patch.setattr(ShinkaEvolveRunner, "_print_final_summary", print_and_record)
    return texts


def console_costs(text: str) -> dict[str, float]:
    """Give each total of the cost table of a final console summary, by name."""
    return {
        name: float(value)
        for name, value in re.findall(r"(Total \w+ Cost)\s*│\s*\$([\d.]+)", text)
    }


def final_table_meta_costs(run: FakeWandbRun, run_dir: Path) -> dict:
    """Give the meta cost of each evaluated program in the last final W&B table.

    A row of the table gives the sum of the costs of its program. The meta
    cost is that sum less the other costs, which the database holds. The
    result is by generation. As on SakanaAI main, the row of an island copy
    has no cost: the row of its program has the costs.
    """
    tables = [p[PROGRAM_TABLE_KEY] for p in run.payloads if PROGRAM_TABLE_KEY in p]
    row_costs = {row[0]: row[-1] for row in tables[-1].data}
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        programs = db.get_all_programs()
        evaluated_ids = {p.id for p in db.get_evaluated_programs()}
    finally:
        db.close()
    assert set(row_costs) == {p.id for p in programs}
    meta_costs = {}
    for program in programs:
        if is_island_copy(program):
            assert row_costs[program.id] == 0, program.metadata
            continue
        other_costs = sum(
            program.metadata.get(key) or 0.0
            for key in ("api_costs", "embed_cost", "novelty_cost")
        )
        meta_cost = round(row_costs[program.id] - other_costs, 6)
        if program.id in evaluated_ids:
            meta_costs[program.generation] = meta_cost
        else:
            assert meta_cost == 0, program.metadata
    return meta_costs


def stored_meta_costs_by_generation(run_dir: Path, with_copies: bool = False) -> dict:
    """Give the meta cost of each evaluated program of the database, by generation.

    With with_copies, also give the meta cost of each island copy, by
    generation and island.
    """
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        evaluated = db.get_evaluated_programs()
        costs: dict = {
            p.generation: round(p.metadata.get("meta_cost") or 0.0, 6)
            for p in evaluated
        }
        if with_copies:
            evaluated_ids = {p.id for p in evaluated}
            costs.update(
                {
                    (p.generation, p.island_idx): round(
                        p.metadata.get("meta_cost") or 0.0, 6
                    )
                    for p in db.get_all_programs()
                    if p.id not in evaluated_ids
                }
            )
        return costs
    finally:
        db.close()


def record_final_summaries(patch) -> list[tuple[int, float]]:
    """Record the generation of the best program and the cost of each final summary."""
    final_summaries: list[tuple[int, float]] = []
    summarize = AsyncMetaSummarizer.perform_final_summary_async

    async def summarize_and_record(self, results_dir, best_program, store_cost):
        success, meta_cost = await summarize(
            self, results_dir, best_program, store_cost
        )
        if success:
            final_summaries.append((best_program.generation, meta_cost))
        return success, meta_cost

    patch.setattr(
        AsyncMetaSummarizer, "perform_final_summary_async", summarize_and_record
    )
    return final_summaries


def pass_the_time_limit_then_fail(patch, first_update_only: bool = False) -> list[bool]:
    """Make each cost write wait until the final W&B table, then fail.

    Thus each cost write passes the time limit of the wait for the cost writes
    at the end of the launch, as when the database hangs. Each cost write
    waits before the write queue (see wait_before_the_write_queue()). With
    first_update_only, only the cost write of the first meta update waits, in
    its worker thread, as when that thread hangs: the other cost writes
    commit. Returns, for each write that waits, whether the final table came
    before its end.
    """
    final_table_logged = threading.Event()
    waits: list[bool] = []
    log_final = ShinkaWandbLogger.log_final
    add_meta_cost = ProgramDatabase.add_meta_cost
    first_update: dict[str, str] = {}

    def log_then_release(self, **kwargs):
        log_final(self, **kwargs)
        final_table_logged.set()

    def wait_in_the_worker_then_fail(db, program_id, meta_cost, update_id):
        if first_update.setdefault("id", update_id) != update_id:
            return add_meta_cost(db, program_id, meta_cost, update_id)
        waits.append(final_table_logged.wait(timeout=30))
        raise sqlite3.OperationalError("database is locked")

    async def wait_then_fail(self, *args):
        # The cleanup of the launch can stop the wait after the final table
        try:
            await wait_before_the_write_queue(final_table_logged)
        finally:
            waits.append(final_table_logged.is_set())
        raise sqlite3.OperationalError("database is locked")

    patch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 1.0)
    patch.setattr(ShinkaWandbLogger, "log_final", log_then_release)
    if first_update_only:
        patch.setattr(ProgramDatabase, "add_meta_cost", wait_in_the_worker_then_fail)
    else:
        patch.setattr(AsyncProgramDatabase, "add_meta_cost_async", wait_then_fail)
    return waits


def check_each_save(patch, run_dir: Path) -> list[list[str]]:
    """Check each save of the state file of run_dir against the database.

    Returns, for each save, the recommendation of each update of the save
    whose cost the database does not hold.
    """
    update_ids: dict[str, str] = {}
    queue_cost_write = AsyncMetaSummarizer._queue_cost_write

    def record_then_queue(self, update, store_cost):
        update_ids[update.recommendations] = update.update_id
        return queue_cost_write(self, update, store_cost)

    saves: list[list[str]] = []
    save = MetaSummarizer.save_meta_state

    def save_then_check(self, filepath):
        save(self, filepath)
        if Path(filepath) == run_dir / STATE_FILE:
            stored = set(stored_update_ids(run_dir))
            history = read_state(run_dir)["meta_recommendations_history"]
            saves.append([text for text in history if update_ids[text] not in stored])

    patch.setattr(AsyncMetaSummarizer, "_queue_cost_write", record_then_queue)
    patch.setattr(MetaSummarizer, "save_meta_state", save_then_check)
    return saves


# The failure points of a cost write, for the final readers of the costs.
FINAL_READER_FAILURES: dict[str, Callable[[Any], Any]] = {
    "fails 1 time": partial(fail_the_first_cost_writes, count=1),
    "fails 3 times": lambda patch: fail_the_first_cost_writes(
        patch, async_summarizer.META_COST_WRITE_TRIES
    ),
    "is slow": commit_each_cost_write_late,
    "passes the time limit": pass_the_time_limit_then_fail,
    "the first passes the time limit": partial(
        pass_the_time_limit_then_fail, first_update_only=True
    ),
}
# What the final readers show after a fresh search with no failure point, by
# interval.
REFERENCE_FINAL_READERS: dict = {}


def final_readers_of_a_search(run_dir, monkeypatch, interval, failure=None) -> dict:
    """Run a fresh search of 3 generations, with the failure point of a cost write.

    Return what the readers at the end of the launch show: the meta cost of
    each program in the final W&B table, the totals of the final console
    summary, and the cost total. Also return the generation of the program
    and the cost of each final summary, the meta cost of each program in the
    database, and the value that the failure point gave.
    """
    totals: list[float] = []
    with monkeypatch.context() as patch:
        run = record_wandb(patch)
        texts = record_final_console_summaries(patch)
        final_summaries = record_final_summaries(patch)
        failure_point = FINAL_READER_FAILURES[failure](patch) if failure else None
        search(run_dir, StubMetaLLM("launch 1"), interval, 3, totals=totals)
    (text,) = texts
    return {
        "table": final_table_meta_costs(run, run_dir),
        "console": console_costs(text),
        "cost total": round(totals[-1], 6),
        "final summaries": final_summaries,
        "stored": stored_meta_costs_by_generation(run_dir),
        "failure point": failure_point,
    }


def without_the_final_summary_cost(readers: dict) -> dict:
    """Give the final table and the console totals with no cost of the final summary."""
    table, console = dict(readers["table"]), dict(readers["console"])
    for generation, meta_cost in readers["final summaries"]:
        table[generation] = round(table[generation] - meta_cost, 6)
        for name in ("Total Meta Cost", "Total Combined Cost"):
            console[name] = round(console[name] - meta_cost, 2)
    return {"table": table, "console": console}


@pytest.mark.parametrize("path", FRESH_PATHS)
@pytest.mark.parametrize("failure", FINAL_READER_FAILURES)
def test_final_table_and_console_summary_are_as_on_main_then_two_resumes(
    tmp_path, monkeypatch, failure, path
):
    run_dir = tmp_path / "run"
    interval = FRESH_PATHS[path]
    saves = check_each_save(monkeypatch, run_dir)
    if interval not in REFERENCE_FINAL_READERS:
        REFERENCE_FINAL_READERS[interval] = final_readers_of_a_search(
            tmp_path / "reference", monkeypatch, interval
        )
    reference = REFERENCE_FINAL_READERS[interval]

    result = final_readers_of_a_search(run_dir, monkeypatch, interval, failure)

    # With no failure point, the database holds each cost, and the final
    # table and the console summary hold each cost one time.
    assert reference["table"] == reference["stored"]
    assert reference["console"]["Total Meta Cost"] == pytest.approx(
        sum(reference["stored"].values())
    )
    # main writes the cost of an update of the setup or of a job with the
    # metadata of its program. Thus a cost write of that update changes no
    # final table and no console summary of main. main writes the cost of its
    # final summary with 1 try, inside the time limit of the final summary.
    # Thus main loses that cost when the try fails or passes the time limit.
    # A failure point of the first cost write goes to that try only when the
    # final summary is the first update.
    main = {"table": reference["table"], "console": reference["console"]}
    final_summary_is_first = path == "final summary"
    first_write_fails = failure != "is slow"
    if failure == "passes the time limit" or (
        final_summary_is_first and first_write_fails
    ):
        main = without_the_final_summary_cost(reference)
    shown = {"table": result["table"], "console": result["console"]}
    if final_summary_is_first and failure == "fails 1 time":
        # Here the cost write tries again, and the database holds the cost.
        assert shown != main
        assert shown == {"table": reference["table"], "console": reference["console"]}
    else:
        assert shown == main
    # As on main, the cost total counts each update. main does not count a
    # final summary whose cost write passes its time limit.
    assert result["cost total"] == reference["cost total"]
    if failure.endswith("passes the time limit"):
        assert result["failure point"] and all(result["failure point"])

    # 2 resumes. Their final table and their console summary show the costs
    # of the database: a resume completes no job.
    for name in ("launch 2", "launch 3"):
        with monkeypatch.context() as patch:
            run = record_wandb(patch)
            texts = record_final_console_summaries(patch)
            launch(run_dir, StubMetaLLM(name), [], meta_rec_interval=interval)
        stored = stored_meta_costs_by_generation(run_dir)
        assert final_table_meta_costs(run, run_dir) == stored
        assert console_costs(texts[-1])["Total Meta Cost"] == pytest.approx(
            round(sum(stored.values()), 2)
        )
    # Each program got its summary. No save holds an update whose cost the
    # database does not hold.
    assert waiting_generations(read_state(run_dir)) == []
    assert saves and not any(saves)


# The paths whose meta update writes its cost to the program of the update,
# in a launch whose new island then copies that program. Each path gives the
# interval, the generations of the jobs, the scores that make that program
# the best program, and its generation.
ISLAND_COPY_PATHS = {
    "initial program": (1, [1], {0: 2.0}, 0),
    "completed job": (2, [1, 2], {1: 2.0}, 1),
}


def commit_the_first_cost_write_after_the_end_wait(patch) -> None:
    """Make the cost write of the first meta update commit after the final W&B table.

    Its worker thread hangs until that table, after the time limit of the
    wait for the cost writes at the end of the launch. Then the write commits.
    """
    final_table_logged = threading.Event()
    log_final = ShinkaWandbLogger.log_final
    add_meta_cost = ProgramDatabase.add_meta_cost
    first_update: dict[str, str] = {}

    def log_then_release(self, **kwargs):
        log_final(self, **kwargs)
        final_table_logged.set()

    def wait_then_write(db, program_id, meta_cost, update_id):
        if first_update.setdefault("id", update_id) == update_id:
            assert final_table_logged.wait(timeout=30)
        return add_meta_cost(db, program_id, meta_cost, update_id)

    patch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 1.0)
    patch.setattr(ShinkaWandbLogger, "log_final", log_then_release)
    patch.setattr(ProgramDatabase, "add_meta_cost", wait_then_write)


# The failure points of the cost write of the update of a copied program.
ISLAND_COPY_FAILURES: dict[str, Callable[[Any], Any]] = {
    "fails 1 time": FINAL_READER_FAILURES["fails 1 time"],
    "fails 3 times": FINAL_READER_FAILURES["fails 3 times"],
    "commits late": commit_the_first_cost_write_after_the_end_wait,
}
# One island at the start. A job with no better score makes a new island,
# which gets a copy of the best program, with its metadata in the database.
DYNAMIC_ISLANDS = {
    "num_islands": 1,
    "enable_dynamic_islands": True,
    "stagnation_threshold": 1,
    "island_spawn_strategy": "best",
}


def launch_with_no_wait(run_dir, meta_llm, generations, interval, **options) -> None:
    """Run one launch that completes one job for each generation of the list.

    As launch(), but the launch does not wait for the cost writes after a
    job (see make_runner() for the options).
    """
    runner = make_runner(run_dir, meta_llm, interval, **options)

    async def complete_jobs():
        try:
            for generation in generations:
                await complete_job(runner, generation)
        finally:
            runner.should_stop.set()
            runner.finalization_complete.set()

    async def no_proposals():
        return None

    runner._job_monitor_task = complete_jobs
    runner._proposal_coordinator_task = no_proposals
    asyncio.run(runner.run_async())


def final_readers_with_dynamic_islands(
    run_dir, monkeypatch, path, name, generations, failure=None
) -> dict:
    """Run one launch with dynamic islands, with the failure point of a cost write.

    Return the meta cost of each evaluated program in the final W&B table, and
    the totals of the final console summary.
    """
    interval, _, scores, _ = ISLAND_COPY_PATHS[path]
    with monkeypatch.context() as patch:
        run = record_wandb(patch)
        texts = record_final_console_summaries(patch)
        if failure:
            ISLAND_COPY_FAILURES[failure](patch)
        launch_with_no_wait(
            run_dir,
            StubMetaLLM(name),
            generations,
            interval,
            scores=scores,
            db_options=DYNAMIC_ISLANDS,
        )
    (text,) = texts
    return {
        "table": final_table_meta_costs(run, run_dir),
        "console": console_costs(text),
    }


@pytest.mark.parametrize("path", ISLAND_COPY_PATHS)
@pytest.mark.parametrize("failure", ISLAND_COPY_FAILURES)
def test_island_copy_shows_the_meta_cost_of_its_program_as_on_main(
    tmp_path, monkeypatch, failure, path
):
    run_dir = tmp_path / "run"
    _, generations, _, update_generation = ISLAND_COPY_PATHS[path]
    reference = final_readers_with_dynamic_islands(
        tmp_path / "reference", monkeypatch, path, "launch 1", generations
    )

    result = final_readers_with_dynamic_islands(
        run_dir, monkeypatch, path, "launch 1", generations, failure
    )

    # The new island 1 copies the program of the update after the update. As
    # on main, the copy repeats the cost that its program has at that time, in
    # the database and in the console summary. As on SakanaAI main, the final
    # W&B table gives no cost to the copy.
    copy_key = (update_generation, 1)
    stored = stored_meta_costs_by_generation(tmp_path / "reference", with_copies=True)
    assert 0 < stored[copy_key] <= stored[update_generation]
    assert reference["console"]["Total Meta Cost"] == pytest.approx(
        round(sum(stored.values()), 2)
    )
    # main writes the cost of an update of the setup or of a job with the
    # metadata of its program, and a new island copies that metadata. Thus a
    # cost write changes no row of the final table and no console summary of
    # main, also the cost of the copy.
    assert result == reference

    # A resume. Its final readers show the costs of the database. When the
    # cost write commits after the copy, the copy gets the cost in the
    # database at the end of the launch, as on main. When each try of the
    # cost write fails, the database holds that cost for neither the program
    # nor its copy, and the resume makes the update again. main keeps the
    # cost of the copy: a limit of this design.
    resume = final_readers_with_dynamic_islands(
        run_dir, monkeypatch, path, "launch 2", []
    )
    stored = stored_meta_costs_by_generation(run_dir, with_copies=True)
    assert resume["table"] == stored_meta_costs_by_generation(run_dir)
    assert resume["console"]["Total Meta Cost"] == pytest.approx(
        round(sum(stored.values()), 2)
    )
    assert (stored[copy_key] > 0) == (failure != "fails 3 times")


class HardStop(BaseException):
    """A stop of the process with no cleanup, as a kill gives."""


def hold_the_queue_worker_after_the_jobs(
    patch, separate_write: str = "commits", hard_stop: bool = False
) -> dict[str, list[str]]:
    """Make the cost write of the first meta update hold the 1 worker of the queue.

    That write waits before the write queue until the jobs end. Then its
    worker thread hangs until the final W&B table, past the time limit of the
    wait for the cost writes at the end of the launch, and the write commits.
    The final summary starts when that write holds the worker, thus the cost
    write of the final summary waits in the queue. The database accepts the
    other writes meanwhile: a cost write in an other thread commits, fails, or
    passes its time limit (separate_write).

    With hard_stop, the process stops after the final table, as a kill does:
    the held write and the queued writes do not commit, and no cleanup comes.

    Returns the update id of each cost write in the worker of the queue and in
    an other thread.
    """
    jobs_ended, worker_held, table_logged, killed = (
        threading.Event() for _ in range(4)
    )
    writes: dict[str, list[str]] = {"queue": [], "other thread": []}
    worker: list[threading.Thread] = []
    add_meta_cost = ProgramDatabase.add_meta_cost
    add_meta_cost_async = AsyncProgramDatabase.add_meta_cost_async
    summarize = AsyncMetaSummarizer.perform_final_summary_async
    log_final = ShinkaWandbLogger.log_final

    async def wait_for_the_jobs_then_queue(self, *args):
        assert await wait_before_the_write_queue(jobs_ended)
        return await add_meta_cost_async(self, *args)

    def hold_then_write(db, program_id, meta_cost, update_id):
        if not worker:
            worker.append(threading.current_thread())
            writes["queue"].append(update_id)
            worker_held.set()
            assert (killed if hard_stop else table_logged).wait(timeout=30)
            if hard_stop:
                raise sqlite3.OperationalError("the process stopped")
        elif threading.current_thread() is worker[0]:
            writes["queue"].append(update_id)
        else:
            writes["other thread"].append(update_id)
            if separate_write == "fails":
                raise sqlite3.OperationalError("database is locked")
            if separate_write == "passes its time limit":
                assert table_logged.wait(timeout=30)
        return add_meta_cost(db, program_id, meta_cost, update_id)

    async def start_when_the_worker_is_held(self, *args, **kwargs):
        jobs_ended.set()
        assert await wait_before_the_write_queue(worker_held)
        return await summarize(self, *args, **kwargs)

    def log_then_release(self, **kwargs):
        log_final(self, **kwargs)
        table_logged.set()

    async def stop_the_process(self):
        # The queued writes do not run, and the held write does not commit
        self.write_executor.shutdown(wait=False, cancel_futures=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
        killed.set()
        raise HardStop

    patch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 1.0)
    patch.setattr(
        AsyncProgramDatabase, "add_meta_cost_async", wait_for_the_jobs_then_queue
    )
    patch.setattr(ProgramDatabase, "add_meta_cost", hold_then_write)
    patch.setattr(
        AsyncMetaSummarizer,
        "perform_final_summary_async",
        start_when_the_worker_is_held,
    )
    patch.setattr(ShinkaWandbLogger, "log_final", log_then_release)
    if hard_stop:
        patch.setattr(AsyncProgramDatabase, "close_async", stop_the_process)
    return writes


def final_readers_with_one_worker(run_dir, monkeypatch, name, model=None) -> dict:
    """Run one launch with 1 worker in the write queue, and a model of its writes.

    The update of job 1 writes its cost to program 1, the best program. Job 2
    waits for the final summary, which writes its cost to the same program.
    Return the meta cost of each program in the final W&B table, the totals of
    the final console summary, the programs of each meta update, what the
    model returns, and whether the launch stopped.
    """
    meta_llm = StubMetaLLM(name)
    stopped = False
    with monkeypatch.context() as patch:
        run = record_wandb(patch)
        texts = record_final_console_summaries(patch)
        model_result = model(patch) if model else None
        try:
            launch_with_no_wait(
                run_dir, meta_llm, [1, 2], 2, scores={1: 2.0}, max_db_workers=1
            )
        except HardStop:
            stopped = True
    return {
        "table": final_table_meta_costs(run, run_dir),
        "console": console_costs(texts[-1]) if texts else None,
        "updates": meta_llm.batches,
        "model": model_result,
        "stopped": stopped,
    }


@pytest.mark.parametrize("hard_stop", [False, True], ids=["ends", "hard stop"])
def test_final_cost_write_behind_a_held_queue_worker_is_as_on_main(
    tmp_path, monkeypatch, hard_stop
):
    run_dir = tmp_path / "run"
    reference = final_readers_with_one_worker(
        tmp_path / "reference", monkeypatch, "launch 1"
    )
    saves = check_each_save(monkeypatch, run_dir)

    result = final_readers_with_one_worker(
        run_dir,
        monkeypatch,
        "launch 1",
        partial(hold_the_queue_worker_after_the_jobs, hard_stop=hard_stop),
    )

    # main writes the cost of its final summary in a free thread. Here, after
    # the wait for the cost writes, that write goes outside the write queue
    # one time. Thus the final table and the console summary are as on main.
    assert result["stopped"] == hard_stop
    assert result["table"] == reference["table"]
    writes = result["model"]
    assert len(writes["other thread"]) == 1
    assert writes["other thread"][0] != writes["queue"][0]
    if hard_stop:
        # The cost write of the first update did not commit before the stop
        final_cost = round(update_costs(result["updates"][1:]), 6)
        assert stored_meta_costs_by_generation(run_dir) == {0: 0, 1: final_cost, 2: 0}
    else:
        assert result["console"] == reference["console"]
        # The queued write of the final summary runs after the held write. It
        # has the same update id, thus the database holds each cost 1 time.
        assert len(writes["queue"]) == 2
        paid = round(update_costs(result["updates"]), 6)
        assert stored_meta_costs_by_generation(run_dir) == {0: 0, 1: paid, 2: 0}
    assert saves and not any(saves)

    # A resume pays the final summary 1 time. After the hard stop, it makes
    # the first update again, because the database does not hold its cost.
    resume_llm = StubMetaLLM("launch 2")
    launch(run_dir, resume_llm, [], meta_rec_interval=2, max_db_workers=1)
    assert resume_llm.batches == ([[0, 1]] if hard_stop else [])
    assert waiting_generations(read_state(run_dir)) == []


@pytest.mark.parametrize("separate_write", ["fails", "passes its time limit"])
def test_failed_final_cost_write_outside_the_queue_goes_to_the_log(
    tmp_path, monkeypatch, caplog, separate_write
):
    run_dir = tmp_path / "run"
    reference = final_readers_with_one_worker(
        tmp_path / "reference", monkeypatch, "launch 1"
    )
    saves = check_each_save(monkeypatch, run_dir)
    monkeypatch.setattr(async_runner, "META_FINAL_COST_WRITE_TIMEOUT", 0.5)

    result = final_readers_with_one_worker(
        run_dir,
        monkeypatch,
        "launch 1",
        partial(hold_the_queue_worker_after_the_jobs, separate_write=separate_write),
    )

    # As the final write of main, the write outside the queue has 1 try with
    # a time limit, and a failure goes to the log. Thus the final table does
    # not show that cost, as on main when its write fails.
    assert len(result["model"]["other thread"]) == 1
    assert "Failed to store final meta cost in database" in caplog.text
    assert ("TimeoutError" in caplog.text) == (separate_write != "fails")
    final_cost = round(update_costs(result["updates"][1:]), 6)
    assert result["table"] == {
        **reference["table"],
        1: round(reference["table"][1] - final_cost, 6),
    }
    # The queued write commits after the held write, at the end of the launch.
    # Then the database holds each cost 1 time, and the state file holds the
    # update: a resume does not pay the final summary again.
    paid = round(update_costs(result["updates"]), 6)
    assert stored_meta_costs_by_generation(run_dir) == {0: 0, 1: paid, 2: 0}
    assert saves and not any(saves)
    resume_llm = StubMetaLLM("launch 2")
    launch(run_dir, resume_llm, [], meta_rec_interval=2, max_db_workers=1)
    assert resume_llm.batches == []


def make_meta_program(generation: int) -> Program:
    """Give a correct program of a generation, for a meta update."""
    program = make_program(f"gen-{generation}", generation, score=1.0)
    program.code = PROGRAM.format(generation=generation)
    program.metadata = {"patch_name": f"patch_{generation}"}
    return program


def make_summarizer(meta_llm: StubMetaLLM, state_path: Path | None = None):
    """Build a meta summarizer with generations 0 and 1 in its waiting list."""
    summarizer = AsyncMetaSummarizer(
        MetaSummarizer(async_mode=True),
        meta_llm,
        state_path=str(state_path) if state_path else None,
    )
    for generation in (0, 1):
        summarizer.add_evaluated_program(make_meta_program(generation))
    return summarizer


def test_update_takes_its_result_while_its_cost_write_waits():
    summarizer = make_summarizer(StubMetaLLM("launch 1"))

    async def store_cost(meta_cost, update_id):
        await asyncio.Event().wait()

    async def update():
        result = await asyncio.wait_for(
            summarizer.update_meta_memory_async(None, store_cost), timeout=5
        )
        # As on main, the state in memory takes the result after step 3: the
        # next proposal gets the recommendation of the update.
        assert summarizer.get_sampled_recommendation() == (
            "Recommendation of launch 1 update 1."
        )
        assert summarizer.total_programs_processed == 2
        assert summarizer.evaluated_since_last_meta == []
        # The cost write waits.
        assert len(summarizer.unsaved_updates) == 1
        await summarizer.stop_cost_writes()
        return result

    recommendations, meta_cost = asyncio.run(update())

    assert recommendations == "1. Recommendation of launch 1 update 1."
    assert meta_cost == pytest.approx(0.04)


def test_cost_write_outside_the_queue_keeps_the_update_after_a_failed_try(tmp_path):
    summarizer = make_summarizer(StubMetaLLM("launch 1"), tmp_path / STATE_FILE)
    may_fail = asyncio.Event()

    async def store_cost(meta_cost, update_id):
        await may_fail.wait()
        raise sqlite3.OperationalError("database is locked")

    async def update():
        await summarizer.update_meta_memory_async(None, store_cost)
        # The end of a launch writes the cost outside the write queue, while
        # the queued write waits
        (meta_update,) = summarizer.unsaved_updates
        meta_update.cost_stored = True
        may_fail.set()
        await summarizer.wait_for_cost_writes()

    asyncio.run(update())

    # Each try of the queued write then fails. The database holds the cost,
    # thus the saved state holds the update.
    assert summarizer.unsaved_updates == []
    assert read_state(tmp_path)["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 1."
    ]


def test_two_updates_can_wait_for_their_cost_at_the_same_time(tmp_path):
    meta_llm = StubMetaLLM("launch 1")
    state_path = tmp_path / STATE_FILE
    summarizer = make_summarizer(meta_llm, state_path)
    # The id of each cost write, and the event that lets it end.
    writes: list[tuple[str, asyncio.Event]] = []

    async def store_cost(meta_cost, update_id):
        may_end = asyncio.Event()
        writes.append((update_id, may_end))
        await may_end.wait()

    async def two_updates():
        # Update 1 takes generations 0 and 1, and its cost write waits.
        await summarizer.update_meta_memory_async(None, store_cost)
        for generation in (2, 3):
            summarizer.add_evaluated_program(make_meta_program(generation))
        # Update 2 takes generations 2 and 3. Its cost write does not wait for
        # the cost write of update 1: a late write does not stop a later one.
        await summarizer.update_meta_memory_async(None, store_cost)
        await asyncio.sleep(0.05)
        assert len(writes) == 2
        # The state in memory holds the 2 updates, as on main. The 2 updates
        # wait for their cost, thus a save holds neither of them.
        assert len(summarizer.meta_recommendations_history) == 2
        assert len(summarizer.unsaved_updates) == 2
        summarizer.save_meta_state(str(state_path))
        states = [read_state(tmp_path)]
        # The cost of update 2 commits first, and update 2 waits for the cost
        # write of update 1. A save holds an update exactly when the database
        # holds its cost, thus its task saves update 2 at once.
        writes[1][1].set()
        await asyncio.sleep(0.05)
        assert len(summarizer.unsaved_updates) == 2
        states.append(read_state(tmp_path))
        # The cost of update 1 commits: the save holds update 1, then update 2.
        writes[0][1].set()
        await summarizer.wait_for_cost_writes()
        states.append(read_state(tmp_path))
        return states

    states = asyncio.run(two_updates())

    assert meta_llm.batches == [[0, 1], [2, 3]]
    assert [state["meta_recommendations_history"] for state in states] == [
        [],
        ["1. Recommendation of launch 1 update 2."],
        [
            "1. Recommendation of launch 1 update 1.",
            "1. Recommendation of launch 1 update 2.",
        ],
    ]
    assert [waiting_generations(state) for state in states] == [
        [0, 1, 2, 3],
        [0, 1],
        [],
    ]
    assert [state["total_programs_meta_processed"] for state in states] == [0, 2, 4]
    assert states[0]["meta_recommendations"] is None
    assert summary_labels(states[1]["meta_summary"]) == [(2, 2), (3, 3)]
    assert states[2]["meta_recommendations"] == (
        "1. Recommendation of launch 1 update 2."
    )
    assert summary_labels(states[2]["meta_summary"]) == [(g, g) for g in range(4)]
    assert summarizer.unsaved_updates == []
    assert writes[0][0] != writes[1][0]


def test_failed_cost_write_keeps_the_order_of_the_saved_state(tmp_path):
    meta_llm = StubMetaLLM("launch 1")
    summarizer = make_summarizer(meta_llm, tmp_path / STATE_FILE)
    update_numbers: dict[str, int] = {}
    first_may_end: list[asyncio.Event] = []

    async def store_cost(meta_cost, update_id):
        number = update_numbers.setdefault(update_id, len(update_numbers) + 1)
        if number == 1:
            await first_may_end[0].wait()
        if number == 2:
            raise sqlite3.OperationalError("database is locked")

    async def three_updates():
        first_may_end.append(asyncio.Event())
        # Update 1 takes generations 0 and 1, and its cost write waits.
        await summarizer.update_meta_memory_async(None, store_cost)
        # Each try of the cost write of update 2 fails, and the cost of update
        # 3 commits.
        for generations in ((2, 3), (4, 5)):
            for generation in generations:
                summarizer.add_evaluated_program(make_meta_program(generation))
            await summarizer.update_meta_memory_async(None, store_cost)
        await asyncio.sleep(0.05)
        unsaved = len(summarizer.unsaved_updates)
        first_may_end[0].set()
        await summarizer.wait_for_cost_writes()
        return unsaved

    unsaved = asyncio.run(three_updates())

    # Update 3 waits for update 1, also after the failed cost write of update
    # 2. Thus the saved state takes update 1, then update 3, and the programs
    # of update 2 wait.
    assert meta_llm.batches == [[0, 1], [2, 3], [4, 5]]
    assert unsaved == 3
    state = read_state(tmp_path)
    assert state["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 1.",
        "1. Recommendation of launch 1 update 3.",
    ]
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 3."
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (4, 4), (5, 5)]
    assert waiting_generations(state) == [2, 3]


def test_failed_cost_write_leaves_the_next_update_to_the_saved_state(tmp_path):
    meta_llm = StubMetaLLM("launch 1")
    summarizer = make_summarizer(meta_llm, tmp_path / STATE_FILE)
    tries = []

    async def store_cost(meta_cost, update_id):
        tries.append(update_id)
        if len(tries) <= async_summarizer.META_COST_WRITE_TRIES:
            raise sqlite3.OperationalError("database is locked")

    async def two_updates():
        # Each try of the cost write of update 1 fails.
        await summarizer.update_meta_memory_async(None, store_cost)
        await summarizer.wait_for_cost_writes()
        for generation in (2, 3):
            summarizer.add_evaluated_program(make_meta_program(generation))
        # The cost write of update 2 commits.
        await summarizer.update_meta_memory_async(None, store_cost)
        await summarizer.wait_for_cost_writes()

    asyncio.run(two_updates())

    # Update 1 gets its 3 tries with one update id. As on main, the next
    # update is a new update: no new LLM query for generations 0 and 1.
    tries_of_update_1 = tries[: async_summarizer.META_COST_WRITE_TRIES]
    assert len(set(tries_of_update_1)) == 1
    assert tries[-1] not in tries_of_update_1
    assert meta_llm.batches == [[0, 1], [2, 3]]
    assert summarizer.meta_recommendations == "1. Recommendation of launch 1 update 2."
    # The saved state holds update 2, whose cost the database holds. The
    # programs of update 1 wait in the saved state, thus a resume makes that
    # update again.
    state = read_state(tmp_path)
    assert state["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 2."
    ]
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 2."
    assert state["meta_scratch_pad"] == "Insights of launch 1 update 2."
    assert summary_labels(state["meta_summary"]) == [(2, 2), (3, 3)]
    assert state["total_programs_meta_processed"] == 2
    assert state["processed_program_ids"] == ["gen-2", "gen-3"]
    assert waiting_generations(state) == [0, 1]


def test_update_with_no_cost_makes_no_cost_write(tmp_path):
    class FreeMetaLLM(StubMetaLLM):
        async def query(self, msg, system_msg):
            response = await super().query(msg, system_msg)
            return StubResponse(response.content, cost=0.0)

    summarizer = make_summarizer(FreeMetaLLM("launch 1"), tmp_path / STATE_FILE)
    cost_writes = []

    async def store_cost(meta_cost, update_id):
        cost_writes.append(meta_cost)

    async def update():
        result = await summarizer.update_meta_memory_async(None, store_cost)
        await summarizer.wait_for_cost_writes()
        return result

    recommendations, meta_cost = asyncio.run(update())

    # The database holds no cost of the update, thus the saved state takes it.
    assert cost_writes == []
    assert recommendations == "1. Recommendation of launch 1 update 1."
    assert meta_cost == 0.0
    assert summarizer.total_programs_processed == 2
    assert summarizer.unsaved_updates == []
    assert read_state(tmp_path)["total_programs_meta_processed"] == 2


def test_cost_write_to_a_program_that_the_database_does_not_hold(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    db = ProgramDatabase(config=config, embedding_model="")
    runner = make_runner(tmp_path / "run", StubMetaLLM("launch 1"), 2)
    summarizer = make_summarizer(StubMetaLLM("launch 1"), tmp_path / STATE_FILE)
    program = make_program("gen-9", generation=9, score=1.0)

    async def update():
        runner.async_db = AsyncProgramDatabase(db)
        try:
            store_cost = partial(runner._store_meta_cost, program)
            result = await summarizer.update_meta_memory_async(None, store_cost)
            await summarizer.wait_for_cost_writes()
            return result
        finally:
            await runner.async_db.close_async()
            db.close()

    recommendations, _ = asyncio.run(update())

    # As on main, the state in memory takes the update. The database holds no
    # cost of the update, thus no saved state holds it.
    assert recommendations == "1. Recommendation of launch 1 update 1."
    assert summarizer.total_programs_processed == 2
    assert len(summarizer.unsaved_updates) == 1
    summarizer.save_meta_state(str(tmp_path / STATE_FILE))
    state = read_state(tmp_path)
    assert state["total_programs_meta_processed"] == 0
    assert waiting_generations(state) == [0, 1]


def write_the_output_file_after_the_stop(patch) -> None:
    """Make the output file write of a meta update end after the stop of the launch.

    The caller of the update waits for that write until the stop reaches it.
    Then the write runs, as a write in progress on main ends after a stop,
    and the stop continues. A real write can end before the stop reaches the
    caller. Then on Python 3.10 and 3.11, asyncio.wait_for() gives the result
    of the final summary or of the update of a resume, and the launch
    continues to its end.
    """
    write_meta_output_async = AsyncMetaSummarizer.write_meta_output_async

    async def write_after_the_stop(self, results_dir):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await write_meta_output_async(self, results_dir)
            raise

    patch.setattr(AsyncMetaSummarizer, "write_meta_output_async", write_after_the_stop)


@pytest.mark.parametrize("path", ["completed job", "resume", "final summary"])
def test_stop_between_step_3_and_the_cost_commit_then_two_resumes(
    tmp_path, monkeypatch, path
):
    run_dir = tmp_path / "run"
    interval, num_generations = UPDATE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # Launch 1 stops as a stop of the process does, at the start of the cost
    # write of its first update: after step 3, before the commit of the cost.
    # At that time, the update waits for the write of its output file.
    first_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        write_the_output_file_after_the_stop(patch)
        prompts = search(
            run_dir,
            first_llm,
            interval,
            num_generations,
            stop_at=("_store_meta_cost", 1),
        )
    assert first_llm.batches == [[0, 1]]
    stored = 0.01 * len(prompts)

    # As a write in progress on main, the write of the output file of the
    # update ends after the stop.
    assert meta_files(run_dir) == ["meta_2.txt"]
    output = (run_dir / "meta" / "meta_2.txt").read_text(encoding="utf-8")
    assert "1. Recommendation of launch 1 update 1." in output
    # The database holds no cost of the update, thus no state file holds it.
    assert stored_meta_costs(run_dir) == 0
    assert read_state(run_dir) is None

    # A resume, then a second resume.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=interval)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=interval)

    # The resume makes the update again, and the second resume makes none.
    assert second_llm.batches == [[0, 1]]
    assert third_llm.batches == []
    # The cost total of each later launch holds each update whose cost the
    # database holds. The total of the resume is the total at its first
    # proposal.
    paid_at_the_first_proposal = update_costs(second_llm.batches[: second.updates])
    assert second.api_cost == pytest.approx(stored + paid_at_the_first_proposal)
    assert third.api_cost == pytest.approx(stored + update_costs(second_llm.batches))
    assert stored_meta_costs(run_dir) == pytest.approx(update_costs(second_llm.batches))
    # The state holds each program one time, and the output file of the
    # update of the resume takes the place of the first one.
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1)]
    assert meta_files(run_dir) == ["meta_2.txt"]
    output = (run_dir / "meta" / "meta_2.txt").read_text(encoding="utf-8")
    assert "1. Recommendation of launch 2 update 1." in output


def test_stop_between_step_3_and_the_cost_commit_of_the_initial_program(tmp_path):
    run_dir = tmp_path / "run"
    # Launch 1 stops as a stop of the process does, at the start of the cost
    # write of the update of generation 0.
    first_llm = StubMetaLLM("launch 1")
    search(run_dir, first_llm, 1, num_generations=2, stop_at=("_store_meta_cost", 1))
    assert first_llm.batches == [[0]]
    # The stop can cut the write of the output file of the update, as on main.
    # The database holds no cost of the update, thus no state file holds it.
    first_files = meta_files(run_dir)
    assert first_files in ([], ["meta_1.txt"])
    assert stored_meta_costs(run_dir) == 0
    assert read_state(run_dir) is None

    # The database holds generation 0 only, thus each of the 2 next launches
    # evaluates the initial program again: a new program, with its own summary.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=1)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=1)

    # Launch 2 summarizes the 2 programs of generation 0. The cost total of
    # each later launch holds each update whose cost the database holds.
    assert second_llm.batches == [[0, 0]]
    assert third_llm.batches == [[0]]
    paid = update_costs(second_llm.batches)
    assert second.api_cost == pytest.approx(paid)
    assert third.api_cost == pytest.approx(paid + update_costs(third_llm.batches))
    assert meta_files(run_dir) == [*first_files, "meta_2.txt", "meta_3.txt"]
    state = read_state(run_dir)
    assert state["total_programs_meta_processed"] == 3
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )


def test_stop_after_the_failed_cost_write_of_the_final_summary_keeps_its_output_file(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    # A fresh search of 4 programs, at the interval 5. Each cost write of the
    # final summary fails. Then the launch stops as a stop of the process does,
    # before the save of the bandit state.
    meta_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        fail_cost_writes(patch)
        search(run_dir, meta_llm, 5, 4, stop_at=("_save_bandit_state", 1))

    # As on main, the output file of the final summary is there.
    assert meta_llm.batches == [[0, 1, 2, 3]]
    assert meta_files(run_dir) == ["meta_4.txt"]
    output = (run_dir / "meta" / "meta_4.txt").read_text(encoding="utf-8")
    assert "1. Recommendation of launch 1 update 1." in output
    # The database holds no cost of the update, thus the state file does not
    # hold the update.
    assert stored_meta_costs(run_dir) == 0
    state = read_state(run_dir)
    assert state["meta_recommendations"] is None
    assert waiting_generations(state) == [0, 1, 2, 3]

    # A resume, then a second resume: the final summary of the resume makes
    # the update again, and the second resume makes none.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    launch(run_dir, second_llm, generations=[], meta_rec_interval=5)
    launch(run_dir, third_llm, generations=[], meta_rec_interval=5)
    assert second_llm.batches == [[0, 1, 2, 3]]
    assert third_llm.batches == []
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert state["meta_recommendations"] == "1. Recommendation of launch 2 update 1."
    assert meta_files(run_dir) == ["meta_4.txt"]


def test_cost_write_that_ends_after_the_launch_goes_into_the_state_file(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    # A fresh search of 2 correct programs. The cost write of its final
    # summary takes 1.5 seconds, more than the wait for the cost writes at the
    # end of the launch (a delay of 0.5 seconds, then the time limit). The
    # cleanup of the launch waits for the writers of the database.
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    # The update ids that the database holds at the late save of the state.
    late_takes: list[set] = []
    take_stored_updates = AsyncMetaSummarizer.take_stored_updates

    def record_then_take(self, stored_update_ids):
        late_takes.append(set(stored_update_ids))
        take_stored_updates(self, stored_update_ids)

    with monkeypatch.context() as patch:
        patch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 0.1)
        commit_each_cost_write_late(patch, seconds=1.5)
        patch.setattr(AsyncMetaSummarizer, "take_stored_updates", record_then_take)
        prompts = search(run_dir, meta_llm, 5, 2, totals=totals)

    # The write commits in the cleanup. Then the late save of the state takes
    # the update. As on main, the output file holds the update, and the cost
    # total counts it one time.
    assert [len(ids) for ids in late_takes] == [1]
    assert meta_llm.batches == [[0, 1]]
    paid = update_costs(meta_llm.batches)
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    assert meta_files(run_dir) == ["meta_2.txt"]
    state = read_state(run_dir)
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 1."
    assert waiting_generations(state) == []
    paid += 0.01 * len(prompts)
    assert totals[-1] == pytest.approx(paid)

    # A resume, then a second resume: no new update, and each total holds the
    # paid update one time.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=5)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=5)
    assert second_llm.batches == third_llm.batches == []
    assert second.api_cost == third.api_cost == pytest.approx(paid)
    assert meta_files(run_dir) == ["meta_2.txt"]


def hold_the_first_cost_write(patch) -> None:
    """Make the cost write of the first meta update wait until the process stops."""
    add_meta_cost_async = AsyncProgramDatabase.add_meta_cost_async
    first_update: dict[str, str] = {}

    async def hold_then_write(self, program_id, meta_cost, update_id):
        if first_update.setdefault("id", update_id) == update_id:
            await asyncio.Event().wait()
        return await add_meta_cost_async(self, program_id, meta_cost, update_id)

    patch.setattr(AsyncProgramDatabase, "add_meta_cost_async", hold_then_write)


def test_stop_while_an_update_waits_for_an_earlier_cost_write(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    # The cost write of the update of generation 1 does not end. The cost of
    # the final summary commits, then the process stops with no cleanup at
    # the wait for the cost writes.
    first_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        hold_the_first_cost_write(patch)
        prompts = search(run_dir, first_llm, 2, 3, stop_at=("wait_for_cost_writes", 1))
    assert first_llm.batches == [[0, 1], [2]]
    paid = update_costs(first_llm.batches[1:])
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    paid += 0.01 * len(prompts)

    # The state file holds the final summary, whose cost the database holds.
    # Thus the resume makes the first update again, and not the final summary.
    state = read_state(run_dir)
    assert state["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 2."
    ]
    assert waiting_generations(state) == [0, 1]
    second_llm = StubMetaLLM("launch 2")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=2)
    assert second_llm.batches == [[0, 1]]
    assert second.final_api_cost == pytest.approx(
        paid + update_costs(second_llm.batches)
    )
    assert waiting_generations(read_state(run_dir)) == []


@pytest.mark.parametrize("cost_write", ["fails each try", "is slow, then fails"])
def test_final_summary_with_no_cost_in_the_database_is_the_final_summary_of_main(
    tmp_path, monkeypatch, cost_write
):
    run_dir = tmp_path / "run"
    # A fresh search of 2 correct programs. The cost of its final summary does
    # not get into the database: each try of the write fails, or a slow write
    # ends after the wait at the end of the launch, then fails.
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    with monkeypatch.context() as patch:
        if cost_write == "fails each try":
            fail_cost_writes(patch)
        else:
            patch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 0.1)
            patch.setattr(ProgramDatabase, "add_meta_cost", slow_cost_write)
        prompts = search(run_dir, meta_llm, 5, 2, totals=totals)

    # As on main, the output file holds the update, and the cost total counts
    # its cost.
    assert meta_llm.batches == [[0, 1]]
    stored = 0.01 * len(prompts)
    assert totals[-1] == pytest.approx(stored + update_costs(meta_llm.batches))
    assert meta_files(run_dir) == ["meta_2.txt"]
    output = (run_dir / "meta" / "meta_2.txt").read_text(encoding="utf-8")
    assert "1. Recommendation of launch 1 update 1." in output
    assert summary_labels(output) == [(0, 0), (1, 1)]
    # The database holds no cost of the update, thus the state file does not
    # hold the update.
    assert stored_meta_costs(run_dir) == 0
    state = read_state(run_dir)
    assert state["meta_recommendations"] is None
    assert waiting_generations(state) == [0, 1]

    # A resume, then a second resume. The resume makes the update again, and
    # its output file takes the place of the first one.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=5)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=5)
    assert second_llm.batches == [[0, 1]]
    assert third_llm.batches == []
    assert second.api_cost == pytest.approx(stored)
    assert third.api_cost == pytest.approx(stored + update_costs(second_llm.batches))
    assert meta_files(run_dir) == ["meta_2.txt"]
    output = (run_dir / "meta" / "meta_2.txt").read_text(encoding="utf-8")
    assert "1. Recommendation of launch 2 update 1." in output


@pytest.mark.parametrize("path", ["resume", "final summary"])
def test_failed_cost_write_of_an_update_keeps_each_cost(tmp_path, monkeypatch, path):
    run_dir = tmp_path / "run"
    interval, _ = UPDATE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # Launch 1 pays the 3 steps of its update, and each write of a cost fails.
    first_llm = StubMetaLLM("launch 1")
    generations = [] if path == "resume" else [1]
    with monkeypatch.context() as patch:
        fail_cost_writes(patch)
        first = launch(run_dir, first_llm, generations, meta_rec_interval=interval)

    # As on main, the state in memory takes the update after step 3, and the
    # cost total counts its cost.
    assert first_llm.batches == [[0, 1]]
    paid = update_costs(first_llm.batches)
    assert first.final_api_cost == pytest.approx(paid)
    if path == "resume":
        assert first.first_recommendation == "Recommendation of launch 1 update 1."
        assert first.api_cost == pytest.approx(paid)
    # The database holds no cost of the update, thus the state file does not
    # hold the update. Each program of the update waits one time.
    assert stored_meta_costs(run_dir) == 0
    state = read_state(run_dir)
    assert state["total_programs_meta_processed"] == 0
    assert state["meta_recommendations"] is None
    assert waiting_generations(state) == [0, 1]

    # A resume, then a second resume.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=interval)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=interval)

    # The resume makes the update again, and it writes the cost of that update.
    assert second_llm.batches == [[0, 1]]
    assert third_llm.batches == []
    # The cost total of each later launch holds each update whose cost the
    # database holds. The total of the resume is the total at its first
    # proposal.
    paid_at_the_first_proposal = update_costs(second_llm.batches[: second.updates])
    assert second.api_cost == pytest.approx(paid_at_the_first_proposal)
    assert third.api_cost == pytest.approx(update_costs(second_llm.batches))
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert meta_files(run_dir) == ["meta_2.txt"]


def test_failed_cost_write_of_a_resume_leaves_the_next_update_to_the_state_file(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    # A launch with no meta summary gives the 2 programs of the update.
    launch(run_dir, None, generations=[1])
    # Each try of the cost write of the update of the resume fails. Then the
    # job of generation 3 makes the next update, whose cost write commits.
    meta_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        fail_the_first_cost_writes(patch, async_summarizer.META_COST_WRITE_TRIES)
        launch(run_dir, meta_llm, generations=[2, 3], meta_rec_interval=2)

    # The updates are the updates of main.
    assert meta_llm.batches == [[0, 1], [2, 3]]
    assert meta_files(run_dir) == ["meta_2.txt", "meta_4.txt"]
    # The state file holds the update whose cost the database holds, and the
    # programs of the other update wait.
    assert stored_meta_costs(run_dir) == pytest.approx(
        update_costs(meta_llm.batches[1:])
    )
    state = read_state(run_dir)
    assert state["meta_recommendations_history"] == [
        "1. Recommendation of launch 1 update 2."
    ]
    assert summary_labels(state["meta_summary"]) == [(2, 2), (3, 3)]
    assert waiting_generations(state) == [0, 1]

    # The next resume makes the update of generations 0 and 1 again.
    second_llm = StubMetaLLM("launch 2")
    launch(run_dir, second_llm, generations=[], meta_rec_interval=2)
    assert second_llm.batches == [[0, 1]]
    assert "Insights of launch 1 update 2." in second_llm.step2_messages[0]
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )


def test_each_save_holds_the_updates_whose_cost_the_database_holds(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    # At each save of the state file: the count of the updates of the state,
    # and the count of the updates whose cost the database holds.
    counts = []
    save = MetaSummarizer.save_meta_state

    def save_then_count(self, filepath):
        save(self, filepath)
        if Path(filepath) == run_dir / STATE_FILE:
            history = read_state(run_dir)["meta_recommendations_history"]
            counts.append((len(history), len(stored_update_ids(run_dir))))

    monkeypatch.setattr(MetaSummarizer, "save_meta_state", save_then_count)

    # The update of the initial program. The 2 next updates give no result.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1], 1)
    # The update of the resume, then the update of a completed job.
    launch(run_dir, StubMetaLLM("launch 2"), [2], 1)
    # The final summary.
    launch(run_dir, StubMetaLLM("launch 3"), [3], 5)

    # Each save holds each update whose cost the database holds, and no other.
    assert counts
    assert all(updates == stored for updates, stored in counts)
    assert counts[-1] == (4, 4)
    assert meta_files(run_dir) == [
        "meta_1.txt",
        "meta_2.txt",
        "meta_3.txt",
        "meta_4.txt",
    ]


@pytest.mark.parametrize("budget_used", [False, True])
def test_resume_with_no_later_update_makes_the_update_with_no_best_program(
    tmp_path, caplog, budget_used
):
    run_dir = tmp_path / "run"
    # An older results directory: 3 programs that are not correct, and no
    # state file. Thus the database has no best program.
    launch(run_dir, None, generations=[1, 2], nan_generations={0, 1, 2})
    assert read_state(run_dir) is None
    # Each resume has no generation left, or generations left and a cost
    # budget that the stored costs reach.
    num_generations, max_api_costs = (10, 0.0) if budget_used else (3, None)

    # A resume with fewer waiting programs than the interval. Then a second
    # resume and a third resume.
    llms = [StubMetaLLM(f"launch {number}") for number in (1, 2, 3)]
    totals: list[list[float]] = [[], [], []]
    for meta_llm, launch_totals in zip(llms, totals):
        prompts = search(
            run_dir,
            meta_llm,
            5,
            num_generations,
            totals=launch_totals,
            max_api_costs=max_api_costs,
        )
        assert prompts == []

    # No proposal comes, and with no best program, no final summary comes.
    # Thus the first resume summarizes the 3 programs whatever the interval,
    # and it writes the cost to a waiting program. The later resumes make no
    # update.
    assert [meta_llm.batches for meta_llm in llms] == [[[0, 1, 2]], [], []]
    paid = update_costs(llms[0].batches)
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    assert totals == [[pytest.approx(paid)]] * 3
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2)]
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert meta_files(run_dir) == ["meta_3.txt"]
    # No launch logs an error.
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def commit_then_fail_the_first_cost_write(patch) -> list:
    """Make the first write of a meta cost commit, then fail.

    A cut of the connection after the commit does that. Returns the list of
    the arguments of each write.
    """
    add_meta_cost = ProgramDatabase.add_meta_cost
    calls = []

    def write_then_fail(db, *args):
        calls.append(args)
        stored = add_meta_cost(db, *args)
        if len(calls) == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return stored

    patch.setattr(ProgramDatabase, "add_meta_cost", write_then_fail)
    return calls


def test_cost_write_that_commits_then_fails_adds_its_cost_one_time(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        calls = commit_then_fail_the_first_cost_write(patch)
        launch(run_dir, meta_llm, [1], meta_rec_interval=2)

    # The second try has the same update id, thus the database adds the cost
    # one time, and the state file takes the update.
    assert meta_llm.batches == [[0, 1]]
    assert len(calls) == 2
    assert calls[0] == calls[1]
    paid = update_costs(meta_llm.batches)
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    assert len(stored_update_ids(run_dir)) == 1
    state = read_state(run_dir)
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 1."
    assert waiting_generations(state) == []
    # The cost total of the next launch holds the cost one time.
    next_launch = launch(run_dir, StubMetaLLM("launch 2"), [], meta_rec_interval=2)
    assert next_launch.updates == 0
    assert next_launch.api_cost == pytest.approx(paid)


def add_cost_to_each_program(run_dir: Path, meta_cost: float) -> None:
    """Add a meta cost to each evaluated program, as an other cost write does."""
    program_ids = list(evaluated_program_ids(run_dir))
    conn = sqlite3.connect(run_dir / "programs.sqlite", timeout=60)
    try:
        for program_id in program_ids:
            (metadata_text,) = conn.execute(
                "SELECT metadata FROM programs WHERE id = ?", (program_id,)
            ).fetchone()
            metadata = json.loads(metadata_text)
            metadata["meta_cost"] = (metadata.get("meta_cost") or 0.0) + meta_cost
            conn.execute(
                "UPDATE programs SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), program_id),
            )
        conn.commit()
    finally:
        conn.close()


class OtherCostWriteMetaLLM(StubMetaLLM):
    """Answer as StubMetaLLM. Before step 3 of update 1, an other cost write
    adds 0.5 to the meta cost of each evaluated program of the database."""

    def __init__(self, name: str, run_dir: Path):
        super().__init__(name)
        self.run_dir = run_dir
        self.programs_at_the_other_write = 0

    async def query(self, msg, system_msg):
        if system_msg == META_STEP3_SYSTEM_MSG and not self.step3_messages:
            self.programs_at_the_other_write = len(evaluated_program_ids(self.run_dir))
            add_cost_to_each_program(self.run_dir, 0.5)
        return await super().query(msg, system_msg)


# The interval of a launch whose first meta update is on each path of the
# runner, and the generations of the jobs of that launch.
COST_WRITE_PATHS = {
    "completed job": (2, [1]),
    "initial program": (1, []),
    "resume": (2, []),
    "final summary": (5, [1]),
}


@pytest.mark.parametrize("path", COST_WRITE_PATHS)
def test_cost_write_of_each_update_adds_to_the_cost_in_the_database(tmp_path, path):
    run_dir = tmp_path / "run"
    interval, generations = COST_WRITE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # An other cost write commits after the update reads its program, and
    # before the update writes its cost.
    meta_llm = OtherCostWriteMetaLLM("launch 1", run_dir)

    launch(run_dir, meta_llm, generations, meta_rec_interval=interval)

    # The database holds the cost of the 2 writes. The later write of the full
    # metadata of the program keeps them.
    assert len(meta_llm.batches) == 1
    other_costs = 0.5 * meta_llm.programs_at_the_other_write
    assert other_costs > 0
    assert stored_meta_costs(run_dir) == pytest.approx(
        other_costs + update_costs(meta_llm.batches)
    )


@pytest.mark.parametrize("path", COST_WRITE_PATHS)
def test_failed_cost_write_is_written_again_with_the_same_update_id(
    tmp_path, monkeypatch, path
):
    run_dir = tmp_path / "run"
    interval, generations = COST_WRITE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # The first cost write of launch 1 fails. The later writes succeed.
    first_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        calls = fail_the_first_cost_writes(patch, 1)
        first = launch(run_dir, first_llm, generations, meta_rec_interval=interval)

    # The update writes its cost again with the same update id and no new LLM
    # query. Then the state file takes it.
    assert len(first_llm.batches) == 1
    assert len(calls) == 2
    assert calls[0] == calls[1]
    paid = update_costs(first_llm.batches)
    assert stored_meta_costs(run_dir) == pytest.approx(paid)
    state = read_state(run_dir)
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 1."
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    if path in ("initial program", "resume"):
        # As on main, the update comes before the first proposal, thus that
        # proposal gets its recommendation, and the cost total holds its cost.
        assert first.first_recommendation == "Recommendation of launch 1 update 1."
        assert first.api_cost == pytest.approx(paid)

    # The cost total of the next launch holds the cost of the update one time.
    # With generation 0 only, that launch also makes its own update.
    second_llm = StubMetaLLM("launch 2")
    second = launch(run_dir, second_llm, [], meta_rec_interval=interval)
    assert second.api_cost == pytest.approx(
        paid + update_costs(second_llm.batches[: second.updates])
    )


def test_full_metadata_write_keeps_the_meta_costs_of_the_database(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    db = ProgramDatabase(config=config, embedding_model="")
    program = make_program("gen-1", generation=1, score=1.0)
    program.metadata = {"patch_name": "patch_1", "max_similarity": float("nan")}
    db.add(program)
    # The side effects of a job take the metadata of their program before the
    # cost write of the meta update. Then they write that full metadata.
    full_metadata = dict(program.metadata)

    async def write():
        async_db = AsyncProgramDatabase(db)
        try:
            await async_db.add_meta_cost_async("gen-1", 0.04, "update-1")
            await async_db.update_program_metadata_async(
                "gen-1", {**full_metadata, "postprocess_side_effects_applied": True}
            )
            # A meta cost in the metadata of a write does not go into the
            # database: only the cost write of a meta update changes it.
            await async_db.update_program_metadata_async(
                "gen-1",
                {**full_metadata, "meta_cost": 9.0, "meta_cost_update_ids": ["x"]},
            )
        finally:
            await async_db.close_async()

    asyncio.run(write())

    metadata = db.get("gen-1").metadata
    assert metadata["meta_cost"] == pytest.approx(0.04)
    assert metadata["meta_cost_update_ids"] == ["update-1"]
    assert metadata["patch_name"] == "patch_1"
    assert math.isnan(metadata["max_similarity"])
    # A program with no meta cost gets none.
    db.add(make_program("gen-2", generation=2, score=1.0))
    asyncio.run(update_metadata(db, "gen-2", {"meta_cost": 9.0}))
    assert "meta_cost" not in db.get("gen-2").metadata
    # As before, the write replaces stored metadata that is not JSON.
    db.cursor.execute("UPDATE programs SET metadata = 'not JSON' WHERE id = 'gen-2'")
    db.conn.commit()
    asyncio.run(update_metadata(db, "gen-2", {"patch_name": "patch_2"}))
    assert db.get("gen-2").metadata == {"patch_name": "patch_2"}
    db.close()


def test_metadata_write_of_the_runner_keeps_the_meta_costs_of_the_database(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    meta_llm = StubMetaLLM("launch 1")
    with monkeypatch.context() as patch:
        # With no metadata write in the async database, the runner writes the
        # full metadata of a program itself.
        patch.delattr(AsyncProgramDatabase, "update_program_metadata_async")
        fail_cost_writes(patch)
        run = record_wandb(patch)
        launch(run_dir, meta_llm, [1], meta_rec_interval=2)

    # As on main, the program of the update holds the cost of the update, thus
    # W&B gets it. Each cost write fails: the metadata write of that program
    # puts no meta cost into the database, and no save holds the update.
    paid = round(update_costs(meta_llm.batches), 6)
    assert wandb_costs(run)[1]["individual/cost/meta"] == paid
    config = DatabaseConfig(db_path=str(run_dir / "programs.sqlite"), num_islands=2)
    db = ProgramDatabase(config=config, embedding_model="", read_only=True)
    try:
        (program,) = [p for p in db.get_evaluated_programs() if p.generation == 1]
    finally:
        db.close()
    assert program.metadata["postprocess_side_effects_applied"]
    assert stored_meta_costs(run_dir) == 0
    state = read_state(run_dir)
    assert state["meta_recommendations_history"] == []
    assert sorted(waiting_generations(state)) == [0, 1]


async def update_metadata(db: ProgramDatabase, program_id: str, metadata: dict):
    """Write the full metadata of a program, as the runner does."""
    async_db = AsyncProgramDatabase(db)
    try:
        await async_db.update_program_metadata_async(program_id, metadata)
    finally:
        await async_db.close_async()


def record_async_databases(patch) -> list[AsyncProgramDatabase]:
    """Record each async database that a runner builds."""
    databases = []
    build = AsyncProgramDatabase.__init__

    def build_and_record(self, *args, **kwargs):
        build(self, *args, **kwargs)
        databases.append(self)

    patch.setattr(AsyncProgramDatabase, "__init__", build_and_record)
    return databases


class ProgramWriteAtStep3MetaLLM(StubMetaLLM):
    """Answer as StubMetaLLM. At step 3 of update 1, a program write starts in
    the write queue of the database of the launch. It holds the write lock of
    the database for 0.5 seconds. The stub records the thread of that write."""

    def __init__(self, name: str, run_dir: Path, databases: list):
        super().__init__(name)
        self.run_dir = run_dir
        self.databases = databases
        self.threads: dict[str, threading.Thread] = {}

    def write_a_program(self, has_the_lock: threading.Event) -> None:
        conn = sqlite3.connect(self.run_dir / "programs.sqlite", timeout=60)
        try:
            conn.execute("BEGIN IMMEDIATE")
            self.threads["program write"] = threading.current_thread()
            has_the_lock.set()
            time.sleep(0.5)
            conn.commit()
        finally:
            conn.close()

    async def query(self, msg, system_msg):
        if system_msg == META_STEP3_SYSTEM_MSG and not self.step3_messages:
            has_the_lock = threading.Event()
            write_queue = self.databases[-1].write_executor
            loop = asyncio.get_running_loop()
            loop.run_in_executor(write_queue, self.write_a_program, has_the_lock)
            for _ in range(500):
                if has_the_lock.is_set():
                    break
                await asyncio.sleep(0.01)
            assert has_the_lock.is_set()
        return await super().query(msg, system_msg)


@pytest.mark.parametrize("path", COST_WRITE_PATHS)
def test_cost_write_waits_in_the_write_queue_of_the_database(
    tmp_path, monkeypatch, path
):
    run_dir = tmp_path / "run"
    interval, generations = COST_WRITE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # A program write holds the write lock when step 3 of the update ends.
    meta_llm = ProgramWriteAtStep3MetaLLM(
        "launch 1", run_dir, record_async_databases(monkeypatch)
    )
    add_meta_cost = ProgramDatabase.add_meta_cost

    def record_the_cost_write(db, program_id, meta_cost, update_id):
        meta_llm.threads["cost write"] = threading.current_thread()
        return add_meta_cost(db, program_id, meta_cost, update_id)

    monkeypatch.setattr(ProgramDatabase, "add_meta_cost", record_the_cost_write)

    # The write queue of the database has 1 worker.
    launch(run_dir, meta_llm, generations, meta_rec_interval=interval, max_db_workers=1)

    # The cost write waits in the write queue: the one worker of the queue
    # makes it after the program write. Thus the 2 writes do not compete for
    # the write lock. The queue is open at the cost write, also at the final
    # summary, thus the database holds the cost. The write of the full
    # metadata of the program comes after the cost write, and it keeps it.
    assert len(meta_llm.batches) == 1
    assert meta_llm.threads["cost write"] is meta_llm.threads["program write"]
    assert stored_meta_costs(run_dir) == pytest.approx(update_costs(meta_llm.batches))


def test_resume_writes_the_missing_output_file_of_the_last_update(tmp_path):
    run_dir = tmp_path / "run"
    launch(run_dir, StubMetaLLM("launch 1"), generations=[1, 2])
    output_path = run_dir / "meta" / "meta_3.txt"
    output = output_path.read_text(encoding="utf-8")
    # A stop after the save of the state leaves no output file.
    output_path.unlink()

    meta_llm = StubMetaLLM("launch 2")
    launch(run_dir, meta_llm, generations=[])

    assert meta_llm.batches == []
    assert output_path.read_text(encoding="utf-8") == output


def test_failed_write_leaves_no_output_file(tmp_path):
    summarizer = AsyncMetaSummarizer(MetaSummarizer(async_mode=True))
    summarizer.sync_summarizer.total_programs_processed = 3
    # UTF-8 cannot encode this text, thus the write stops before its end.
    summarizer.sync_summarizer.meta_recommendations = "1. \ud800"

    with pytest.raises(UnicodeEncodeError):
        asyncio.run(summarizer.write_meta_output_async(str(tmp_path)))

    assert not (tmp_path / "meta" / "meta_3.txt").exists()


@pytest.mark.parametrize(
    "failure, cost",
    [("missing_answers", 0.08), ("blank_answers", 0.09), ("error_answers", 0.08)],
)
def test_program_with_a_failed_step_1_answer_waits_for_the_next_update(
    tmp_path, failure, cost
):
    run_dir = tmp_path / "run"
    # In update 1, the step 1 query for generation 1 gives no answer, a blank
    # answer, or an error.
    meta_llm = StubMetaLLM("launch 1", **{failure: {(1, 1)}})

    seen = launch(run_dir, meta_llm, generations=[1, 2, 3])

    # Update 1 gives a summary to generations 0 and 2, then recommendations.
    # Generation 1 stays in the waiting list.
    state = seen.states[1]
    assert waiting_generations(state) == [1]
    assert state["total_programs_meta_processed"] == 2
    assert summary_labels(state["meta_summary"]) == [(0, 0), (2, 2)]
    assert state["meta_recommendations"] == "1. Recommendation of launch 1 update 1."
    # The final summary tries generation 1 again, with generation 3.
    assert meta_llm.batches == [[0, 1, 2], [1, 3]]
    assert meta_files(run_dir) == ["meta_2.txt", "meta_4.txt"]
    # Each program has one summary, with its own identifier.
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert state["total_programs_meta_processed"] == 4
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert summary_labels(state["meta_summary"]) == [(0, 0), (2, 2), (1, 1), (3, 3)]
    # The cost of the search holds each answer of the 2 updates, at 0.01 for
    # each answer: 2 answers in step 1 of update 1 and 1 more for a blank
    # answer, 2 answers in step 1 of update 2, and 4 answers in steps 2 and 3.
    next_launch = launch(run_dir, StubMetaLLM("launch 2"), generations=[])
    assert next_launch.api_cost == pytest.approx(cost)


def test_update_with_no_step_1_answer_keeps_each_program(tmp_path):
    run_dir = tmp_path / "run"
    # In update 1, each step 1 answer is blank.
    meta_llm = StubMetaLLM("launch 1", blank_answers={(1, 0), (1, 1), (1, 2)})

    seen = launch(run_dir, meta_llm, generations=[1, 2, 3])

    # No program of update 1 has a summary, thus steps 2 and 3 do not run.
    assert waiting_generations(seen.memory[1]) == [0, 1, 2]
    assert seen.memory[1]["total_programs_meta_processed"] == 0
    assert seen.memory[1]["meta_summary"] is None
    assert seen.states[1] is None
    # Generation 3 starts the 3 steps again, on the 4 programs.
    assert meta_llm.batches == [[0, 1, 2], [0, 1, 2, 3]]
    assert len(meta_llm.step2_messages) == len(meta_llm.step3_messages) == 1
    assert meta_files(run_dir) == ["meta_4.txt"]
    state = read_state(run_dir)
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_program_with_an_always_blank_answer_stops_no_other_summary(tmp_path):
    run_dir = tmp_path / "run"
    # A search of 17 generations in 3 launches. In each update of each launch,
    # the step 1 answer for generation 2 is blank.
    meta_llms = [
        StubMetaLLM(f"launch {number}", blank_generations={2}) for number in (1, 2, 3)
    ]

    # Launch 1 ends at 6 generations, with its final summary.
    search(run_dir, meta_llms[0], meta_rec_interval=5, num_generations=6)
    # Launch 2 resumes. It ends as a stop of the process does, after the write
    # of generation 13 to the database and before the program goes into the
    # waiting list: the eighth call. Generations 10, 11 and 12 wait in memory
    # only.
    search(
        run_dir,
        meta_llms[1],
        meta_rec_interval=5,
        num_generations=17,
        stop_at=("add_evaluated_program", 8),
    )
    assert waiting_generations(read_state(run_dir)) == [2]
    # Launch 3 resumes, and ends at 17 generations, with its final summary.
    search(run_dir, meta_llms[2], meta_rec_interval=5, num_generations=17)

    # Each update tries generation 2 again, with the programs that came after
    # the last update.
    assert [meta_llm.batches for meta_llm in meta_llms] == [
        # The update after the job of generation 4, then the final summary.
        [[0, 1, 2, 3, 4], [2, 5]],
        # The update after the job of generation 9.
        [[2, 6, 7, 8, 9]],
        # The update of the resume, then the final summary.
        [[2, 10, 11, 12, 13], [2, 14, 15, 16]],
    ]
    # Each update gives recommendations, thus it writes its output file.
    assert [len(meta_llm.step3_messages) for meta_llm in meta_llms] == [2, 1, 2]
    assert meta_files(run_dir) == sorted(
        f"meta_{processed}.txt" for processed in (4, 5, 9, 13, 16)
    )
    state = read_state(run_dir)
    assert state["meta_recommendations"] == "1. Recommendation of launch 3 update 2."
    # Each other program has one summary, with its own identifier.
    programs = evaluated_program_ids(run_dir)
    assert sorted(programs.values()) == list(range(17))
    other_generations = [generation for generation in range(17) if generation != 2]
    assert summary_labels(state["meta_summary"]) == [
        (generation, generation) for generation in other_generations
    ]
    assert sorted(state["processed_program_ids"]) == sorted(
        program_id for program_id, generation in programs.items() if generation != 2
    )
    assert state["total_programs_meta_processed"] == 16
    # The program with the blank answer is in the waiting list at the end.
    assert waiting_generations(state) == [2]


def test_resume_update_stops_at_the_time_limit(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    # Update 2 and the final summary of launch 1 give no recommendations.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1, 2, 3, 4, 5])
    assert waiting_generations(read_state(run_dir)) == [3, 4, 5]
    monkeypatch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 0.05)

    # The update of the resume takes more time than the limit.
    meta_llm = StubMetaLLM("launch 2", slow_updates={1})
    seen = launch(run_dir, meta_llm, generations=[6])

    # The search starts with no new summary: the first proposal gets the
    # recommendation of launch 1, and the programs stay in the waiting list.
    assert seen.first_recommendation == "Recommendation of launch 1 update 1."
    assert waiting_generations(seen.start_state) == [3, 4, 5]
    assert seen.start_state["total_programs_meta_processed"] == 3
    # The next completed job makes the update, on each waiting program.
    assert meta_llm.batches == [[3, 4, 5], [3, 4, 5, 6]]
    assert meta_files(run_dir) == ["meta_3.txt", "meta_7.txt"]


def test_final_summary_stops_at_the_same_time_limit(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 0.05)

    # The final summary of the launch takes more time than the limit.
    meta_llm = StubMetaLLM("launch 1", slow_updates={1})
    launch(run_dir, meta_llm, generations=[1])

    assert meta_llm.batches == [[0, 1]]
    assert waiting_generations(read_state(run_dir)) == [0, 1]
    assert not (run_dir / "meta").exists()


def test_slow_update_of_a_resume_does_not_look_like_a_stuck_search(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    # Update 2 and the final summary of launch 1 give no recommendations.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1, 2, 3, 4, 5])
    monkeypatch.setattr(async_runner, "META_SUMMARY_TIMEOUT", 1.0)

    # The update of the resume takes the full second of its time limit.
    seen = launch(run_dir, StubMetaLLM("launch 2", slow_updates={1}), generations=[])

    # The proposal coordinator finds a stuck search from the time of the last
    # progress. That time is the end of the update, not the start of the launch.
    assert seen.progress_age < 0.5


@pytest.mark.parametrize("first_launch_has_meta_summary", [False, True])
def test_resume_summarizes_a_program_with_a_nan_score(
    tmp_path, first_launch_has_meta_summary
):
    run_dir = tmp_path / "run"
    # The evaluation of generation 1 gives a NaN score. The database and the
    # state file keep no NaN, thus a resume gets this program with no score.
    # With a meta summary, the final summary of launch 1 gives no
    # recommendations, and the state file holds generations 0 and 1.
    first_llm = (
        StubMetaLLM("launch 1", failing_updates={1})
        if first_launch_has_meta_summary
        else None
    )
    launch(run_dir, first_llm, generations=[1], nan_generations={1})

    meta_llm = StubMetaLLM("launch 2")
    launch(run_dir, meta_llm, generations=[2])

    # Generation 1 gets its summary, as it does in a fresh search.
    assert meta_llm.batches == [[0, 1, 2]]
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1), (2, 2)]
    assert "Patch Name patch_1 - Correct Program: False" in state["meta_summary"]


def no_write(self, path, content):
    raise OSError("The disk is full.")


def test_resume_starts_when_the_write_of_the_missing_output_file_fails(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    launch(run_dir, StubMetaLLM("launch 1"), generations=[1, 2])
    (run_dir / "meta" / "meta_3.txt").unlink()

    # Each write of an output file fails in launch 2.
    with monkeypatch.context() as patch:
        patch.setattr(AsyncMetaSummarizer, "_write_file", no_write)
        seen = launch(run_dir, StubMetaLLM("launch 2"), generations=[3])

    # The search starts, and the final summary of the launch takes generation 3.
    assert waiting_generations(seen.memory[0]) == [3]
    assert read_state(run_dir)["total_programs_meta_processed"] == 4
    assert meta_files(run_dir) == []

    # The next resume writes the output file of the last update.
    launch(run_dir, StubMetaLLM("launch 3"), generations=[])
    assert meta_files(run_dir) == ["meta_4.txt"]


@pytest.mark.parametrize("generations_left", [0, 1])
def test_resume_with_no_correct_program_makes_the_due_update(
    tmp_path, generations_left
):
    run_dir = tmp_path / "run"
    # An older results directory: 2 programs that are not correct, and no
    # state file. Thus the database has no best program.
    launch(run_dir, None, generations=[1], nan_generations={0, 1})
    assert read_state(run_dir) is None

    # A full interval waits at the resume.
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    num_generations = 2 + generations_left
    prompts = search(run_dir, meta_llm, 2, num_generations, totals=totals)

    # The update of the resume runs with no best program, as the update of a
    # completed job does on main. It writes its cost to a waiting program.
    assert meta_llm.batches[0] == [0, 1]
    paid = 0.01 * len(prompts) + update_costs(meta_llm.batches)
    assert stored_meta_costs(run_dir) == pytest.approx(update_costs(meta_llm.batches))
    assert totals[-1] == pytest.approx(paid)
    if generations_left:
        # The update comes before the first proposal. With no correct program,
        # that proposal fixes a program, and as on main, a fix prompt holds no
        # recommendation.
        assert totals[0] == pytest.approx(update_costs(meta_llm.batches[:1]))

    # A resume, then a second resume: no new update, and each total holds
    # each paid update one time.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=2)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=2)
    assert second_llm.batches == third_llm.batches == []
    assert second.api_cost == third.api_cost == pytest.approx(paid)
    # Each program has one summary. A fix proposal can take 2 queries of the
    # stub, thus the summary text can have the mark of the next generation.
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert [
        identifier for _, identifier in summary_labels(state["meta_summary"])
    ] == list(range(num_generations))


def test_fresh_search_with_no_best_program_leaves_the_update_to_the_resume(
    tmp_path,
):
    run_dir = tmp_path / "run"
    # A fresh search of 3 programs that are not correct, below the interval.
    first_llm = StubMetaLLM("launch 1")
    search(run_dir, first_llm, 5, 3, nan_generations={0, 1, 2})

    # As on main, the final summary needs a best program: the 3 programs wait.
    assert first_llm.batches == []
    assert len(read_state(run_dir)["unprocessed_programs"]) == 3
    assert not (run_dir / "meta").exists()

    # The resume has no generation left, thus it summarizes them.
    second_llm = StubMetaLLM("launch 2")
    search(run_dir, second_llm, 5, 3, nan_generations={0, 1, 2})
    assert len(second_llm.batches) == 1
    assert len(second_llm.batches[0]) == 3
    state = read_state(run_dir)
    assert waiting_generations(state) == []
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert meta_files(run_dir) == ["meta_3.txt"]


def test_resume_starts_when_its_update_gives_an_error(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    # Update 2 and the final summary of launch 1 give no recommendations.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1, 2, 3, 4, 5])
    monkeypatch.setattr(AsyncMetaSummarizer, "_write_file", no_write)

    meta_llm = StubMetaLLM("launch 2")
    seen = launch(run_dir, meta_llm, generations=[])

    # The write of the output file of the update of the resume fails. The
    # search starts with the recommendations of the update, and the state
    # file takes the update after its cost write.
    assert meta_llm.batches == [[3, 4, 5]]
    assert seen.first_recommendation == "Recommendation of launch 2 update 1."
    assert seen.start_state["total_programs_meta_processed"] == 6
    assert meta_files(run_dir) == ["meta_3.txt"]


# The interval and the count of generations of a search whose first meta
# update is on each path of the runner after the setup.
SEARCH_PATHS = {"completed job": (2, 2), "resume": (2, 3), "final summary": (5, 2)}


@pytest.mark.parametrize("path", SEARCH_PATHS)
def test_failed_write_of_an_output_file_keeps_each_cost_in_the_cost_total(
    tmp_path, monkeypatch, path
):
    run_dir = tmp_path / "run"
    interval, num_generations = SEARCH_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # Each write of an output file fails in launch 1.
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    with monkeypatch.context() as patch:
        patch.setattr(AsyncMetaSummarizer, "_write_file", no_write)
        prompts = search(run_dir, meta_llm, interval, num_generations, totals=totals)

    # The runner logs the error of the output file of an update, and the cost
    # total then counts the cost of the update. On main, that error made the
    # cost total skip the update.
    paid = 0.01 * len(prompts) + update_costs(meta_llm.batches)
    assert totals[-1] == pytest.approx(paid)
    if path == "resume":
        # The update of the resume comes before the first proposal.
        assert "Recommendation of launch 1 update 1." in prompts[0]
        assert totals[0] == pytest.approx(update_costs(meta_llm.batches[:1]))
    state = read_state(run_dir)
    assert len(state["meta_recommendations_history"]) == len(meta_llm.batches)
    assert meta_files(run_dir) == []

    # A resume, then a second resume: no new update. Each total holds each
    # paid update one time, and the resume writes the missing output file.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=interval)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=interval)
    assert second_llm.batches == third_llm.batches == []
    assert second.api_cost == third.api_cost == pytest.approx(paid)
    state = read_state(run_dir)
    processed = state["total_programs_meta_processed"]
    assert meta_files(run_dir) == [f"meta_{processed}.txt"]
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )


def fail_the_first_output_write(patch) -> None:
    """Make the first write of an output file fail, and the later succeed."""
    write_file = AsyncMetaSummarizer._write_file
    calls = itertools.count(1)

    def fail_then_write(self, path, content):
        if next(calls) == 1:
            raise OSError("The disk is full.")
        write_file(self, path, content)

    patch.setattr(AsyncMetaSummarizer, "_write_file", fail_then_write)


def test_failed_write_of_the_output_file_of_the_setup_does_not_stop_the_search(
    tmp_path, monkeypatch, caplog
):
    run_dir = tmp_path / "run"
    # The write of the output file of the update of the initial program fails.
    meta_llm = StubMetaLLM("launch 1")
    totals: list[float] = []
    with monkeypatch.context() as patch:
        fail_the_first_output_write(patch)
        prompts = search(run_dir, meta_llm, 1, num_generations=2, totals=totals)

    # The runner logs the error, and the search continues: the first proposal
    # gets the recommendation of the update, and the cost total holds its
    # cost. On main, that error stopped the setup.
    assert any(
        record.levelno == logging.ERROR and "meta output file" in record.getMessage()
        for record in caplog.records
    )
    assert meta_llm.batches == [[0], [1]]
    assert "Recommendation of launch 1 update 1." in prompts[0]
    assert totals[0] == pytest.approx(update_costs(meta_llm.batches[:1]))
    paid = 0.01 * len(prompts) + update_costs(meta_llm.batches)
    assert totals[-1] == pytest.approx(paid)
    assert stored_meta_costs(run_dir) == pytest.approx(update_costs(meta_llm.batches))
    # Only the output file of the update of the setup is missing.
    assert meta_files(run_dir) == ["meta_2.txt"]

    # A resume, then a second resume: no new update, and each total holds
    # each paid update one time.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=1)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=1)
    assert second_llm.batches == third_llm.batches == []
    assert second.api_cost == third.api_cost == pytest.approx(paid)
    state = read_state(run_dir)
    assert summary_labels(state["meta_summary"]) == [(0, 0), (1, 1)]
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )


def make_program(program_id: str, generation: int, score: float) -> Program:
    return Program(
        id=program_id,
        code=f"value = {generation}",
        generation=generation,
        correct=True,
        combined_score=score,
    )


def test_get_evaluated_programs_orders_by_generation_and_skips_copies(tmp_path):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=2,
        enable_dynamic_islands=True,
        stagnation_threshold=2,
    )
    db = ProgramDatabase(config=config, embedding_model="")
    db.add(make_program("gen-0", generation=0, score=1.0))
    # No generation is better than generation 0, thus the database makes a new
    # island from a copy of the initial program.
    for generation in (2, 1, 3):
        db.add(make_program(f"gen-{generation}", generation, score=0.5))

    copies = [
        program
        for program in db.get_all_programs()
        if program.id not in {"gen-0", "gen-1", "gen-2", "gen-3"}
    ]
    assert any(program.metadata.get("_is_island_copy") for program in copies)
    assert any(program.metadata.get("_spawned_island") for program in copies)

    evaluated = db.get_evaluated_programs()

    assert [program.id for program in evaluated] == ["gen-0", "gen-1", "gen-2", "gen-3"]
    db.close()


def test_meta_costs_of_the_database_skip_the_island_copies(tmp_path):
    config = DatabaseConfig(
        db_path=str(tmp_path / "programs.sqlite"),
        num_islands=2,
        enable_dynamic_islands=True,
        stagnation_threshold=2,
    )
    db = ProgramDatabase(config=config, embedding_model="")
    initial_program = make_program("gen-0", generation=0, score=1.0)
    initial_program.metadata = {"meta_cost": 0.03}
    db.add(initial_program)
    # No generation is better than generation 0, thus the database makes a new
    # island from a copy of the initial program.
    for generation in (1, 2, 3):
        db.add(make_program(f"gen-{generation}", generation, score=0.5))
    # Each copy of the initial program repeats its meta cost.
    stored = [program.metadata.get("meta_cost") for program in db.get_all_programs()]
    assert sum(cost or 0.0 for cost in stored) > 0.03

    runner = make_runner(tmp_path / "run", StubMetaLLM("launch 1"), meta_rec_interval=1)
    runner.db = db

    assert runner._get_meta_costs() == pytest.approx(0.03)
    db.close()


def test_each_island_copy_gives_its_source_program(tmp_path):
    # The first program gets a copy on each other island.
    config = DatabaseConfig(db_path=str(tmp_path / "first.sqlite"), num_islands=3)
    db = ProgramDatabase(config=config, embedding_model="")
    db.add(make_program("gen-0", generation=0, score=1.0))
    copies = [p.id for p in db.get_all_programs() if p.id != "gen-0"]
    assert db.island_manager.copy_sources == {copy_id: "gen-0" for copy_id in copies}
    assert len(copies) == 2
    db.close()

    # Generation 1 is no better than generation 0, thus a new island gets a
    # copy of the initial program and of its best correct child. The
    # maintenance on the write queue gives the copies that it made.
    config = DatabaseConfig(
        db_path=str(tmp_path / "spawn.sqlite"),
        num_islands=1,
        enable_dynamic_islands=True,
        stagnation_threshold=1,
        island_spawn_subtree_size=2,
    )
    db = ProgramDatabase(config=config, embedding_model="")
    db.add(make_program("gen-0", generation=0, score=1.0))
    child = make_program("gen-1", generation=1, score=0.5)
    child.parent_id = "gen-0"

    async def add_then_maintain():
        async_db = AsyncProgramDatabase(db)
        try:
            await async_db.add_program_async(child, defer_maintenance=True)
            return await async_db.run_program_maintenance_async(child)
        finally:
            await async_db.close_async()

    copy_sources = asyncio.run(add_then_maintain())

    assert copy_sources == {
        p.id: p.metadata["_spawned_from_program_id"]
        for p in db.get_all_programs()
        if p.metadata.get("_spawned_island")
    }
    assert sorted(copy_sources.values()) == ["gen-0", "gen-1"]
    db.close()


def test_get_evaluated_programs_skips_the_failed_proposals(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    db = ProgramDatabase(config=config, embedding_model="")
    db.add(make_program("gen-0", generation=0, score=1.0))
    # The row of a proposal that failed before its evaluation.
    failed_proposal = make_program("gen-1", generation=1, score=0.0)
    failed_proposal.correct = False
    failed_proposal.metadata = {"node_kind": "failed_proposal"}
    db.add(failed_proposal)
    db.add(make_program("gen-2", generation=2, score=0.5))

    evaluated = db.get_evaluated_programs()

    assert [program.id for program in evaluated] == ["gen-0", "gen-2"]
    db.close()


def test_add_meta_cost_adds_to_the_cost_in_the_database(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    db = ProgramDatabase(config=config, embedding_model="")
    program = make_program("gen-1", generation=1, score=1.0)
    program.metadata = {
        "patch_name": "patch_1",
        "api_costs": 0.5,
        "max_similarity": float("nan"),
    }
    db.add(program)

    # The first write adds its cost to 0, and the next write adds to the sum.
    first = db.add_meta_cost("gen-1", 0.04, "update-1")
    assert first["meta_cost"] == pytest.approx(0.04)
    stored = db.add_meta_cost("gen-1", 0.08, "update-2")
    assert stored["meta_cost"] == pytest.approx(0.12)
    assert stored["meta_cost_update_ids"] == ["update-1", "update-2"]
    # A second write of an update adds nothing: the next try of a write that
    # commits, and then fails.
    assert db.add_meta_cost("gen-1", 0.04, "update-1") == stored

    # The writes keep each other field of the metadata, also a NaN.
    metadata = db.get("gen-1").metadata
    assert metadata["meta_cost"] == pytest.approx(0.12)
    assert metadata["meta_cost_update_ids"] == ["update-1", "update-2"]
    assert metadata["patch_name"] == "patch_1"
    assert metadata["api_costs"] == 0.5
    assert math.isnan(metadata["max_similarity"])
    # A program that the database does not hold gets no cost. An add that
    # stops before its commit releases the write lock.
    assert db.add_meta_cost("gen-2", 0.04, "update-3") is None
    assert not db.conn.in_transaction
    db.cursor.execute("UPDATE programs SET metadata = 'not JSON' WHERE id = 'gen-1'")
    db.conn.commit()
    with pytest.raises(json.JSONDecodeError):
        db.add_meta_cost("gen-1", 0.04, "update-3")
    assert not db.conn.in_transaction
    db.close()


def test_meta_cost_with_live_costs_counts_each_cost_one_time():
    # The database holds the cost of update 1. The cost write of update 2
    # failed, thus only the launch has that cost.
    stored = {"meta_cost": 0.04, "meta_cost_update_ids": ["update-1"]}
    live_costs = {"update-1": 0.04, "update-2": 0.08}

    assert meta_cost_with_live_costs(stored, live_costs) == pytest.approx(0.12)
    assert meta_cost_with_live_costs(stored, {}) == pytest.approx(0.04)
    assert meta_cost_with_live_costs({}, live_costs) == pytest.approx(0.12)


def test_two_cost_writes_at_the_same_time_keep_each_cost(tmp_path):
    config = DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    db = ProgramDatabase(config=config, embedding_model="")
    db.add(make_program("gen-1", generation=1, score=1.0))

    def second_cost_write():
        second_db = ProgramDatabase(config=config, embedding_model="")
        try:
            second_db.add_meta_cost("gen-1", 0.08, "update-2")
        finally:
            second_db.close()

    second_write = threading.Thread(target=second_cost_write)

    def start_the_second_write(statement):
        # The first write read the metadata, and it starts its update. The
        # second write starts now, and it gets 1 second to commit.
        if "UPDATE programs SET metadata" in statement and not second_write.ident:
            second_write.start()
            second_write.join(timeout=1.0)

    db.conn.set_trace_callback(start_the_second_write)
    first = db.add_meta_cost("gen-1", 0.04, "update-1")
    db.conn.set_trace_callback(None)
    assert first["meta_cost"] == pytest.approx(0.04)
    second_write.join()

    # The first write holds the write lock from its read to its commit. Thus
    # the second write commits after it, and it adds to its cost.
    assert db.get("gen-1").metadata["meta_cost"] == pytest.approx(0.12)
    db.close()
