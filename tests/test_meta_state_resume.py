"""
Tests that the async runner keeps the meta summary across a resume.

Each launch runs the real `run_async()` on one run directory. A stub replaces
the meta LLM and the evaluation of a job. `launch()` completes the jobs of a
launch with a stub job monitor. `search()` keeps the real job monitor and the
real proposal coordinator, with a stub in place of the proposal LLM. Thus no
test starts a search or calls an LLM.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinka.core import EvolutionConfig, ShinkaEvolveRunner, async_runner
from shinka.core.async_runner import AsyncRunningJob
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.launch import LocalJobConfig
from shinka.pricing.catalog import PricingConfig, PricingMode, refresh_model_catalog
from shinka.prompts import META_STEP1_SYSTEM_MSG, META_STEP2_SYSTEM_MSG

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
) -> ShinkaEvolveRunner:
    """Build a runner on run_dir. Without a meta LLM, the meta summary is off.

    The evaluation of each generation of nan_generations gives a NaN score.
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
    )
    runner = ShinkaEvolveRunner(
        evo_config,
        LocalJobConfig(),
        DatabaseConfig(num_islands=2),
        verbose=False,
        max_evaluation_jobs=1,
    )

    async def get_job_results(job_id, results_dir):
        generation = int(re.search(r"(\d+)$", str(job_id)).group(1))
        return NAN_RESULTS if generation in nan_generations else EVAL_RESULTS

    runner.scheduler.run = lambda exec_fname, results_dir: (EVAL_RESULTS, 0.0)
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
    runner.meta_summarizer.save_meta_state(str(memory_path))
    return json.loads(memory_path.read_text(encoding="utf-8"))


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
) -> SimpleNamespace:
    """Run one launch of a search on run_dir, from the setup to the final summary.

    The launch completes one job for each generation of the list. The result
    holds what the first proposal of the launch sees (the recommendation, the
    API cost of the search, the count of the meta updates of the launch, the
    state file, and the seconds from the last progress that the runner
    recorded). It also holds the state file and the
    state in memory after each job, and the count of processed programs that
    the state file holds at each write of an output file.
    """
    runner = make_runner(
        run_dir, meta_llm, meta_rec_interval, nan_generations=nan_generations
    )
    seen = SimpleNamespace(
        first_recommendation=None,
        api_cost=None,
        updates=None,
        start_state=None,
        progress_age=None,
        states=[],
        memory=[],
        saved_counts=[],
    )
    if runner.meta_summarizer:
        write_output = runner.meta_summarizer.write_meta_output_async

        async def record_then_write(results_dir):
            state = read_state(run_dir)
            seen.saved_counts.append(state and state["total_programs_meta_processed"])
            await write_output(results_dir)

        runner.meta_summarizer.write_meta_output_async = record_then_write

    async def complete_jobs():
        try:
            seen.api_cost = runner.total_api_cost
            seen.updates = len(meta_llm.batches) if meta_llm else 0
            seen.start_state = read_state(run_dir)
            seen.progress_age = time.time() - runner.last_progress_time
            if runner.meta_summarizer:
                seen.first_recommendation = (
                    runner.meta_summarizer.get_sampled_recommendation()
                )
            for generation in generations:
                await complete_job(runner, generation)
                seen.states.append(read_state(run_dir))
                if runner.meta_summarizer:
                    seen.memory.append(read_memory(runner))
        finally:
            runner.should_stop.set()
            runner.finalization_complete.set()

    async def no_proposals():
        return None

    runner._job_monitor_task = complete_jobs
    runner._proposal_coordinator_task = no_proposals
    asyncio.run(runner.run_async())
    assert len(seen.states) == len(generations), "a job of the launch failed"
    return seen


def search(
    run_dir: Path,
    meta_llm: StubMetaLLM,
    meta_rec_interval: int,
    num_generations: int,
    stop_at: tuple[str, int] | None = None,
) -> list[str]:
    """Run one launch with the real job monitor and the real proposal coordinator.

    stop_at gives a method of the runner or of its meta summarizer, and a call
    number. The launch then ends at that call as a stop of the process does:
    the call does not run, and the launch makes no final summary and no final
    save.

    The result is the prompt of each proposal of the launch.
    """
    runner = make_runner(run_dir, meta_llm, meta_rec_interval, num_generations)
    last_generation = max(
        (int(path.name[4:]) for path in run_dir.glob("gen_*")), default=0
    )
    runner.llm = StubProposalLLM(first_generation=last_generation + 1)

    async def submit_job(exec_fname, results_dir):
        return f"job-{Path(exec_fname).parent.name}"

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

    if stop_at:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_search())
    else:
        asyncio.run(run_search())
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


def test_launch_on_generation_0_only_continues_the_meta_state(tmp_path, caplog):
    run_dir = tmp_path / "run"
    # Launch 1 stops as a stop of the process does, after the update of
    # generation 0 and before the proposal of generation 1.
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=1,
        num_generations=2,
        stop_at=("_verify_database_ready", 1),
    )
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
    search(
        run_dir,
        StubMetaLLM("launch 1"),
        meta_rec_interval=1,
        num_generations=2,
        stop_at=("_verify_database_ready", 1),
    )

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


# The interval of a search of 2 generations whose first meta update is on
# each path of the runner.
UPDATE_PATHS = {"completed job": 2, "resume": 2, "final summary": 5}


def update_costs(batches: list[list[int]]) -> float:
    """Give the cost of meta updates: 0.01 for each answer of the 3 steps."""
    return sum(0.01 * (len(batch) + 2) for batch in batches)


@pytest.mark.parametrize("path", UPDATE_PATHS)
@pytest.mark.parametrize("stop_before", ["save_meta_state", "write_meta_output_async"])
def test_stop_between_the_writes_of_an_update_keeps_each_cost(
    tmp_path, path, stop_before
):
    run_dir = tmp_path / "run"
    interval = UPDATE_PATHS[path]
    if path == "resume":
        # A launch with no meta summary gives the 2 programs of the update.
        launch(run_dir, None, generations=[1])
    # An update writes its cost to the database, then the state, then the
    # output file. Launch 1 pays its first update. Then it stops as a stop of
    # the process does, before the second write or before the third write.
    first_llm = StubMetaLLM("launch 1")
    prompts = search(
        run_dir, first_llm, interval, num_generations=2, stop_at=(stop_before, 1)
    )
    assert first_llm.batches == [[0, 1]]
    paid = 0.01 * len(prompts) + update_costs(first_llm.batches)
    state_holds_the_update = stop_before == "write_meta_output_async"
    assert (read_state(run_dir) is not None) == state_holds_the_update
    assert not (run_dir / "meta").exists()

    # A resume, then a second resume.
    second_llm, third_llm = StubMetaLLM("launch 2"), StubMetaLLM("launch 3")
    second = launch(run_dir, second_llm, generations=[], meta_rec_interval=interval)
    third = launch(run_dir, third_llm, generations=[], meta_rec_interval=interval)

    if state_holds_the_update:
        # The resume does not make the update again.
        assert second_llm.batches == []
    else:
        # The state does not hold the update, thus the resume makes it again:
        # the search paid 2 updates.
        assert second_llm.batches == [[0, 1]]
    assert third_llm.batches == []
    # The cost total of each later launch holds each paid update one time.
    # The total of the resume is the total at its first proposal.
    paid_at_the_first_proposal = update_costs(second_llm.batches[: second.updates])
    assert second.api_cost == pytest.approx(paid + paid_at_the_first_proposal)
    paid += update_costs(second_llm.batches)
    assert third.api_cost == pytest.approx(paid)
    # The state holds each program one time, and the output file is there.
    state = read_state(run_dir)
    assert sorted(state["processed_program_ids"]) == sorted(
        evaluated_program_ids(run_dir)
    )
    assert meta_files(run_dir) == ["meta_2.txt"]


def test_state_file_holds_each_update_before_its_output_file(tmp_path):
    run_dir = tmp_path / "run"

    # The update of the initial program. The 2 next updates give no result.
    first = launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1], 1)
    # The update of the resume, then the update of a completed job.
    second = launch(run_dir, StubMetaLLM("launch 2"), [2], 1)
    # The final summary.
    third = launch(run_dir, StubMetaLLM("launch 3"), [3], 5)

    assert meta_files(run_dir) == [
        "meta_1.txt",
        "meta_2.txt",
        "meta_3.txt",
        "meta_4.txt",
    ]
    assert [first.saved_counts, second.saved_counts, third.saved_counts] == [
        [1],
        [2, 3],
        [4],
    ]


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


def test_resume_starts_when_its_update_gives_an_error(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    # Update 2 and the final summary of launch 1 give no recommendations.
    launch(run_dir, StubMetaLLM("launch 1", failing_updates={2, 3}), [1, 2, 3, 4, 5])
    monkeypatch.setattr(AsyncMetaSummarizer, "_write_file", no_write)

    meta_llm = StubMetaLLM("launch 2")
    seen = launch(run_dir, meta_llm, generations=[])

    # The update of the resume saves its state, then the write of its output
    # file fails. The search starts with the recommendations of the update.
    assert meta_llm.batches == [[3, 4, 5]]
    assert seen.first_recommendation == "Recommendation of launch 2 update 1."
    assert seen.start_state["total_programs_meta_processed"] == 6
    assert meta_files(run_dir) == ["meta_3.txt"]


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
