"""The copy check of a json task in the async runner, with a scripted LLM."""

import asyncio
import errno
import logging
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinka.core import EvolutionConfig, async_runner
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.sampler import PromptSampler
from shinka.database import DatabaseConfig, Program
from shinka.launch import JobScheduler, LocalJobConfig, local
from shinka.pricing.catalog import PricingConfig, PricingMode, refresh_model_catalog
from shinka.utils.json_config import config_key, load_canonicalizer
from shinka.utils.languages import get_language_extension

PARENT_CODE = """{
    "data": {"freq": "240s"},
// EVOLVE-BLOCK-START
    "train": {"lr": 0.001, "seed": 16777217}
// EVOLVE-BLOCK-END
}
"""
NOISE_COPY = PARENT_CODE.replace('"lr": 0.001', '"lr": 0.0010000000474974513')
NEW_CONFIG = PARENT_CODE.replace('"lr": 0.001', '"lr": 0.003')
PYTHON_CODE = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
# A spec with its 3 sections in the evolve block: a full rewrite can change
# each section
SPEC = """{
// EVOLVE-BLOCK-START
    "data_configuration": {"freq": "240s"},
    "model_configuration": {"T": 32, "leverage": 1},
    "train_configuration": {"lr": 0.001}
// EVOLVE-BLOCK-END
}
"""
SPEC_NO_LEVERAGE = SPEC.replace(', "leverage": 1', "")
SPEC_NEW_LR = SPEC.replace('"lr": 0.001', '"lr": 0.003')
SPEC_NEW_DATA = SPEC.replace('"240s"', '"120s"')
SPEC_REFUSED = SPEC.replace('"T": 32', '"T": "32"')


class ScriptedLLM:
    """An LLM stub: each query gives the next config as a full rewrite."""

    def __init__(self, configs, language="json"):
        self.configs = list(configs)
        self.language = language
        self.prompts = []

    def get_kwargs(self, model_sample_probs=None):
        return {"model_name": "scripted"}

    async def query(self, msg, system_msg, **kwargs):
        self.prompts.append(msg)
        await asyncio.sleep(0)
        content = (
            "<NAME>\nscripted\n</NAME>\n<DESCRIPTION>\nscripted\n</DESCRIPTION>\n"
            f"<CODE>\n```{self.language}\n{self.configs.pop(0)}```\n</CODE>\n"
        )
        return SimpleNamespace(
            content=content, cost=0.01, new_msg_history=[], to_dict=lambda: {}
        )


class SlotPool:
    in_use = 0

    async def acquire(self):
        self.in_use += 1
        return 0

    async def release(self, worker_id):
        self.in_use -= 1


class Scheduler:
    def __init__(self):
        self.submitted = []

    async def submit_async_nonblocking(self, exec_fname, results_dir):
        self.submitted.append(Path(exec_fname).read_text("utf-8"))
        return f"job-{len(self.submitted)}"


def _build_runner(
    tmp_path,
    llm,
    language="json",
    max_novelty_attempts=3,
    parent_code=None,
    canonicalizer=None,
):
    if parent_code is None:
        parent_code = PARENT_CODE if language == "json" else PYTHON_CODE
    parent = Program(id="parent-1", code=parent_code, language=language, correct=True)
    runner = object.__new__(ShinkaEvolveRunner)
    runner.evo_config = EvolutionConfig(
        language=language,
        patch_types=["full"],
        patch_type_probs=[1.0],
        max_novelty_attempts=max_novelty_attempts,
        max_patch_resamples=1,
        max_patch_attempts=1,
    )
    runner.db_config = DatabaseConfig()
    runner.db = SimpleNamespace(get_all_programs=lambda: [parent])
    runner.verbose = False
    runner.llm_selection = None
    runner.llm = llm
    runner.embedding_client = None
    runner.novelty_judge = None
    runner.results_dir = str(tmp_path)
    runner.lang_ext = get_language_extension(language)
    runner.prompt_sampler = PromptSampler(
        language=language, patch_types=["full"], patch_type_probs=[1.0]
    )
    runner._get_current_system_prompt = lambda: (None, None)
    runner.total_api_cost = 0.0
    runner.completed_proposal_costs = []
    runner.avg_proposal_cost = 0.0
    runner.running_jobs = []
    runner.submitted_jobs = {}
    runner.active_proposal_tasks = {}
    runner.max_evaluation_jobs = 2
    runner.max_proposal_jobs = 2
    runner.slot_available = asyncio.Event()
    runner.sampling_slot_pool = SlotPool()
    runner.evaluation_slot_pool = SlotPool()
    runner.scheduler = Scheduler()
    runner.samples = []
    runner.failures = []

    async def sample_with_fix_mode_async(**kwargs):
        runner.samples.append(kwargs["novelty_attempt"])
        return parent, [], [], False

    async def save_patch_attempt(**kwargs):
        return None

    async def record_failure(**kwargs):
        runner.failures.append(kwargs)

    runner.async_db = SimpleNamespace(
        sample_with_fix_mode_async=sample_with_fix_mode_async
    )
    runner._save_patch_attempt_async = save_patch_attempt
    runner._record_terminal_failed_proposal = record_failure
    runner.json_canonicalizer = canonicalizer
    if language == "json":
        runner._load_json_config_keys()
    return runner


async def _propose(runner, generation):
    gen_dir = Path(runner.results_dir) / f"gen_{generation}"
    return await runner._generate_evolved_proposal(
        generation=generation,
        task_id=f"task-{generation}",
        exec_fname=str(gen_dir / f"main.{runner.lang_ext}"),
        results_dir=str(gen_dir / "results"),
        meta_recs=None,
        meta_summary=None,
        meta_scratch=None,
        proposal_started_at=time.time(),
        sampling_worker_id=None,
        active_proposals_at_start=1,
    )


def _copy_check_lines(caplog, word):
    return [
        record.getMessage()
        for record in caplog.records
        if "JSON COPY CHECK" in record.getMessage() and word in record.getMessage()
    ]


def test_the_key_set_holds_each_parsed_config_of_the_database(tmp_path):
    runner = _build_runner(tmp_path, ScriptedLLM([]))
    island_copy = Program(id="copy-1", code=PARENT_CODE, language="json")
    broken = Program(id="broken-1", code='{"lr": ', language="json")
    other = Program(id="other-1", code=NEW_CONFIG, language="json")
    runner.db = SimpleNamespace(
        get_all_programs=lambda: [
            Program(id="parent-1", code=PARENT_CODE, language="json"),
            island_copy,
            broken,
            other,
        ]
    )
    runner._load_json_config_keys()
    assert runner.json_config_keys == {
        config_key(PARENT_CODE): "program parent-1",
        config_key(NEW_CONFIG): "program other-1",
    }


def test_two_copies_go_to_the_next_attempts_and_a_new_config_goes_through(
    tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="shinka.core.async_runner")
    runner = _build_runner(tmp_path, ScriptedLLM([PARENT_CODE, NOISE_COPY, NEW_CONFIG]))

    job = asyncio.run(_propose(runner, generation=1))

    assert job is not None
    assert runner.samples == [1, 2, 3]
    assert len(runner.llm.prompts) == 3
    assert [config_key(code) for code in runner.scheduler.submitted] == [
        config_key(NEW_CONFIG)
    ]
    assert runner.failures == []
    rejections = _copy_check_lines(caplog, "rejected")
    assert len(rejections) == 2
    assert all("program parent-1" in line for line in rejections)
    assert len(_copy_check_lines(caplog, "accepted")) == 1
    assert runner.json_config_keys[config_key(NEW_CONFIG)] == "gen 1 (in flight)"


def test_three_copies_fail_the_round(tmp_path):
    runner = _build_runner(tmp_path, ScriptedLLM([PARENT_CODE, NOISE_COPY, PARENT_CODE]))

    job = asyncio.run(_propose(runner, generation=1))

    assert job is None
    assert runner.samples == [1, 2, 3]
    assert runner.scheduler.submitted == []
    assert len(runner.failures) == 1
    assert runner.failures[0]["failure_stage"] == "novelty"
    assert "program parent-1" in runner.failures[0]["failure_reason"]


def test_two_proposals_in_flight_with_the_same_config_give_one_job(tmp_path):
    runner = _build_runner(
        tmp_path, ScriptedLLM([NEW_CONFIG, NEW_CONFIG]), max_novelty_attempts=1
    )

    read_file = runner._read_file_async
    readers = []
    both_read = asyncio.Event()

    async def read_file_together(path):
        # Both proposals reach the copy check in the same loop turn
        text = await read_file(path)
        readers.append(path)
        if len(readers) == 2:
            both_read.set()
        await both_read.wait()
        return text

    async def propose_both():
        return await asyncio.gather(_propose(runner, 1), _propose(runner, 2))

    runner._read_file_async = read_file_together

    jobs = asyncio.run(propose_both())

    assert sum(job is not None for job in jobs) == 1
    assert len(runner.scheduler.submitted) == 1
    assert len(runner.failures) == 1
    assert "(in flight)" in runner.failures[0]["failure_reason"]


def test_the_copy_check_replaces_the_novelty_judge_of_a_json_task(tmp_path):
    runner = _build_runner(tmp_path, ScriptedLLM([NEW_CONFIG]))

    class NoveltyJudge:
        async def should_check_novelty_async(self, *args):
            raise AssertionError("a json task does not use the novelty judge")

    async def get_code_embedding(exec_fname):
        return [0.1, 0.2], 0.0

    runner.novelty_judge = NoveltyJudge()
    runner._get_code_embedding_async = get_code_embedding
    job = asyncio.run(_propose(runner, generation=1))
    assert job is not None
    assert job.code_embedding == [0.1, 0.2]


def test_a_code_task_has_no_copy_check(tmp_path):
    runner = _build_runner(
        tmp_path, ScriptedLLM([PYTHON_CODE], language="python"), language="python"
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert runner.json_config_keys is None
    assert job is not None
    assert runner.scheduler.submitted == [PYTHON_CODE]


def test_a_stored_program_replaces_the_in_flight_label_and_keeps_its_key(
    tmp_path,
):
    runner = _build_runner(tmp_path, ScriptedLLM([]))
    runner.json_config_keys[config_key(NEW_CONFIG)] = "gen 4 (in flight)"
    runner._record_json_config_key(
        Program(id="program-4", code=NEW_CONFIG, language="json")
    )
    runner._release_json_config_key(4)
    assert runner.json_config_keys[config_key(NEW_CONFIG)] == "program program-4"


def test_a_failed_submit_frees_the_key_for_a_later_proposal(tmp_path):
    runner = _build_runner(
        tmp_path, ScriptedLLM([NEW_CONFIG, NEW_CONFIG]), max_novelty_attempts=1
    )
    submit = runner.scheduler.submit_async_nonblocking
    calls = []

    async def submit_that_fails_one_time(exec_fname, results_dir):
        calls.append(exec_fname)
        if len(calls) == 1:
            # subprocess.Popen can raise this error before the process starts
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
        return await submit(exec_fname, results_dir)

    async def propose_two_times():
        return await _propose(runner, 1), await _propose(runner, 2)

    runner.scheduler.submit_async_nonblocking = submit_that_fails_one_time

    failed_job, job = asyncio.run(propose_two_times())

    assert failed_job is None
    assert [failure["failure_stage"] for failure in runner.failures] == [
        "evaluation_submit"
    ]
    assert job is not None
    assert runner.scheduler.submitted == [NEW_CONFIG]
    assert runner.json_config_keys[config_key(NEW_CONFIG)] == "gen 2 (in flight)"


def test_a_proposal_cancelled_at_the_slot_wait_frees_its_key(tmp_path):
    runner = _build_runner(tmp_path, ScriptedLLM([NEW_CONFIG]))

    async def record_unattached_cost(cost, stage, generation):
        return None

    async def cancel_at_the_slot_wait():
        waiting = asyncio.Event()

        async def acquire_no_slot():
            waiting.set()
            await asyncio.Event().wait()

        runner.evaluation_slot_pool.acquire = acquire_no_slot
        task = asyncio.create_task(_propose(runner, generation=1))
        await waiting.wait()
        assert runner.json_config_keys[config_key(NEW_CONFIG)] == "gen 1 (in flight)"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    runner._record_unattached_cost = record_unattached_cost

    asyncio.run(cancel_at_the_slot_wait())

    assert config_key(NEW_CONFIG) not in runner.json_config_keys


def test_a_log_open_error_after_the_process_start_kills_the_job_and_frees_the_key(
    tmp_path, monkeypatch
):
    runner = _build_runner(tmp_path, ScriptedLLM([NEW_CONFIG]), max_novelty_attempts=1)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("import time\ntime.sleep(60)\n")
    runner.scheduler = JobScheduler(
        "local", LocalJobConfig(eval_program_path=str(evaluator))
    )
    # The stderr log is a directory: its open gives an error after Popen
    (tmp_path / "gen_1" / "results" / "job_log.err").mkdir(parents=True)
    processes = []

    def popen(*args, **kwargs):
        processes.append(subprocess.Popen(*args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(
        local, "subprocess", SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
    )

    job = asyncio.run(_propose(runner, generation=1))
    runner.scheduler.shutdown()
    returncode = processes[0].returncode
    processes[0].kill()
    processes[0].wait()

    assert job is None
    assert [failure["failure_stage"] for failure in runner.failures] == [
        "evaluation_submit"
    ]
    assert "Is a directory" in runner.failures[0]["failure_reason"]
    assert returncode == -signal.SIGKILL
    assert config_key(NEW_CONFIG) not in runner.json_config_keys


@pytest.fixture
def spec_key(canonicalizer_dir):
    return load_canonicalizer("spec_keys:spec_key", canonicalizer_dir)


@pytest.fixture
def strict_spec_key(canonicalizer_dir):
    return load_canonicalizer("spec_keys:strict_spec_key", canonicalizer_dir)


def test_a_canonicalizer_rejects_a_config_of_the_same_run(
    tmp_path, spec_key, caplog
):
    caplog.set_level(logging.INFO, logger="shinka.core.async_runner")
    runner = _build_runner(
        tmp_path,
        ScriptedLLM([SPEC_NO_LEVERAGE, SPEC_NEW_LR]),
        parent_code=SPEC,
        canonicalizer=spec_key,
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert job is not None
    assert runner.samples == [1, 2]
    assert [config_key(code) for code in runner.scheduler.submitted] == [
        config_key(SPEC_NEW_LR)
    ]
    assert _copy_check_lines(caplog, "rejected") == [
        "JSON COPY CHECK 1/3: rejected gen 1, same config as program parent-1"
    ]


def test_with_a_canonicalizer_a_change_of_a_data_value_is_a_new_config(
    tmp_path, spec_key
):
    runner = _build_runner(
        tmp_path,
        ScriptedLLM([SPEC_NEW_DATA]),
        max_novelty_attempts=1,
        parent_code=SPEC,
        canonicalizer=spec_key,
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert job is not None
    assert [config_key(code) for code in runner.scheduler.submitted] == [
        config_key(SPEC_NEW_DATA)
    ]


def test_without_a_canonicalizer_a_key_at_its_default_is_a_new_config(tmp_path):
    runner = _build_runner(
        tmp_path,
        ScriptedLLM([SPEC_NO_LEVERAGE]),
        max_novelty_attempts=1,
        parent_code=SPEC,
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert job is not None
    assert runner.failures == []


def test_a_config_that_the_canonicalizer_refuses_goes_to_the_next_attempt(
    tmp_path, strict_spec_key, caplog
):
    caplog.set_level(logging.INFO, logger="shinka.core.async_runner")
    runner = _build_runner(
        tmp_path,
        ScriptedLLM([SPEC_REFUSED, SPEC_NEW_LR]),
        parent_code=SPEC,
        canonicalizer=strict_spec_key,
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert job is not None
    assert runner.samples == [1, 2]
    assert [config_key(code) for code in runner.scheduler.submitted] == [
        config_key(SPEC_NEW_LR)
    ]
    assert _copy_check_lines(caplog, "rejected") == [
        "JSON COPY CHECK 1/3: rejected gen 1, the canonicalizer refuses the "
        "config: ValueError: model_configuration.T: the format needs an integer"
    ]


def test_a_round_of_refused_configs_gives_the_error_as_the_reason(
    tmp_path, strict_spec_key
):
    runner = _build_runner(
        tmp_path,
        ScriptedLLM([SPEC_REFUSED]),
        max_novelty_attempts=1,
        parent_code=SPEC,
        canonicalizer=strict_spec_key,
    )

    job = asyncio.run(_propose(runner, generation=1))

    assert job is None
    assert runner.scheduler.submitted == []
    assert [failure["failure_stage"] for failure in runner.failures] == ["novelty"]
    assert runner.failures[0]["failure_reason"] == (
        "The canonicalizer refuses the JSON config: ValueError: "
        "model_configuration.T: the format needs an integer"
    )
    assert list(runner.json_config_keys.values()) == ["program parent-1"]


def test_a_program_that_the_canonicalizer_refuses_has_no_key(
    tmp_path, strict_spec_key, caplog
):
    runner = _build_runner(
        tmp_path, ScriptedLLM([]), parent_code=SPEC, canonicalizer=strict_spec_key
    )
    runner.db = SimpleNamespace(
        get_all_programs=lambda: [
            Program(id="parent-1", code=SPEC, language="json"),
            Program(id="refused-1", code=SPEC_REFUSED, language="json"),
        ]
    )

    runner._load_json_config_keys()
    runner._record_json_config_key(
        Program(id="refused-2", code=SPEC_REFUSED, language="json")
    )

    assert runner.json_config_keys == {
        config_key(SPEC, strict_spec_key): "program parent-1"
    }
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 2
    assert "program refused-1 has no config key" in warnings[0]
    assert "program refused-2 has no config key" in warnings[1]


@pytest.fixture
def offline_runner(monkeypatch, tmp_path):
    """Build a full runner with the packaged prices and no check of the models."""
    config = PricingConfig(mode=PricingMode.OFFLINE, cache_dir=tmp_path / "prices")
    monkeypatch.setattr(
        async_runner, "refresh_model_catalog", lambda: refresh_model_catalog(config)
    )
    monkeypatch.setattr(
        async_runner, "_validate_evo_config_model_env_access", lambda evo_config: None
    )
    yield
    refresh_model_catalog(config)


def _full_runner(tmp_path, **evo_options):
    return ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            language="json",
            llm_models=["stub-model"],
            llm_dynamic_selection=None,
            meta_rec_interval=None,
            embedding_model=None,
            num_generations=1,
            results_dir=str(tmp_path / "results"),
            **evo_options,
        ),
        job_config=LocalJobConfig(),
        db_config=DatabaseConfig(),
        verbose=False,
    )


def test_the_runner_imports_the_canonicalizer_of_a_json_task(
    tmp_path, offline_runner, canonicalizer_dir
):
    runner = _full_runner(
        tmp_path,
        json_canonicalizer="spec_keys:spec_key",
        json_canonicalizer_path=canonicalizer_dir,
    )

    canonicalizer = runner.json_canonicalizer
    assert config_key(SPEC_NO_LEVERAGE, canonicalizer) == config_key(
        SPEC, canonicalizer
    )


def test_a_canonicalizer_that_does_not_import_stops_the_runner_at_its_start(
    tmp_path, offline_runner
):
    with pytest.raises(ValueError, match="'no_such_module:spec_key'"):
        _full_runner(tmp_path, json_canonicalizer="no_such_module:spec_key")
    assert not (tmp_path / "results").exists()
