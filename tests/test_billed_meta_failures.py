"""Failed prompt calls reach both live and restored budgets.

The fork tests the meta calls in tests/test_meta_state_resume.py
(test_failed_meta_update_charges_its_cost and
test_meta_cost_survives_an_output_file_error): there, each meta cost goes
through the cost write of its update (fork PR #6).
"""

import asyncio
from types import SimpleNamespace

import pytest

from shinka.core import EvolutionConfig
from shinka.core.async_runner import (
    ShinkaEvolveRunner,
)
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)


def test_failed_prompt_evolution_is_durable(tmp_path):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.total_api_cost = 0.0
        runner.prompt_api_cost = 0.0
        runner.prompt_evolution_counter = 0
        runner.completed_generations = 1
        runner.evo_config = EvolutionConfig(
            evolve_prompts=True, prompt_evolution_interval=1
        )
        runner.meta_summarizer = None
        runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: object())
        runner.prompt_db = SimpleNamespace(
            last_generation=0, get_total_evolution_costs=lambda: 0.0
        )

        async def top(count):
            return []

        async def evolve(**kwargs):
            return None, "full", 3.3

        async def record(**kwargs):
            db.record_attempt_event(**kwargs)

        runner.async_db = SimpleNamespace(
            get_top_programs_async=top, record_attempt_event_async=record
        )
        runner.prompt_evolver = SimpleNamespace(evolve=evolve)
        asyncio.run(runner._maybe_evolve_prompt())
        assert runner.total_api_cost == pytest.approx(3.3)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
    finally:
        db.close()


@pytest.mark.parametrize("persist_before_error", [False, True])
def test_prompt_insert_failure_charges_exactly_once(
    tmp_path, monkeypatch, persist_before_error
):
    db = ProgramDatabase(
        DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=1)
    )
    prompts = SystemPromptDatabase(
        SystemPromptConfig(db_path=str(tmp_path / "prompts.sqlite"))
    )
    try:
        runner = object.__new__(ShinkaEvolveRunner)
        runner.db = db
        runner.prompt_db = prompts
        runner.total_api_cost = runner.prompt_api_cost = 0.0
        runner.prompt_evolution_counter = 0
        runner.completed_generations = 1
        runner.verbose = False
        runner.evo_config = EvolutionConfig(
            evolve_prompts=True, prompt_evolution_interval=1
        )
        runner.meta_summarizer = None
        runner.prompt_sampler_evo = SimpleNamespace(sample=lambda: object())
        prompt = create_system_prompt(
            prompt_text="task",
            generation=1,
            patch_type="full",
            metadata={"llm": {"cost": 3.3}},
        )

        async def top(count):
            return []

        async def evolve(**kwargs):
            return prompt, "full", 3.3

        async def record(**kwargs):
            db.record_attempt_event(**kwargs)

        add = prompts.add

        def fail(prompt, **kwargs):
            if persist_before_error:
                add(prompt, **kwargs)
            raise OSError("prompt insert failed")

        monkeypatch.setattr(prompts, "add", fail)
        runner.async_db = SimpleNamespace(
            get_top_programs_async=top, record_attempt_event_async=record
        )
        runner.prompt_evolver = SimpleNamespace(evolve=evolve)
        asyncio.run(runner._maybe_evolve_prompt())
        assert runner.total_api_cost == pytest.approx(3.3)
        assert asyncio.run(runner._get_total_api_costs()) == pytest.approx(3.3)
    finally:
        prompts.close()
        db.close()
