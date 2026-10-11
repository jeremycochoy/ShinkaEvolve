"""
Async meta summarizer for concurrent meta-analysis.
Provides non-blocking meta summarization with concurrent LLM calls.
"""

import asyncio
import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Set, Tuple
from pathlib import Path
from .summarizer import MetaSummarizer
from ..llm import AsyncLLMClient
from ..database import Program
from ..prompts import (
    construct_individual_program_msg,
    META_STEP1_SYSTEM_MSG,
    META_STEP1_USER_MSG,
    META_STEP2_SYSTEM_MSG,
    META_STEP2_USER_MSG,
    META_STEP3_SYSTEM_MSG,
    META_STEP3_USER_MSG,
)

logger = logging.getLogger(__name__)

# Tries of the cost write of a meta update, and the seconds between 2 tries
META_COST_WRITE_TRIES = 3
META_COST_WRITE_RETRY_DELAY = 1.0


@dataclass
class MetaUpdate:
    """The result and the cost of a meta update."""

    update_id: str
    programs: List[Program]
    individual_summaries: str
    global_insights: str
    recommendations: str
    cost: float


def apply_meta_update(state: MetaSummarizer, update: MetaUpdate) -> None:
    """Put the result of a meta update into a meta state."""
    # Concatenate new individual summaries to existing ones
    if state.meta_summary:
        state.meta_summary += "\n\n" + update.individual_summaries
    else:
        state.meta_summary = update.individual_summaries

    state.meta_scratch_pad = update.global_insights
    state.meta_recommendations = update.recommendations

    # Store the newly generated recommendations in history immediately
    if update.recommendations and isinstance(update.recommendations, str):
        state.meta_recommendations_history.append(update.recommendations)

    # Remove only the summarized programs from the evaluated programs list
    # A program with no step 1 answer stays there, and the next meta
    # update tries it again
    program_ids = [p.id for p in update.programs]
    state.total_programs_processed += len(program_ids)
    state.processed_program_ids.extend(program_ids)
    state.evaluated_since_last_meta = [
        p for p in state.evaluated_since_last_meta if p.id not in program_ids
    ]


def copy_meta_state(state: MetaSummarizer) -> MetaSummarizer:
    """Copy a meta state, with an empty waiting list."""
    copied = copy.copy(state)
    copied.meta_recommendations_history = list(state.meta_recommendations_history)
    copied.processed_program_ids = list(state.processed_program_ids)
    copied.evaluated_since_last_meta = []
    return copied


class AsyncMetaSummarizer:
    """Async version of MetaSummarizer for concurrent meta-analysis.

    As on main, the state in memory takes the result of a meta update after
    its step 3. Then a task writes the cost of the update to the database. A
    save of the state holds an update only when the database holds its cost.
    """

    def __init__(
        self,
        sync_summarizer: MetaSummarizer,
        async_llm_client: Optional[AsyncLLMClient] = None,
        state_path: Optional[str] = None,
    ):
        """Initialize with existing sync summarizer.

        Args:
            sync_summarizer: The synchronous MetaSummarizer instance
            async_llm_client: Optional async LLM client for meta analysis
            state_path: Optional file that gets the saved state after each
                cost write of a meta update
        """
        self.sync_summarizer = sync_summarizer
        self.async_llm_client = async_llm_client
        self.state_path = state_path
        # The updates of the state in memory whose cost the database does not
        # hold: their cost write is in progress, or it failed. In the order
        # of the updates
        self.unsaved_updates: List[MetaUpdate] = []
        # With unsaved updates, the state that a save writes: the state from
        # before the first unsaved update, and each later update whose cost
        # the database holds
        self.saved_state = copy_meta_state(sync_summarizer)
        # The task of each cost write of a meta update
        self.cost_writes: List[asyncio.Task] = []
        # The last update that the state in memory took. A caller reads it
        # when update_meta_memory_async() gives recommendations
        self.last_update: Optional[MetaUpdate] = None

    async def update_meta_memory_async(
        self,
        best_program: Optional[Program] = None,
        store_cost: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Async version of update_meta_memory.
        Perform 3-step meta-analysis and update internal state.
        A program with no step 1 answer stays in the waiting list.
        With store_cost, a task writes the cost of the update after the
        update, and the caller does not wait for it. The cost writes run one
        at a time, in the order of the updates.
        Returns tuple of (updated_recommendations, total_cost) or
        (None, 0.0) if no update occurred.
        """
        if not self.async_llm_client:
            logger.warning("No async meta LLM client configured")
            return None, 0.0

        # Use recently evaluated programs for memory scratchpad
        # Make a copy to avoid issues if the list is modified during processing
        programs_to_analyze = (
            self.sync_summarizer.evaluated_since_last_meta.copy()
            if self.sync_summarizer.evaluated_since_last_meta
            else []
        )

        if len(programs_to_analyze) == 0:
            logger.info("No programs evaluated since last meta query, skipping")
            return None, 0.0

        total_meta_cost = 0.0

        try:
            # Step 1: Create individual program summaries
            (
                summarized_programs,
                individual_summaries,
                step1_cost,
            ) = await self._step1_individual_summaries_async(programs_to_analyze)
            total_meta_cost += step1_cost
            if not individual_summaries:
                logger.error("Step 1 failed - no individual summaries generated")
                return None, total_meta_cost

            # Step 2: Generate global insights scratchpad
            global_insights, step2_cost = await self._step2_global_insights_async(
                individual_summaries, best_program
            )
            total_meta_cost += step2_cost
            if not global_insights:
                logger.error("Step 2 failed - no global insights generated")
                return None, total_meta_cost

            # Step 3: Generate recommendations based on insights
            (
                recommendations,
                step3_cost,
            ) = await self._step3_generate_recommendations_async(
                global_insights, best_program
            )
            total_meta_cost += step3_cost
            if not recommendations:
                logger.error("Step 3 failed - no recommendations generated")
                return None, total_meta_cost

            logger.info(
                f"==> Meta-analysis completed successfully with 3-step process (total cost: ${total_meta_cost:.4f})"
            )
        except Exception as e:
            logger.error(f"Failed to complete 3-step meta-analysis: {e}")
            return None, total_meta_cost

        update = MetaUpdate(
            update_id=uuid.uuid4().hex,
            programs=summarized_programs,
            individual_summaries=individual_summaries,
            global_insights=global_insights,
            recommendations=recommendations,
            cost=total_meta_cost,
        )
        # A save keeps the state from before the update until the database
        # holds the cost of the update
        if not self.unsaved_updates:
            self.saved_state = copy_meta_state(self.sync_summarizer)
        self.unsaved_updates.append(update)
        self.last_update = update

        # Update internal state of sync summarizer
        apply_meta_update(self.sync_summarizer, update)
        logger.info(
            f"Processed and cleared {len(update.programs)} programs from meta memory "
            f"(total processed: {self.sync_summarizer.total_programs_processed}, "
            f"waiting: {len(self.sync_summarizer.evaluated_since_last_meta)})"
        )
        self._queue_cost_write(update, store_cost)

        return (
            (
                self.sync_summarizer.meta_recommendations
                if isinstance(self.sync_summarizer.meta_recommendations, str)
                else None
            ),
            total_meta_cost,
        )

    def _queue_cost_write(
        self,
        update: MetaUpdate,
        store_cost: Optional[Callable[[float, str], Awaitable[None]]],
    ) -> None:
        """Start the task that writes the cost of an update to the database."""
        self.cost_writes = [task for task in self.cost_writes if not task.done()]
        previous = self.cost_writes[-1] if self.cost_writes else None
        self.cost_writes.append(
            asyncio.create_task(self._write_update_cost(update, store_cost, previous))
        )

    async def _write_update_cost(
        self,
        update: MetaUpdate,
        store_cost: Optional[Callable[[float, str], Awaitable[None]]],
        previous: Optional[asyncio.Task],
    ) -> None:
        """Write the cost of an update, then put the update into the saved state.

        The task waits for the cost write of the update before it, thus the
        saved state takes the updates in their order. When each try fails, no
        save holds the update, and a later launch makes the update again.
        """
        if previous is not None:
            await asyncio.wait([previous])
        if (
            store_cost is not None
            and update.cost > 0
            and not await self._try_cost_write(update, store_cost)
        ):
            return
        self._store_update(update)

    async def _try_cost_write(
        self,
        update: MetaUpdate,
        store_cost: Callable[[float, str], Awaitable[None]],
    ) -> bool:
        """Write the cost of an update. Return False when each try fails.

        Each try has the same update id, thus the database adds the cost one
        time, also when a try commits and then fails.
        """
        for attempt in range(1, META_COST_WRITE_TRIES + 1):
            try:
                await store_cost(update.cost, update.update_id)
                return True
            except Exception as e:
                logger.warning(
                    f"Cost write {attempt}/{META_COST_WRITE_TRIES} of meta update "
                    f"{update.update_id} failed: {e}"
                )
            if attempt < META_COST_WRITE_TRIES:
                await asyncio.sleep(META_COST_WRITE_RETRY_DELAY)
        logger.error(
            f"The database holds no cost of meta update {update.update_id}: "
            f"no save of the meta state holds the update"
        )
        return False

    def _store_update(self, update: MetaUpdate) -> None:
        """Put an update into the saved state when the database holds its cost.

        Then save the state to the state file.
        """
        self.unsaved_updates.remove(update)
        apply_meta_update(self.saved_state, update)
        if self.state_path:
            self.save_meta_state(self.state_path)

    def take_stored_updates(self, stored_update_ids: Set[str]) -> None:
        """Put each unsaved update whose cost the database holds into the saved state.

        A stop can cut a cost write after its commit. An update with no cost
        has no cost write.
        """
        for update in list(self.unsaved_updates):
            if update.cost <= 0 or update.update_id in stored_update_ids:
                self._store_update(update)

    async def wait_for_cost_writes(self, timeout: Optional[float] = None) -> None:
        """Wait for the cost writes of the meta updates, up to timeout seconds."""
        pending = [task for task in self.cost_writes if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def stop_cost_writes(self) -> None:
        """Stop the cost writes that did not end, and wait for their end."""
        pending = [task for task in self.cost_writes if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)

    async def _step1_individual_summaries_async(
        self, programs_to_analyze: List[Program]
    ) -> Tuple[List[Program], str, float]:
        """Async version of Step 1: Create individual summaries for each program using concurrent queries.

        Returns the programs that got a summary, their combined summaries and
        the cost of all queries.
        """
        if not programs_to_analyze:
            logger.warning("No programs to analyze in Step 1")
            return [], "", 0.0

        # Use one query for each program, all at the same time. Each query
        # gives its program back, thus a summary always has the identifier
        # of its program
        num_programs = len(programs_to_analyze)
        logger.info(
            f"==> Step 1 - Processing {num_programs} programs with concurrent async queries"
        )
        results = await asyncio.gather(
            *(self.summarize_program_async(p) for p in programs_to_analyze)
        )
        total_cost = sum(cost for _, _, cost in results)
        summarized = [(program, summary) for program, summary, _ in results if summary]
        logger.info(
            f"==> Step 1 - {len(summarized)}/{num_programs} "
            f"individual summaries generated (cost: ${total_cost:.4f})"
        )

        # Sort the summaries by generation, then join them with double newlines
        summaries_by_gen = sorted(summarized, key=lambda x: x[0].generation)
        final_summary = "\n\n".join(summary for _, summary in summaries_by_gen)
        return [program for program, _ in summarized], final_summary, total_cost

    async def summarize_program_async(
        self, program: Program
    ) -> Tuple[Program, Optional[str], float]:
        """Step 1 for one program: query its summary and add its identifier.

        Returns the program, its summary and the cost of the query. The
        summary is None when the answer is missing or blank, or after an error.
        """
        summary, cost = None, 0.0
        try:
            individual_program_msg = construct_individual_program_msg(
                program,
                language=self.sync_summarizer.language,
                include_text_feedback=self.sync_summarizer.use_text_feedback,
            )
            user_msg = META_STEP1_USER_MSG.replace(
                "{individual_program_msg}", individual_program_msg
            )
            response = await self.async_llm_client.query(
                msg=user_msg,
                system_msg=META_STEP1_SYSTEM_MSG,
            )
            if response is not None:
                cost = response.cost or 0.0
                if response.content and response.content.strip():
                    identifier = f"Generation {program.generation} - Patch Name {program.metadata['patch_name']} - Correct Program: {program.correct}"
                    summary = f"{response.content.strip()}\n**Program Identifier:** {identifier}"
        except Exception as e:
            logger.error(f"Step 1: Error for program {program.id}: {e}")

        if summary is None:
            logger.warning(
                f"Step 1: No summary for program {program.id} "
                f"(generation {program.generation})"
            )
        return program, summary, cost

    async def _step2_global_insights_async(
        self, individual_summaries: str, best_program: Optional[Program] = None
    ) -> Tuple[Optional[str], float]:
        """Async version of Step 2: Generate global insights from individual summaries."""
        previous_insights = (
            self.sync_summarizer.meta_scratch_pad or "*No previous insights available.*"
        )

        # Format best program information
        if best_program:
            best_program_info = construct_individual_program_msg(
                best_program,
                language=self.sync_summarizer.language,
                include_text_feedback=self.sync_summarizer.use_text_feedback,
            )
        else:
            best_program_info = "*No best program information available.*"

        user_msg = (
            META_STEP2_USER_MSG.replace("{individual_summaries}", individual_summaries)
            .replace("{previous_insights}", previous_insights)
            .replace("{best_program_info}", best_program_info)
        )

        response = await self.async_llm_client.query(
            msg=user_msg,
            system_msg=META_STEP2_SYSTEM_MSG,
        )

        if response is None:
            logger.error("Step 2: Failed to get response from async meta LLM client")
            return None, 0.0

        cost = response.cost or 0.0
        logger.info(f"==> Step 2 - Global insights generated (cost: ${cost:.4f})")
        return response.content.strip(), cost

    async def _step3_generate_recommendations_async(
        self, global_insights: str, best_program: Optional[Program] = None
    ) -> Tuple[Optional[str], float]:
        """Async version of Step 3: Generate recommendations based on global insights."""
        previous_recommendations = (
            self.sync_summarizer.meta_recommendations
            or "*No previous recommendations available.*"
        )

        # Format best program information
        if best_program:
            best_program_info = construct_individual_program_msg(
                best_program,
                language=self.sync_summarizer.language,
                include_text_feedback=self.sync_summarizer.use_text_feedback,
            )
        else:
            best_program_info = "*No best program information available.*"

        user_msg = (
            META_STEP3_USER_MSG.replace("{global_insights}", global_insights)
            .replace("{previous_recommendations}", previous_recommendations)
            .replace(
                "{max_recommendations}", str(self.sync_summarizer.max_recommendations)
            )
            .replace("{best_program_info}", best_program_info)
        )

        response = await self.async_llm_client.query(
            msg=user_msg,
            system_msg=META_STEP3_SYSTEM_MSG,
        )

        if response is None:
            logger.error("Step 3: Failed to get response from async meta LLM client")
            return None, 0.0

        cost = response.cost or 0.0
        logger.info(f"==> Step 3 - Recommendations generated (cost: ${cost:.4f})")
        return response.content.strip(), cost

    async def perform_final_summary_async(
        self,
        results_dir: str,
        best_program: Optional[Program] = None,
        store_cost: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> tuple[bool, float]:
        """Async version of perform_final_summary.

        With store_cost, a task writes the cost of the update to the database
        after the update. An error of the output file goes to the log.

        Returns:
            Tuple of (success, meta_cost)
        """
        if not self.async_llm_client:
            logger.info("No async meta LLM client configured, skipping final summary")
            return False, 0.0

        unprocessed_count = len(self.sync_summarizer.evaluated_since_last_meta)
        if unprocessed_count == 0:
            logger.info("No unprocessed programs for final summary")
            return False, 0.0

        logger.info(
            f"Performing final meta summary for {unprocessed_count} "
            f"remaining programs..."
        )

        updated_recs, meta_cost = await self.update_meta_memory_async(
            best_program, store_cost
        )
        if updated_recs:
            try:
                await self.write_meta_output_async(results_dir)
            except Exception as e:
                logger.error(f"❌ Failed to write the meta output file: {e}")
            logger.info(f"Final meta summary completed (cost: ${meta_cost:.4f})")
            return True, meta_cost
        else:
            logger.warning("Final meta summary failed to generate recommendations")
            return False, 0.0

    async def write_meta_output_async(self, results_dir: str) -> None:
        """Async version of write_meta_output - write files in thread pool."""
        output_str = ""

        if self.sync_summarizer.meta_summary:
            output_str += "# INDIVIDUAL PROGRAM SUMMARIES\n\n"
            output_str += (
                "The following are summaries of individual programs "
                "evaluated since the last meta update:\n\n"
            )
            output_str += str(self.sync_summarizer.meta_summary)
            output_str += "\n\n"

        if self.sync_summarizer.meta_scratch_pad:
            output_str += "# GLOBAL INSIGHTS SCRATCHPAD\n\n"
            output_str += (
                "The following are global insights about optimization "
                "approaches and their effectiveness:\n\n"
            )
            output_str += str(self.sync_summarizer.meta_scratch_pad)
            output_str += "\n\n"

        if self.sync_summarizer.meta_recommendations:
            output_str += "# META RECOMMENDATIONS\n\n"
            output_str += (
                "The following are actionable recommendations for the next "
                "program generations:\n\n"
            )
            output_str += str(self.sync_summarizer.meta_recommendations)

        if output_str:
            # Create meta subdirectory if it doesn't exist
            meta_path = self.sync_summarizer.get_meta_output_path(results_dir)
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_file, meta_path, output_str)
            logger.info(f"Wrote meta output to {meta_path}")

    async def restore_meta_output_async(self, results_dir: str) -> None:
        """Write the output file of the last meta update again when it is missing."""
        if (
            self.sync_summarizer.total_programs_processed > 0
            and not self.sync_summarizer.get_meta_output_path(results_dir).exists()
        ):
            await self.write_meta_output_async(results_dir)

    def _write_file(self, path: Path, content: str) -> None:
        """Helper to write file synchronously (for thread pool)."""
        # Write to a temporary file first, then rename it: a stop leaves no
        # partial output file
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            f.write(content)
        temp_path.replace(path)

    # Delegate read-only methods to sync summarizer
    def add_evaluated_program(self, program: Program) -> None:
        """Add newly evaluated program to the tracking list."""
        return self.sync_summarizer.add_evaluated_program(program)

    def add_unknown_programs(self, programs: List[Program]) -> None:
        """Add each program that the state does not hold (delegated to sync)."""
        return self.sync_summarizer.add_unknown_programs(programs)

    def should_update_meta(self, meta_rec_interval: Optional[int]) -> bool:
        """Check if meta update should be performed based on interval."""
        return self.sync_summarizer.should_update_meta(meta_rec_interval)

    def get_current(self):
        """Get current meta recommendations without updating."""
        return self.sync_summarizer.get_current()

    def get_sampled_recommendation(self):
        """Sample a single recommendation from the current recommendations."""
        return self.sync_summarizer.get_sampled_recommendation()

    def save_meta_state(self, filepath: str) -> None:
        """Save the meta state to a file (delegated to sync).

        The file holds each update whose cost the database holds, and no
        other. The programs of an unsaved update wait in the file.
        """
        if not self.unsaved_updates:
            return self.sync_summarizer.save_meta_state(filepath)
        unsaved_programs = [p for u in self.unsaved_updates for p in u.programs]
        self.saved_state.evaluated_since_last_meta = (
            unsaved_programs + self.sync_summarizer.evaluated_since_last_meta
        )
        return self.saved_state.save_meta_state(filepath)

    def load_meta_state(self, filepath: str) -> bool:
        """Load the meta state from a file (delegated to sync)."""
        return self.sync_summarizer.load_meta_state(filepath)

    # Delegate other methods to sync summarizer
    def __getattr__(self, name):
        """Delegate unknown methods to sync summarizer."""
        return getattr(self.sync_summarizer, name)
