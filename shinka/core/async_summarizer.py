"""
Async meta summarizer for concurrent meta-analysis.
Provides non-blocking meta summarization with concurrent LLM calls.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Tuple
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

# Tries of the cost write of a meta update in one call, and the seconds between
# 2 tries
META_COST_WRITE_TRIES = 3
META_COST_WRITE_RETRY_DELAY = 1.0


@dataclass
class MetaUpdate:
    """The result and the cost of a meta update, before the state takes it."""

    update_id: str
    program_ids: List[str]
    individual_summaries: str
    global_insights: str
    recommendations: str
    cost: float
    store_cost: Optional[Callable[[float, str], Awaitable[None]]]


class AsyncMetaSummarizer:
    """Async version of MetaSummarizer for concurrent meta-analysis."""

    def __init__(
        self,
        sync_summarizer: MetaSummarizer,
        async_llm_client: Optional[AsyncLLMClient] = None,
    ):
        """Initialize with existing sync summarizer.

        Args:
            sync_summarizer: The synchronous MetaSummarizer instance
            async_llm_client: Optional async LLM client for meta analysis
        """
        self.sync_summarizer = sync_summarizer
        self.async_llm_client = async_llm_client
        # A meta update whose cost write failed: it waits for that write
        self.pending_update: Optional[MetaUpdate] = None

    async def update_meta_memory_async(
        self,
        best_program: Optional[Program] = None,
        store_cost: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Async version of update_meta_memory.
        Perform 3-step meta-analysis and update internal state.
        A program with no step 1 answer stays in the waiting list.
        With store_cost, the update writes its cost before its result goes
        into the state. When each try of that write fails, the update keeps
        its result and its cost, and the next call writes the cost again
        with no new LLM query. A time limit or a cancel in the write also
        keeps the update.
        Returns tuple of (updated_recommendations, total_cost) or
        (None, 0.0) if no update occurred.
        """
        if not self.async_llm_client:
            logger.warning("No async meta LLM client configured")
            return None, 0.0

        if self.pending_update is not None:
            logger.info("Writing the cost of the pending meta update again...")
            return await self._take_update(self.pending_update)

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

        self.pending_update = MetaUpdate(
            update_id=uuid.uuid4().hex,
            program_ids=[p.id for p in summarized_programs],
            individual_summaries=individual_summaries,
            global_insights=global_insights,
            recommendations=recommendations,
            cost=total_meta_cost,
            store_cost=store_cost,
        )
        return await self._take_update(self.pending_update)

    async def _take_update(self, update: MetaUpdate) -> Tuple[Optional[str], float]:
        """Write the cost of an update, then put its result into the state.

        Write the cost before the result goes into the state: no save of the
        state then holds an update with no cost in the database.
        """
        if not await self._write_update_cost(update):
            return None, 0.0
        self.pending_update = None

        # Update internal state of sync summarizer
        # Concatenate new individual summaries to existing ones
        if self.sync_summarizer.meta_summary:
            self.sync_summarizer.meta_summary += "\n\n" + update.individual_summaries
        else:
            self.sync_summarizer.meta_summary = update.individual_summaries

        self.sync_summarizer.meta_scratch_pad = update.global_insights
        self.sync_summarizer.meta_recommendations = update.recommendations

        # Store the newly generated recommendations in history immediately
        if update.recommendations and isinstance(update.recommendations, str):
            self.sync_summarizer.meta_recommendations_history.append(
                update.recommendations
            )
            logger.debug(
                f"Added new recommendations to history "
                f"(total: {len(self.sync_summarizer.meta_recommendations_history)})"
            )

        # Remove only the summarized programs from the evaluated programs list
        # A program with no step 1 answer stays there, and the next meta
        # update tries it again
        num_processed = len(update.program_ids)
        self.sync_summarizer.total_programs_processed += num_processed
        self.sync_summarizer.processed_program_ids.extend(update.program_ids)
        self.sync_summarizer.evaluated_since_last_meta = [
            p
            for p in self.sync_summarizer.evaluated_since_last_meta
            if p.id not in update.program_ids
        ]
        logger.info(
            f"Processed and cleared {num_processed} programs from meta memory "
            f"(total processed: {self.sync_summarizer.total_programs_processed}, "
            f"waiting: {len(self.sync_summarizer.evaluated_since_last_meta)})"
        )

        return (
            (
                self.sync_summarizer.meta_recommendations
                if isinstance(self.sync_summarizer.meta_recommendations, str)
                else None
            ),
            update.cost,
        )

    async def _write_update_cost(self, update: MetaUpdate) -> bool:
        """Write the cost of an update. Return False when each try fails.

        Each try has the same update id, thus the database adds the cost one
        time, also when an earlier try commits after a time limit.
        """
        store_cost = update.store_cost
        if store_cost is None or update.cost <= 0:
            return True
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
            f"Meta update {update.update_id} keeps its result, "
            f"and the next update writes its cost again"
        )
        return False

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
        state_path: Optional[str] = None,
    ) -> tuple[bool, float]:
        """Async version of perform_final_summary.

        With store_cost, the update writes its cost to the database first.
        Then its result goes into the state. With a state_path, the state then
        goes to that file, before the output file. Thus the database holds the
        cost of each update of the state, in memory and in the file.

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

        # A pending update holds only its own programs. Then one more update
        # takes the programs that came after it
        updates, total_cost = 0, 0.0
        while self.sync_summarizer.evaluated_since_last_meta:
            takes_a_pending_update = self.pending_update is not None
            updated_recs, meta_cost = await self.update_meta_memory_async(
                best_program, store_cost
            )
            if not updated_recs:
                break
            logger.info(f"Final meta summary completed (cost: ${meta_cost:.4f})")
            updates, total_cost = updates + 1, total_cost + meta_cost

            # Save the state after the cost, and before the output file, so
            # that a resume does not repeat the update
            if state_path:
                self.save_meta_state(state_path)
            await self.write_meta_output_async(results_dir)
            if not takes_a_pending_update:
                break

        if updates == 0:
            logger.warning("Final meta summary failed to generate recommendations")
            return False, 0.0
        return True, total_cost

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
        """Save the meta state to a file (delegated to sync)."""
        return self.sync_summarizer.save_meta_state(filepath)

    def load_meta_state(self, filepath: str) -> bool:
        """Load the meta state from a file (delegated to sync)."""
        return self.sync_summarizer.load_meta_state(filepath)

    # Delegate other methods to sync summarizer
    def __getattr__(self, name):
        """Delegate unknown methods to sync summarizer."""
        return getattr(self.sync_summarizer, name)
