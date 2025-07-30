from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from browser_use.agent.events import Decision, PerceptionOutput
from browser_use.agent.prompts import PlannerPrompt, SystemPrompt
from browser_use.agent.state_manager import AgentStatus, agent_log
from browser_use.agent.views import AgentOutput, ReflectionPlannerOutput
from browser_use.exceptions import LLMException

if TYPE_CHECKING:
    from browser_use.agent.message_manager.service import MessageManager
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import BaseMessage

logger = logging.getLogger(__name__)


class DecisionMaker:
    """
    "The Sacred/Profane Split"
    This component is the "sacred" synchronous core. It receives perception data,
    constructs prompts, invokes the LLM, and makes decisions. It does NOT perform
    any direct I/O (like browser actions).
    """

    def __init__(
        self,
        settings: AgentSettings,
        state_manager: StateManager,
        message_manager: MessageManager,
        llm: BaseChatModel,
        planner_llm: Optional[BaseChatModel],
    ):
        self.settings = settings
        self.state_manager = state_manager
        self.message_manager = message_manager
        self.llm = llm
        self.planner_llm = planner_llm

        self._setup_action_models()

    def _setup_action_models(self):
        # This logic is from the original Agent, needed for creating the correct output schema.
        # It's assumed Controller is already configured.
        ActionModel = self.settings.controller.registry.create_action_model()
        done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])
        if self.settings.flash_mode:
            self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
        elif self.settings.use_thinking:
            self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
        else:
            self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)

    async def decide(self, perception: PerceptionOutput) -> Decision:
        """The main entry point for the decision-making process."""
        state = self.state_manager.state
        self.message_manager.update_history_representation(state.history)

        self._setup_action_models()

        state = self.state_manager.state
        status = state.status

        if status == AgentStatus.RUNNING:
            return await self._decide_next_action(perception)
        elif status == AgentStatus.REFLECTING:
            return await self._decide_reflection()
        else:
            return Decision(messages_to_llm=[])

    async def _decide_next_action(self, perception: PerceptionOutput) -> Decision:
        state = self.state_manager.state
        agent_log(logging.INFO, state.agent_id, state.n_steps,
                  f'📍 Evaluating page with {len(perception.browser_state.selector_map)} elements on: {perception.browser_state.url[:70]}')

        messages = self.message_manager.prepare_messages_for_llm(
            browser_state=perception.browser_state,
            current_goal=state.current_goal,
            last_error=state.last_error,
            page_filtered_actions=self.settings.controller.registry.get_prompt_description(url=perception.browser_state.url),
            agent_history_list=state.history
        )
        try:
            llm_output = await self._invoke_llm_with_retry(messages)
            if not llm_output or not llm_output.action:
                raise LLMException("LLM failed to produce valid actions.")
            return Decision(messages_to_llm=messages, llm_output=llm_output, browser_state=perception.browser_state)
        except Exception as e:
            await self.state_manager.record_error(f"Error during decision making: {e}", is_critical=True)
            return Decision(messages_to_llm=messages)

    async def _decide_reflection(self) -> Decision:
        """Handles the reflection step by delegating prompt creation to the MessageManager."""
        state = self.state_manager.state
        if not self.planner_llm:
            await self.state_manager.set_status(AgentStatus.RUNNING)
            return Decision(messages_to_llm=[])

        agent_log(logging.INFO, state.agent_id, state.n_steps, "🧠 Reflecting...")
        history_for_planner = state.history.get_last_n_items(5)
        planner_prompt = PlannerPrompt(task=state.task, history=history_for_planner, last_error=state.last_error)
        
        try:
            planner_llm_with_schema = self.planner_llm.with_structured_output(ReflectionPlannerOutput)
            response = await planner_llm_with_schema.ainvoke(planner_prompt.get_messages())
            if response.next_goal:
                await self.state_manager.update_task(response.next_goal)
                agent_log(logging.INFO, state.agent_id, state.n_steps, f"Next goal set by planner: {response.next_goal}")
        except Exception as e:
            agent_log(logging.ERROR, state.agent_id, state.n_steps, f"Planner failed: {e}")
        finally:
            await self.state_manager.clear_error_and_failures()
        
        return Decision(messages_to_llm=[])

    async def _invoke_llm_with_retry(self, messages: List[BaseMessage], max_retries: int = 2) -> AgentOutput:
        output_schema = self.DoneAgentOutput if (self.state_manager.state.n_steps + 1) >= self.settings.max_steps else self.AgentOutput
        for attempt in range(max_retries + 1):
            try:
                llm_with_schema = self.llm.with_structured_output(output_schema)
                response = await llm_with_schema.ainvoke(messages)
                if len(response.action) > self.settings.max_actions_per_step:
                    response.action = response.action[:self.settings.max_actions_per_step]
                return response
            except Exception as e:
                if attempt >= max_retries:
                    raise LLMException("LLM call failed after all retries.") from e
                await asyncio.sleep(1.0 * (2 ** attempt))
        raise LLMException("LLM call failed.")