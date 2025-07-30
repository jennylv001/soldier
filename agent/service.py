from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Generic, List, Optional, TypeVar

from browser_use.agent.settings import AgentSettings
from browser_use.agent.supervisor import Supervisor
from browser_use.agent.views import AgentHistory, AgentHistoryList, ActionResult, AgentOutput
from browser_use.dom.history_tree_processor.service import DOMHistoryElement, HistoryTreeProcessor
from browser_use.controller.service import Controller

if TYPE_CHECKING:
    from browser_use.agent.views import ActionModel
    from browser_use.browser.session import BrowserSession
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.llm.base import BaseChatModel

ContextT = TypeVar('ContextT')
logger = logging.getLogger(__name__)

class Agent(Generic[ContextT]):
    """Public-facing wrapper for the "Bulletproof" agent architecture."""
    settings: AgentSettings
    supervisor: Supervisor

    def __init__(self, settings: AgentSettings):
        self.settings = settings
        if not hasattr(self.settings, 'controller') or not self.settings.controller:
             self.settings.controller = Controller()
        self.supervisor = Supervisor(settings)

    async def run(self) -> AgentHistoryList:
        return await self.supervisor.run()
    
    def pause(self): self.supervisor.pause()
    def resume(self): self.supervisor.resume()
    def stop(self): self.supervisor.stop()

    def inject_human_guidance(self, text: str):
        asyncio.create_task(self.supervisor.state_manager.add_human_guidance(text))

    async def add_new_task(self, new_task: str):
        await self.supervisor.state_manager.update_task(new_task)
        self.supervisor.message_manager.add_new_task(new_task)
        logger.info(f"Agent task updated to: {new_task}")

    @property
    def state(self): return self.supervisor.state_manager.state
    @property
    def browser_session(self) -> BrowserSession: return self.supervisor.browser_session

    async def close(self): await self.supervisor.close()

    def _get_agent_output_schema(self, include_done: bool = False) -> type[AgentOutput]:
        """ ** FIX: Helper to build the correct AgentOutput schema based on settings. ** """
        action_model_source = self.settings.controller.registry
        action_model = action_model_source.create_action_model(include_actions=['done'] if include_done else [])
        
        if self.settings.flash_mode:
            return AgentOutput.type_with_custom_actions_flash_mode(action_model)
        elif self.settings.use_thinking:
            return AgentOutput.type_with_custom_actions(action_model)
        else:
            return AgentOutput.type_with_custom_actions_no_thinking(action_model)

    async def load_and_rerun(self, history_file: str, **kwargs) -> list[ActionResult]:
        agent_output_model = self._get_agent_output_schema(include_done=True)
        history = AgentHistoryList.load_from_file(history_file, agent_output_model)
        return await self.rerun_history(history, **kwargs)

    async def rerun_history(self, history: AgentHistoryList, max_retries: int = 3, skip_failures: bool = True, delay_between_actions: float = 2.0) -> list[ActionResult]:
        results = []
        for i, history_item in enumerate(history.history):
            if not history_item.model_output or not history_item.model_output.action: continue
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result); break
                except Exception as e:
                    logger.warning(f"Failed to execute step {i} from history, retry {retry_count+1}/{max_retries}. Error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        if not skip_failures: raise
                    else: await asyncio.sleep(delay_between_actions)
        return results

    async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
        state = await self.browser_session.get_state_summary()
        if not state or not history_item.model_output or not history_item.state:
            raise ValueError("Invalid state or model output in history item for replay")
        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            historical_element = history_item.state.interacted_element[i] if history_item.state.interacted_element else None
            updated_action = await self._update_action_indices(historical_element, action, state)
            if updated_action is None:
                raise ValueError(f"Could not find matching element for action {i} in current page state")
            updated_actions.append(updated_action)
        result = await self.settings.controller.multi_act(actions=updated_actions, browser_session=self.browser_session)
        await asyncio.sleep(delay)
        return result

    async def _update_action_indices(self, historical_element: DOMHistoryElement, action: ActionModel, browser_state_summary: BrowserStateSummary) -> Optional[ActionModel]:
        if not historical_element:

            if action.get_index() is not None:
                logger.warning(f"Rejecting replay of indexed action {action} due to missing historical element context.")
                return None 
            return action

        if not browser_state_summary.element_tree: return None
        current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, browser_state_summary.element_tree)
        if not current_element or current_element.highlight_index is None: return None
        action.set_index(current_element.highlight_index)
        return action