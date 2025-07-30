from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from browser_use.agent.events import ActuationResult, Decision
from browser_use.agent.views import ActionResult, StepMetadata

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.browser import BrowserSession
    from browser_use.controller.service import Controller

logger = logging.getLogger(__name__)

class Actuator:
    """
    "Async I/O periphery"
    This component is responsible for executing actions in the environment. It
    receives a `Decision` from the core and uses the `Controller` to perform
    the actions. It is an async I/O-bound component.
    """

    def __init__(
        self,
        controller: Controller,
        browser_session: BrowserSession,
        state_manager: StateManager,
        settings: AgentSettings,
    ):
        self.controller = controller
        self.browser_session = browser_session
        self.state_manager = state_manager
        self.settings = settings

    async def execute(self, decision: Decision) -> ActuationResult:
        """Executes the actions from a decision and returns the results."""
        state = self.state_manager.state
        step_start_time = time.monotonic()
        
        if not decision.llm_output or not decision.llm_output.action:
            metadata = StepMetadata(step_number=state.n_steps, step_start_time=step_start_time, step_end_time=time.monotonic())
            return ActuationResult(action_results=[], llm_output=None, browser_state=decision.browser_state, step_metadata=metadata)

        action_results = await self.controller.multi_act(
            actions=decision.llm_output.action,
            browser_session=self.browser_session,
            page_extraction_llm=self.settings.page_extraction_llm,
            context=self.settings.context,
            sensitive_data=self.settings.sensitive_data,
            available_file_paths=self.settings.available_file_paths,
            file_system=self.settings.file_system,
        )

        metadata = StepMetadata(step_number=state.n_steps, step_start_time=step_start_time, step_end_time=time.monotonic())
        
        return ActuationResult(
            action_results=action_results,
            llm_output=decision.llm_output,
            browser_state=decision.browser_state,
            step_metadata=metadata,
        )