from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import ValidationError

from browser_use.agent.views import AgentHistoryList
from browser_use.agent.state_manager import AgentState
from browser_use.browser import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.browser.types import Browser, BrowserContext, Page
from browser_use.controller.service import Controller
from browser_use.llm.base import BaseChatModel
from browser_use.exceptions import AgentConfigurationError

AgentHookFunc = Callable[['Agent'], Awaitable[None]]
AgentDoneHookFunc = Callable[['AgentHistoryList'], Awaitable[None]]

class AgentSettings(BaseModel):
    task: str
    llm: BaseChatModel
    controller: Controller = Field(default_factory=Controller)
    browser_session: Optional[BrowserSession] = None
    use_planner: bool = True
    reflect_on_error: bool = True
    planner_interval: int = 5
    max_steps: int = 100
    max_failures: int = 3
    max_actions_per_step: int = 10
    use_thinking: bool = True
    flash_mode: bool = False
    planner_llm: Optional[BaseChatModel] = None
    page_extraction_llm: Optional[BaseChatModel] = None
    injected_agent_state: Optional[AgentState] = None
    context: Optional[Any] = None
    sensitive_data: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    initial_actions: Optional[List[Dict[str, Any]]] = None
    available_file_paths: list[Union[str, Path]] = Field(default_factory=list)
    images_per_step: int = 1
    max_history_items: int = 40
    include_attributes: list[str] = Field(default_factory=lambda: ["data-test-id", "data-testid", "aria-label", "placeholder", "title", "alt"])
    include_tool_call_examples: bool = False
    on_run_start: Optional[AgentHookFunc] = None
    on_step_start: Optional[AgentHookFunc] = None
    on_step_end: Optional[AgentHookFunc] = None
    on_run_end: Optional[AgentDoneHookFunc] = None
    generate_gif: Union[bool, str] = False
    save_conversation_path: Optional[str] = None
    file_system_path: Optional[str] = None
    page: Optional[Page] = None
    max_perception_staleness_seconds: float = 10.0
    lock_timeout_seconds: float = Field(5.0, description="Timeout in seconds for acquiring the state lock to prevent deadlocks.")
    check_ui_stability: bool = True
    output_model: Optional[type[BaseModel]] = None
    browser: Optional[Union[Browser, BrowserSession]] = None
    browser_context: Optional[BrowserContext] = None
    browser_profile: Optional[BrowserProfile] = None
    file_system: Any = None # To be populated by Supervisor
    is_planner_reasoning: bool = Field(False, description="Controls if the planner prompt encourages verbose reasoning.")
    extend_planner_system_message: Optional[str] = Field(None, description="Additional text for the planner's system message.")
    calculate_cost: bool = Field(False, description="Whether to calculate and track token costs for LLM calls.")


    class Config:
        arbitrary_types_allowed = True

    def parse_initial_actions(self, action_model: Any) -> list[Any]:
        if not self.initial_actions:
            return []
        try:
            return [action_model.model_validate(a) for a in self.initial_actions]
        except ValidationError as e:
            raise AgentConfigurationError(f"Validation error for initial actions: {e}")