from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from browser_use.agent.views import ActionResult, AgentOutput, BrowserStateSummary, StepMetadata
    from browser_use.llm.messages import BaseMessage


@dataclass
class PerceptionOutput:
    """Data produced by the Perception component."""
    browser_state: BrowserStateSummary
    new_downloaded_files: Optional[list[str]] = None
    step_start_time: float = field(default_factory=time.monotonic)


@dataclass
class Decision:
    """Data produced by the DecisionMaker component."""
    messages_to_llm: list[BaseMessage]
    llm_output: Optional[AgentOutput] = None
    action_results: list[ActionResult] = field(default_factory=list)
    step_metadata: Optional[StepMetadata] = None
    browser_state: Optional[BrowserStateSummary] = None


@dataclass
class ActuationResult:
    """Data produced by the Actuator component."""
    action_results: list[ActionResult]
    llm_output: Optional[AgentOutput]
    browser_state: Optional[BrowserStateSummary]
    step_metadata: StepMetadata