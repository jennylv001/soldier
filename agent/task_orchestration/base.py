"""
Base classes and enums for the Task Orchestration Framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.agent.state_manager import AgentState


class TaskStatus(Enum):
    """Status values that TaskComponents can return."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class TaskComponent(ABC):
    """
    Base class for all task orchestration components.
    
    All logical units in the task orchestration framework must inherit from this class
    and implement the execute method.
    """
    
    @abstractmethod
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """
        Execute this task component.
        
        Args:
            agent_state: The current agent state containing browser session, 
                        task context, and other relevant information.
                        
        Returns:
            TaskStatus: One of SUCCESS, FAILURE, or RUNNING
        """
        pass