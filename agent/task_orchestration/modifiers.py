"""
Action-modifying components for task orchestration.

These components modify the behavior of child components.
"""

import logging
from typing import TYPE_CHECKING

from .base import TaskComponent, TaskStatus

if TYPE_CHECKING:
    from browser_use.agent.state_manager import AgentState

logger = logging.getLogger(__name__)


class ResultInverter(TaskComponent):
    """
    Executes a single child component and flips the result.
    
    SUCCESS becomes FAILURE and FAILURE becomes SUCCESS.
    RUNNING remains RUNNING.
    """
    
    def __init__(self, child: TaskComponent):
        """
        Initialize with a child component.
        
        Args:
            child: TaskComponent instance whose result will be inverted
        """
        self.child = child
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute child and invert SUCCESS/FAILURE results."""
        logger.debug("ResultInverter executing child component")
        
        try:
            result = await self.child.execute(agent_state)
            logger.debug(f"Child returned {result}")
            
            if result == TaskStatus.SUCCESS:
                logger.debug("ResultInverter inverting SUCCESS to FAILURE")
                return TaskStatus.FAILURE
            elif result == TaskStatus.FAILURE:
                logger.debug("ResultInverter inverting FAILURE to SUCCESS")
                return TaskStatus.SUCCESS
            else:  # RUNNING
                logger.debug("ResultInverter returning RUNNING unchanged")
                return TaskStatus.RUNNING
                
        except Exception as e:
            logger.error(f"ResultInverter child raised exception: {e}")
            # Exception is treated as failure, so invert to success
            return TaskStatus.SUCCESS


class ActionRepeater(TaskComponent):
    """
    Re-runs a single child component a specified number of times.
    
    Returns SUCCESS if the child succeeds at least once.
    Returns FAILURE if the child fails all attempts.
    Returns RUNNING if the child is still running on current attempt.
    """
    
    def __init__(self, child: TaskComponent, max_attempts: int = 3):
        """
        Initialize with a child component and max attempts.
        
        Args:
            child: TaskComponent instance to repeat
            max_attempts: Maximum number of attempts (default: 3)
        """
        self.child = child
        self.max_attempts = max(1, max_attempts)  # Ensure at least 1 attempt
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute child up to max_attempts times until success."""
        logger.debug(f"ActionRepeater executing child up to {self.max_attempts} times")
        
        for attempt in range(self.max_attempts):
            try:
                result = await self.child.execute(agent_state)
                logger.debug(f"Attempt {attempt + 1}: Child returned {result}")
                
                if result == TaskStatus.SUCCESS:
                    logger.debug(f"ActionRepeater returning SUCCESS after {attempt + 1} attempts")
                    return TaskStatus.SUCCESS
                elif result == TaskStatus.RUNNING:
                    logger.debug(f"ActionRepeater returning RUNNING on attempt {attempt + 1}")
                    return TaskStatus.RUNNING
                # Continue to next attempt if FAILURE
                
            except Exception as e:
                logger.error(f"ActionRepeater attempt {attempt + 1} raised exception: {e}")
                # Continue to next attempt
                continue
        
        logger.debug(f"ActionRepeater returning FAILURE after {self.max_attempts} failed attempts")
        return TaskStatus.FAILURE