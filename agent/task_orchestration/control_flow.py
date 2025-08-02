"""
Control flow components for task orchestration.

These components manage the execution flow of child components.
"""

import asyncio
import logging
from typing import List, TYPE_CHECKING

from .base import TaskComponent, TaskStatus

if TYPE_CHECKING:
    from browser_use.agent.state_manager import AgentState

logger = logging.getLogger(__name__)


class OrderedSequenceComponent(TaskComponent):
    """
    Runs child components sequentially.
    
    Returns FAILURE on first child failure.
    Returns SUCCESS only when all children succeed.
    """
    
    def __init__(self, children: List[TaskComponent]):
        """
        Initialize with a list of child components.
        
        Args:
            children: List of TaskComponent instances to execute in order
        """
        self.children = children
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute children in sequence, stopping on first failure."""
        logger.debug(f"OrderedSequenceComponent executing {len(self.children)} children")
        
        for i, child in enumerate(self.children):
            try:
                result = await child.execute(agent_state)
                logger.debug(f"Child {i} returned {result}")
                
                if result == TaskStatus.FAILURE:
                    logger.debug("OrderedSequenceComponent returning FAILURE due to child failure")
                    return TaskStatus.FAILURE
                elif result == TaskStatus.RUNNING:
                    logger.debug("OrderedSequenceComponent returning RUNNING due to child still running")
                    return TaskStatus.RUNNING
                # Continue with next child if SUCCESS
                
            except Exception as e:
                logger.error(f"OrderedSequenceComponent child {i} raised exception: {e}")
                return TaskStatus.FAILURE
        
        logger.debug("OrderedSequenceComponent returning SUCCESS - all children succeeded")
        return TaskStatus.SUCCESS


class PrioritySelectorComponent(TaskComponent):
    """
    Tries child components in order.
    
    Returns SUCCESS on first child success.
    Returns FAILURE only if all children fail.
    """
    
    def __init__(self, children: List[TaskComponent]):
        """
        Initialize with a list of child components.
        
        Args:
            children: List of TaskComponent instances to try in priority order
        """
        self.children = children
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Try children in order, stopping on first success."""
        logger.debug(f"PrioritySelectorComponent trying {len(self.children)} children")
        
        for i, child in enumerate(self.children):
            try:
                result = await child.execute(agent_state)
                logger.debug(f"Child {i} returned {result}")
                
                if result == TaskStatus.SUCCESS:
                    logger.debug("PrioritySelectorComponent returning SUCCESS due to child success")
                    return TaskStatus.SUCCESS
                elif result == TaskStatus.RUNNING:
                    logger.debug("PrioritySelectorComponent returning RUNNING due to child still running")
                    return TaskStatus.RUNNING
                # Continue with next child if FAILURE
                
            except Exception as e:
                logger.error(f"PrioritySelectorComponent child {i} raised exception: {e}")
                # Continue trying next child
                continue
        
        logger.debug("PrioritySelectorComponent returning FAILURE - all children failed")
        return TaskStatus.FAILURE


class ParallelExecutionComponent(TaskComponent):
    """
    Runs all child components simultaneously.
    
    Returns SUCCESS only if all children succeed.
    Returns FAILURE if any child fails.
    Returns RUNNING if any child is still running.
    """
    
    def __init__(self, children: List[TaskComponent]):
        """
        Initialize with a list of child components.
        
        Args:
            children: List of TaskComponent instances to execute in parallel
        """
        self.children = children
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute all children in parallel."""
        logger.debug(f"ParallelExecutionComponent executing {len(self.children)} children in parallel")
        
        if not self.children:
            return TaskStatus.SUCCESS
        
        try:
            # Execute all children concurrently
            tasks = [child.execute(agent_state) for child in self.children]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            has_failure = False
            has_running = False
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"ParallelExecutionComponent child {i} raised exception: {result}")
                    has_failure = True
                elif result == TaskStatus.FAILURE:
                    logger.debug(f"ParallelExecutionComponent child {i} failed")
                    has_failure = True
                elif result == TaskStatus.RUNNING:
                    logger.debug(f"ParallelExecutionComponent child {i} still running")
                    has_running = True
            
            # Determine final status based on children results
            if has_failure:
                logger.debug("ParallelExecutionComponent returning FAILURE due to child failures")
                return TaskStatus.FAILURE
            elif has_running:
                logger.debug("ParallelExecutionComponent returning RUNNING due to running children")
                return TaskStatus.RUNNING
            else:
                logger.debug("ParallelExecutionComponent returning SUCCESS - all children succeeded")
                return TaskStatus.SUCCESS
                
        except Exception as e:
            logger.error(f"ParallelExecutionComponent encountered error: {e}")
            return TaskStatus.FAILURE