"""
Atomic action components for task orchestration.

These components wrap existing action functionality into TaskComponent interface.
They require additional context (controller, browser_session) to operate properly.
"""

import logging
from typing import TYPE_CHECKING, Optional

from .base import TaskComponent, TaskStatus

if TYPE_CHECKING:
    from browser_use.agent.state_manager import AgentState
    from browser_use.browser.session import BrowserSession
    from browser_use.controller.service import Controller

logger = logging.getLogger(__name__)


class TaskExecutionContext:
    """
    Context object that provides necessary dependencies for atomic actions.
    
    This allows the task orchestration framework to access browser functionality
    without tight coupling to the agent architecture.
    """
    
    def __init__(self, controller: 'Controller', browser_session: 'BrowserSession'):
        self.controller = controller
        self.browser_session = browser_session


class ClickAction(TaskComponent):
    """
    Wraps existing click functionality into TaskComponent interface.
    
    Clicks an element by its index in the selector map.
    """
    
    def __init__(self, index: int):
        """
        Initialize with element index to click.
        
        Args:
            index: The index of the element to click in the selector map
        """
        self.index = index
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute click action on the specified element."""
        logger.debug(f"ClickAction executing click on element {self.index}")
        
        try:
            # Check if the agent state has our execution context
            context = getattr(agent_state, 'task_execution_context', None)
            if not context or not isinstance(context, TaskExecutionContext):
                logger.error("ClickAction: No TaskExecutionContext available in agent_state")
                return TaskStatus.FAILURE
            
            # Import action model
            from browser_use.controller.views import ClickElementAction
            
            # Create action model
            action_model = ClickElementAction(index=self.index)
            
            # Execute using the controller
            result = await context.controller.act(
                action=action_model,
                browser_session=context.browser_session,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=[],
                file_system=None,
                context=None,
            )
            
            if result.success is False:
                logger.debug(f"ClickAction failed: {result.error}")
                return TaskStatus.FAILURE
            else:
                logger.debug(f"ClickAction succeeded: {result.extracted_content}")
                return TaskStatus.SUCCESS
                
        except Exception as e:
            logger.error(f"ClickAction encountered error: {e}")
            return TaskStatus.FAILURE


class TypeAction(TaskComponent):
    """
    Wraps existing typing functionality into TaskComponent interface.
    
    Types text into an element by its index.
    """
    
    def __init__(self, index: int, text: str):
        """
        Initialize with element index and text to type.
        
        Args:
            index: The index of the element to type into
            text: The text to type
        """
        self.index = index
        self.text = text
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute type action on the specified element."""
        logger.debug(f"TypeAction executing type '{self.text}' into element {self.index}")
        
        try:
            # Check if the agent state has our execution context
            context = getattr(agent_state, 'task_execution_context', None)
            if not context or not isinstance(context, TaskExecutionContext):
                logger.error("TypeAction: No TaskExecutionContext available in agent_state")
                return TaskStatus.FAILURE
            
            # Import action model
            from browser_use.controller.views import InputTextAction
            
            # Create action model
            action_model = InputTextAction(index=self.index, text=self.text)
            
            # Execute using the controller
            result = await context.controller.act(
                action=action_model,
                browser_session=context.browser_session,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=[],
                file_system=None,
                context=None,
            )
            
            if result.success is False:
                logger.debug(f"TypeAction failed: {result.error}")
                return TaskStatus.FAILURE
            else:
                logger.debug(f"TypeAction succeeded: {result.extracted_content}")
                return TaskStatus.SUCCESS
                
        except Exception as e:
            logger.error(f"TypeAction encountered error: {e}")
            return TaskStatus.FAILURE


class GoToUrlAction(TaskComponent):
    """
    Wraps existing URL navigation functionality into TaskComponent interface.
    
    Navigates to a specified URL.
    """
    
    def __init__(self, url: str, new_tab: bool = False):
        """
        Initialize with URL to navigate to.
        
        Args:
            url: The URL to navigate to
            new_tab: Whether to open in a new tab
        """
        self.url = url
        self.new_tab = new_tab
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Execute navigation to the specified URL."""
        logger.debug(f"GoToUrlAction executing navigation to {self.url}")
        
        try:
            # Check if the agent state has our execution context
            context = getattr(agent_state, 'task_execution_context', None)
            if not context or not isinstance(context, TaskExecutionContext):
                logger.error("GoToUrlAction: No TaskExecutionContext available in agent_state")
                return TaskStatus.FAILURE
            
            # Import action model
            from browser_use.controller.views import GoToUrlAction as GoToUrlActionModel
            
            # Create action model
            action_model = GoToUrlActionModel(url=self.url, new_tab=self.new_tab)
            
            # Execute using the controller
            result = await context.controller.act(
                action=action_model,
                browser_session=context.browser_session,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=[],
                file_system=None,
                context=None,
            )
            
            if result.success is False:
                logger.debug(f"GoToUrlAction failed: {result.error}")
                return TaskStatus.FAILURE
            else:
                logger.debug(f"GoToUrlAction succeeded: {result.extracted_content}")
                return TaskStatus.SUCCESS
                
        except Exception as e:
            logger.error(f"GoToUrlAction encountered error: {e}")
            return TaskStatus.FAILURE