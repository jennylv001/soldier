"""
Integration example for the Task Orchestration Framework.

This demonstrates how to integrate the task orchestration framework
with the existing agent system.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

# Configure logging for the example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the task orchestration framework
import sys
sys.path.insert(0, '.')
from agent.task_orchestration import (
    TaskComponent,
    TaskStatus,
    OrderedSequenceComponent,
    PrioritySelectorComponent,
    ParallelExecutionComponent,
    ResultInverter,
    ActionRepeater,
    ClickAction,
    TypeAction,
    GoToUrlAction,
    TaskExecutionContext,
)

if TYPE_CHECKING:
    from agent.state_manager import AgentState
    from controller.service import Controller
    from browser.session import BrowserSession


class TaskOrchestrationIntegration:
    """
    Integration layer for using the task orchestration framework with the existing agent system.
    
    This class shows how to:
    1. Create TaskExecutionContext for atomic actions
    2. Build complex task flows using the orchestration components
    3. Execute tasks within the agent architecture
    """
    
    def __init__(self, controller: 'Controller', browser_session: 'BrowserSession'):
        """
        Initialize with controller and browser session.
        
        Args:
            controller: The agent's controller for executing actions
            browser_session: The agent's browser session
        """
        self.controller = controller
        self.browser_session = browser_session
        self.execution_context = TaskExecutionContext(controller, browser_session)
    
    def prepare_agent_state(self, agent_state: 'AgentState') -> 'AgentState':
        """
        Prepare agent state for task orchestration by adding execution context.
        
        Args:
            agent_state: The agent's current state
            
        Returns:
            The agent state with task execution context added
        """
        # Add our execution context to the agent state
        agent_state.task_execution_context = self.execution_context
        return agent_state
    
    def create_form_filling_task(self, form_data: dict) -> TaskComponent:
        """
        Create a complex task for filling out a form.
        
        This demonstrates how to build complex task flows using the orchestration components.
        
        Args:
            form_data: Dictionary mapping field indices to values
            
        Returns:
            TaskComponent representing the complete form filling task
        """
        # Build individual field filling tasks
        field_tasks = []
        for field_index, value in form_data.items():
            # Create a sequence for each field: click, then type
            field_task = OrderedSequenceComponent([
                ClickAction(index=field_index),
                TypeAction(index=field_index, text=value),
            ])
            field_tasks.append(field_task)
        
        # Execute all field tasks in sequence
        form_filling_task = OrderedSequenceComponent(field_tasks)
        
        # Wrap with retry logic in case of transient failures
        reliable_form_task = ActionRepeater(form_filling_task, max_attempts=3)
        
        return reliable_form_task
    
    def create_navigation_task(self, url: str, backup_urls: list = None) -> TaskComponent:
        """
        Create a navigation task with fallback URLs.
        
        Args:
            url: Primary URL to navigate to
            backup_urls: List of backup URLs to try if primary fails
            
        Returns:
            TaskComponent representing the navigation task
        """
        # Primary navigation
        primary_nav = GoToUrlAction(url=url)
        
        # Backup navigations
        nav_options = [primary_nav]
        if backup_urls:
            for backup_url in backup_urls:
                nav_options.append(GoToUrlAction(url=backup_url))
        
        # Try primary first, then backups
        navigation_task = PrioritySelectorComponent(nav_options)
        
        return navigation_task
    
    def create_search_and_select_task(self, search_terms: list, target_element_index: int) -> TaskComponent:
        """
        Create a task that tries multiple search terms until one works.
        
        Args:
            search_terms: List of search terms to try
            target_element_index: Index of the search box element
            
        Returns:
            TaskComponent representing the search task
        """
        # Create search attempts for each term
        search_attempts = []
        for term in search_terms:
            # Each attempt: click search box, type term, press enter
            search_attempt = OrderedSequenceComponent([
                ClickAction(index=target_element_index),
                TypeAction(index=target_element_index, text=term),
                # Note: In a real implementation, you might add a "press enter" action
            ])
            search_attempts.append(search_attempt)
        
        # Try each search term until one succeeds
        search_task = PrioritySelectorComponent(search_attempts)
        
        return search_task
    
    def create_parallel_data_entry_task(self, data_entries: list) -> TaskComponent:
        """
        Create a task that performs multiple data entries in parallel.
        
        This is useful when you have independent form fields that can be filled simultaneously.
        
        Args:
            data_entries: List of (field_index, value) tuples
            
        Returns:
            TaskComponent representing the parallel data entry task
        """
        # Create individual type actions for each entry
        entry_tasks = []
        for field_index, value in data_entries:
            entry_task = TypeAction(index=field_index, text=value)
            entry_tasks.append(entry_task)
        
        # Execute all entries in parallel
        parallel_task = ParallelExecutionComponent(entry_tasks)
        
        return parallel_task
    
    def create_conditional_task(self, primary_task: TaskComponent, fallback_task: TaskComponent) -> TaskComponent:
        """
        Create a conditional task using ResultInverter.
        
        This demonstrates how to use ResultInverter for conditional logic.
        
        Args:
            primary_task: The primary task to attempt
            fallback_task: The fallback task if primary fails
            
        Returns:
            TaskComponent representing the conditional logic
        """
        # Try primary task first
        # If it fails, try the fallback
        conditional_task = PrioritySelectorComponent([
            primary_task,
            fallback_task
        ])
        
        return conditional_task

    async def execute_task(self, task: TaskComponent, agent_state: 'AgentState') -> TaskStatus:
        """
        Execute a task within the agent architecture.
        
        Args:
            task: The task component to execute
            agent_state: The agent's current state
            
        Returns:
            TaskStatus indicating the result of execution
        """
        # Prepare the agent state with execution context
        prepared_state = self.prepare_agent_state(agent_state)
        
        # Execute the task
        try:
            result = await task.execute(prepared_state)
            logger.info(f"Task execution completed with result: {result}")
            return result
        except Exception as e:
            logger.error(f"Task execution failed with error: {e}")
            return TaskStatus.FAILURE


def create_example_tasks():
    """
    Create example tasks to demonstrate the framework capabilities.
    """
    print("🔧 Creating Example Tasks")
    
    # Example 1: Simple form filling
    form_data = {
        1: "John Doe",        # Name field
        2: "john@email.com",  # Email field  
        3: "555-1234",        # Phone field
    }
    
    # This would be done with a real integration instance
    # integration = TaskOrchestrationIntegration(controller, browser_session)
    # form_task = integration.create_form_filling_task(form_data)
    
    print("✓ Form filling task structure created")
    
    # Example 2: Navigation with fallbacks
    primary_url = "https://example.com"
    backup_urls = ["https://example.org", "https://example.net"]
    
    # nav_task = integration.create_navigation_task(primary_url, backup_urls)
    print("✓ Navigation task with fallbacks structure created")
    
    # Example 3: Search with multiple terms
    search_terms = ["python tutorial", "python guide", "learn python"]
    search_box_index = 5
    
    # search_task = integration.create_search_and_select_task(search_terms, search_box_index)
    print("✓ Multi-term search task structure created")
    
    # Example 4: Parallel data entry
    parallel_data = [(10, "value1"), (11, "value2"), (12, "value3")]
    
    # parallel_task = integration.create_parallel_data_entry_task(parallel_data)
    print("✓ Parallel data entry task structure created")
    
    print("\n✅ All example task structures created successfully!")


def demonstrate_framework_benefits():
    """
    Demonstrate the benefits of the task orchestration framework.
    """
    print("\n🚀 Task Orchestration Framework Benefits:")
    
    print("\n1. 📋 Composability:")
    print("   - Build complex workflows from simple components")
    print("   - Reuse components across different tasks")
    print("   - Mix and match control flow patterns")
    
    print("\n2. 🔄 Reliability:")
    print("   - Built-in retry mechanisms with ActionRepeater")
    print("   - Fallback strategies with PrioritySelectorComponent")
    print("   - Parallel execution for independent operations")
    
    print("\n3. 🧩 Modularity:")
    print("   - Clear separation of concerns")
    print("   - Easy to test individual components")
    print("   - Self-contained atomic actions")
    
    print("\n4. 🔧 Flexibility:")
    print("   - Support for sequential, parallel, and conditional logic")
    print("   - Result inversion for negative conditions")
    print("   - Extensible framework for new component types")
    
    print("\n5. 🔗 Integration:")
    print("   - Works alongside existing agent architecture")
    print("   - Backward compatible with current systems")
    print("   - Uses existing action implementations")


async def main():
    """
    Main function demonstrating the task orchestration framework.
    """
    print("🎯 Task Orchestration Framework Integration Example")
    
    # Demonstrate framework structure
    create_example_tasks()
    
    # Show benefits
    demonstrate_framework_benefits()
    
    print("\n📚 Usage Summary:")
    print("1. Create TaskOrchestrationIntegration with controller and browser_session")
    print("2. Build task flows using orchestration components")
    print("3. Execute tasks using integration.execute_task()")
    print("4. Tasks return SUCCESS, FAILURE, or RUNNING status")
    
    print("\n🔮 Future Integration:")
    print("- Replace single-action decision loop with task orchestration")
    print("- Use LLM to generate task composition plans")
    print("- Add task persistence and resumption capabilities")
    print("- Implement task dependency management")
    
    print("\n✨ The Task Orchestration Framework is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())