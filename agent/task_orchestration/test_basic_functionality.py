"""
Basic functionality test for the Task Orchestration Framework.

This demonstrates how to use the task orchestration components
and validates basic functionality.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

# Configure logging for testing
logging.basicConfig(level=logging.DEBUG)
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
    from browser_use.agent.state_manager import AgentState


class MockTaskComponent(TaskComponent):
    """Mock task component for testing control flow."""
    
    def __init__(self, name: str, result: TaskStatus, delay: float = 0.1):
        self.name = name
        self.result = result
        self.delay = delay
    
    async def execute(self, agent_state: 'AgentState') -> TaskStatus:
        """Mock execution that returns predetermined result."""
        logger.debug(f"MockTaskComponent '{self.name}' executing...")
        await asyncio.sleep(self.delay)
        logger.debug(f"MockTaskComponent '{self.name}' returning {self.result}")
        return self.result


class MockAgentState:
    """Mock agent state for testing."""
    
    def __init__(self):
        self.agent_id = "test-agent"
        self.task = "test task"
        self.current_goal = "test goal"
        self.n_steps = 0


async def test_basic_components():
    """Test basic component functionality."""
    print("\n=== Testing Basic Components ===")
    
    agent_state = MockAgentState()
    
    # Test individual components
    success_task = MockTaskComponent("success", TaskStatus.SUCCESS)
    failure_task = MockTaskComponent("failure", TaskStatus.FAILURE)
    running_task = MockTaskComponent("running", TaskStatus.RUNNING)
    
    print("Testing individual components...")
    assert await success_task.execute(agent_state) == TaskStatus.SUCCESS
    assert await failure_task.execute(agent_state) == TaskStatus.FAILURE
    assert await running_task.execute(agent_state) == TaskStatus.RUNNING
    print("✓ Basic components work correctly")


async def test_ordered_sequence():
    """Test OrderedSequenceComponent."""
    print("\n=== Testing OrderedSequenceComponent ===")
    
    agent_state = MockAgentState()
    
    # Test all success
    sequence = OrderedSequenceComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS),
        MockTaskComponent("task2", TaskStatus.SUCCESS),
        MockTaskComponent("task3", TaskStatus.SUCCESS),
    ])
    result = await sequence.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ All success sequence returns SUCCESS")
    
    # Test with failure
    sequence = OrderedSequenceComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS),
        MockTaskComponent("task2", TaskStatus.FAILURE),  # This should stop execution
        MockTaskComponent("task3", TaskStatus.SUCCESS),
    ])
    result = await sequence.execute(agent_state)
    assert result == TaskStatus.FAILURE
    print("✓ Sequence with failure returns FAILURE")
    
    # Test with running
    sequence = OrderedSequenceComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS),
        MockTaskComponent("task2", TaskStatus.RUNNING),  # This should stop execution
        MockTaskComponent("task3", TaskStatus.SUCCESS),
    ])
    result = await sequence.execute(agent_state)
    assert result == TaskStatus.RUNNING
    print("✓ Sequence with running returns RUNNING")


async def test_priority_selector():
    """Test PrioritySelectorComponent."""
    print("\n=== Testing PrioritySelectorComponent ===")
    
    agent_state = MockAgentState()
    
    # Test first success
    selector = PrioritySelectorComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS),  # This should succeed
        MockTaskComponent("task2", TaskStatus.FAILURE),
        MockTaskComponent("task3", TaskStatus.FAILURE),
    ])
    result = await selector.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ First success selector returns SUCCESS")
    
    # Test all failures
    selector = PrioritySelectorComponent([
        MockTaskComponent("task1", TaskStatus.FAILURE),
        MockTaskComponent("task2", TaskStatus.FAILURE),
        MockTaskComponent("task3", TaskStatus.FAILURE),
    ])
    result = await selector.execute(agent_state)
    assert result == TaskStatus.FAILURE
    print("✓ All failure selector returns FAILURE")
    
    # Test with running
    selector = PrioritySelectorComponent([
        MockTaskComponent("task1", TaskStatus.FAILURE),
        MockTaskComponent("task2", TaskStatus.RUNNING),  # This should return
        MockTaskComponent("task3", TaskStatus.SUCCESS),
    ])
    result = await selector.execute(agent_state)
    assert result == TaskStatus.RUNNING
    print("✓ Selector with running returns RUNNING")


async def test_parallel_execution():
    """Test ParallelExecutionComponent."""
    print("\n=== Testing ParallelExecutionComponent ===")
    
    agent_state = MockAgentState()
    
    # Test all success
    parallel = ParallelExecutionComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS, 0.1),
        MockTaskComponent("task2", TaskStatus.SUCCESS, 0.2),
        MockTaskComponent("task3", TaskStatus.SUCCESS, 0.1),
    ])
    result = await parallel.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ All success parallel returns SUCCESS")
    
    # Test with failure
    parallel = ParallelExecutionComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS, 0.1),
        MockTaskComponent("task2", TaskStatus.FAILURE, 0.1),  # This should cause failure
        MockTaskComponent("task3", TaskStatus.SUCCESS, 0.1),
    ])
    result = await parallel.execute(agent_state)
    assert result == TaskStatus.FAILURE
    print("✓ Parallel with failure returns FAILURE")
    
    # Test with running
    parallel = ParallelExecutionComponent([
        MockTaskComponent("task1", TaskStatus.SUCCESS, 0.1),
        MockTaskComponent("task2", TaskStatus.RUNNING, 0.1),  # This should cause running
        MockTaskComponent("task3", TaskStatus.SUCCESS, 0.1),
    ])
    result = await parallel.execute(agent_state)
    assert result == TaskStatus.RUNNING
    print("✓ Parallel with running returns RUNNING")


async def test_result_inverter():
    """Test ResultInverter."""
    print("\n=== Testing ResultInverter ===")
    
    agent_state = MockAgentState()
    
    # Test success inversion
    inverter = ResultInverter(MockTaskComponent("success", TaskStatus.SUCCESS))
    result = await inverter.execute(agent_state)
    assert result == TaskStatus.FAILURE
    print("✓ SUCCESS inverted to FAILURE")
    
    # Test failure inversion
    inverter = ResultInverter(MockTaskComponent("failure", TaskStatus.FAILURE))
    result = await inverter.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ FAILURE inverted to SUCCESS")
    
    # Test running unchanged
    inverter = ResultInverter(MockTaskComponent("running", TaskStatus.RUNNING))
    result = await inverter.execute(agent_state)
    assert result == TaskStatus.RUNNING
    print("✓ RUNNING unchanged")


async def test_action_repeater():
    """Test ActionRepeater."""
    print("\n=== Testing ActionRepeater ===")
    
    agent_state = MockAgentState()
    
    # Test success on first try
    repeater = ActionRepeater(MockTaskComponent("success", TaskStatus.SUCCESS), max_attempts=3)
    result = await repeater.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ Success on first attempt")
    
    # Test eventual success
    # This is harder to test with our current mock, but the logic is there
    
    # Test all failures
    repeater = ActionRepeater(MockTaskComponent("failure", TaskStatus.FAILURE), max_attempts=2)
    result = await repeater.execute(agent_state)
    assert result == TaskStatus.FAILURE
    print("✓ All attempts failed")
    
    # Test running
    repeater = ActionRepeater(MockTaskComponent("running", TaskStatus.RUNNING), max_attempts=3)
    result = await repeater.execute(agent_state)
    assert result == TaskStatus.RUNNING
    print("✓ Running on first attempt")


async def test_complex_composition():
    """Test complex composition of components."""
    print("\n=== Testing Complex Composition ===")
    
    agent_state = MockAgentState()
    
    # Create a complex task: Try multiple approaches, and if they all fail, try a fallback
    main_approaches = PrioritySelectorComponent([
        # First approach: sequence of actions
        OrderedSequenceComponent([
            MockTaskComponent("step1", TaskStatus.SUCCESS),
            MockTaskComponent("step2", TaskStatus.FAILURE),  # This will fail
        ]),
        # Second approach: parallel execution
        ParallelExecutionComponent([
            MockTaskComponent("parallel1", TaskStatus.SUCCESS),
            MockTaskComponent("parallel2", TaskStatus.SUCCESS),
        ]),
    ])
    
    # Wrap with repeater to try a few times
    repeated_main = ActionRepeater(main_approaches, max_attempts=2)
    
    # Final composition: main approach or fallback
    final_task = PrioritySelectorComponent([
        repeated_main,
        MockTaskComponent("fallback", TaskStatus.SUCCESS),  # Always succeeds
    ])
    
    result = await final_task.execute(agent_state)
    assert result == TaskStatus.SUCCESS
    print("✓ Complex composition works correctly")


async def test_atomic_actions_structure():
    """Test atomic action components structure (without actual browser)."""
    print("\n=== Testing Atomic Actions Structure ===")
    
    # Test that we can create the atomic action components
    click_action = ClickAction(index=5)
    type_action = TypeAction(index=3, text="hello world")
    goto_action = GoToUrlAction(url="https://example.com", new_tab=False)
    
    assert click_action.index == 5
    assert type_action.index == 3
    assert type_action.text == "hello world"
    assert goto_action.url == "https://example.com"
    assert goto_action.new_tab == False
    
    print("✓ Atomic action components can be created with correct parameters")
    
    # Test TaskExecutionContext
    # We can't test with real objects, but we can test the structure
    print("✓ TaskExecutionContext structure is available")


async def main():
    """Run all tests."""
    print("🚀 Starting Task Orchestration Framework Tests")
    
    try:
        await test_basic_components()
        await test_ordered_sequence()
        await test_priority_selector()
        await test_parallel_execution()
        await test_result_inverter()
        await test_action_repeater()
        await test_complex_composition()
        await test_atomic_actions_structure()
        
        print("\n✅ All tests passed! Task Orchestration Framework is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())