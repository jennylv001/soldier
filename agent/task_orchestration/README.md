# Task Orchestration Framework

A modular system for managing and executing sequences of actions in the browser automation agent. This framework provides composable components for building complex task execution flows that can replace the single-action decision-making loop.

## Architecture Overview

The Task Orchestration Framework follows a hierarchical component-based architecture where complex behaviors emerge from the composition of simple, reusable components.

### Core Components

#### Base Classes

- **`TaskComponent`**: Abstract base class for all task orchestration components
- **`TaskStatus`**: Enum defining execution results (`SUCCESS`, `FAILURE`, `RUNNING`)
- **`TaskExecutionContext`**: Context object providing access to browser functionality

#### Control Flow Components

- **`OrderedSequenceComponent`**: Executes child components sequentially
  - Returns `FAILURE` on first child failure
  - Returns `SUCCESS` only when all children succeed
  - Returns `RUNNING` if any child is still running

- **`PrioritySelectorComponent`**: Tries child components in priority order
  - Returns `SUCCESS` on first child success
  - Returns `FAILURE` only if all children fail
  - Returns `RUNNING` if any child is still running

- **`ParallelExecutionComponent`**: Runs all child components simultaneously
  - Returns `SUCCESS` only if all children succeed
  - Returns `FAILURE` if any child fails
  - Returns `RUNNING` if any child is still running

#### Action-Modifying Components

- **`ResultInverter`**: Flips the result of a single child component
  - `SUCCESS` becomes `FAILURE`
  - `FAILURE` becomes `SUCCESS`
  - `RUNNING` remains `RUNNING`

- **`ActionRepeater`**: Re-runs a single child component up to a specified number of times
  - Returns `SUCCESS` if child succeeds at least once
  - Returns `FAILURE` if child fails all attempts
  - Returns `RUNNING` if child is still running on current attempt

#### Atomic Action Components

- **`ClickAction`**: Wraps existing click functionality
- **`TypeAction`**: Wraps existing typing functionality
- **`GoToUrlAction`**: Wraps existing URL navigation functionality

All atomic components integrate with the existing action system through the `TaskExecutionContext`.

## Usage Examples

### Basic Task Creation

```python
from agent.task_orchestration import (
    OrderedSequenceComponent,
    ClickAction,
    TypeAction,
    GoToUrlAction,
    TaskExecutionContext,
)

# Create a simple form filling task
form_task = OrderedSequenceComponent([
    GoToUrlAction(url="https://example.com/form"),
    ClickAction(index=1),  # Click name field
    TypeAction(index=1, text="John Doe"),
    ClickAction(index=2),  # Click email field
    TypeAction(index=2, text="john@example.com"),
    ClickAction(index=3),  # Click submit button
])
```

### Complex Task Composition

```python
from agent.task_orchestration import (
    PrioritySelectorComponent,
    ParallelExecutionComponent,
    ActionRepeater,
    ResultInverter,
)

# Create a robust task with fallbacks and retries
robust_task = PrioritySelectorComponent([
    # Primary approach: try form filling
    ActionRepeater(
        OrderedSequenceComponent([
            GoToUrlAction(url="https://primary.com"),
            form_task,
        ]),
        max_attempts=3
    ),
    # Fallback approach: try alternative URL
    OrderedSequenceComponent([
        GoToUrlAction(url="https://backup.com"),
        form_task,
    ]),
])
```

### Integration with Agent System

```python
from agent.task_orchestration.integration_example import TaskOrchestrationIntegration

# Initialize integration
integration = TaskOrchestrationIntegration(controller, browser_session)

# Prepare agent state
prepared_state = integration.prepare_agent_state(agent_state)

# Execute task
result = await integration.execute_task(robust_task, prepared_state)
print(f"Task completed with result: {result}")
```

## Testing

The framework includes comprehensive tests that validate all components:

```bash
# Run basic functionality tests
cd /path/to/soldier
python agent/task_orchestration/test_basic_functionality.py

# Run integration example
python agent/task_orchestration/integration_example.py
```

### Test Coverage

- ✅ Basic component functionality
- ✅ Sequential execution (OrderedSequenceComponent)
- ✅ Priority selection (PrioritySelectorComponent)
- ✅ Parallel execution (ParallelExecutionComponent)
- ✅ Result inversion (ResultInverter)
- ✅ Action repetition (ActionRepeater)
- ✅ Complex task composition
- ✅ Atomic action structure validation

## Integration Guidelines

### Adding to Existing Agent

1. **Import the framework**:
   ```python
   from agent.task_orchestration import TaskOrchestrationIntegration
   ```

2. **Create integration instance**:
   ```python
   integration = TaskOrchestrationIntegration(
       controller=agent.settings.controller,
       browser_session=agent.browser_session
   )
   ```

3. **Build task components**:
   ```python
   task = integration.create_form_filling_task(form_data)
   ```

4. **Execute within agent loop**:
   ```python
   result = await integration.execute_task(task, agent_state)
   ```

### Backward Compatibility

The framework is designed to work alongside the existing agent architecture without breaking changes:

- ✅ No existing code modification required
- ✅ Uses existing action implementations
- ✅ Integrates with current browser session management
- ✅ Compatible with existing error handling

### Future Migration Path

The framework provides a clear migration path for replacing the single-action decision loop:

1. **Phase 1**: Use task orchestration for complex workflows (current implementation)
2. **Phase 2**: Integrate LLM-based task composition planning
3. **Phase 3**: Replace decision maker with task orchestration as primary execution engine

## Performance Characteristics

- **Sequential Execution**: Linear time complexity based on child count
- **Parallel Execution**: Improved throughput for independent operations
- **Memory Usage**: Minimal overhead per component
- **Error Handling**: Graceful degradation with proper error propagation

## Extension Points

The framework is designed for extensibility:

### Custom Task Components

```python
class CustomTaskComponent(TaskComponent):
    async def execute(self, agent_state: AgentState) -> TaskStatus:
        # Custom logic here
        return TaskStatus.SUCCESS
```

### Custom Control Flow

```python
class ConditionalComponent(TaskComponent):
    def __init__(self, condition_func, true_task, false_task):
        self.condition = condition_func
        self.true_task = true_task
        self.false_task = false_task
    
    async def execute(self, agent_state: AgentState) -> TaskStatus:
        if self.condition(agent_state):
            return await self.true_task.execute(agent_state)
        else:
            return await self.false_task.execute(agent_state)
```

## Benefits

### 🔄 Reliability
- Built-in retry mechanisms
- Fallback strategies
- Parallel execution for independent operations

### 🧩 Modularity
- Clear separation of concerns
- Easy to test individual components
- Reusable component library

### 📋 Composability
- Build complex workflows from simple components
- Mix and match control flow patterns
- Hierarchical task organization

### 🔧 Flexibility
- Support for sequential, parallel, and conditional logic
- Result inversion for negative conditions
- Extensible framework for new component types

### 🔗 Integration
- Works alongside existing agent architecture
- Backward compatible with current systems
- Uses existing action implementations

## License

This framework is part of the soldier browser automation agent and follows the same licensing terms.