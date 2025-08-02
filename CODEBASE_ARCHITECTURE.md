Module Analysis: agent
1. Core Identity

    Purpose: The agent module orchestrates an autonomous, LLM-driven browser agent lifecycle, coordinating perception, decision-making, actuation, state management, error recovery, and high-level supervision for robust, concurrent, web automation and task execution.

    Key Patterns: Orchestrator/Supervisor, Producer–Consumer, State Machine, Event-Driven, Command, Locking for atomicity, Dependency Injection, Adapter for LLM schema, CQRS (segregation of perception, decision, and actuation).

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "agent"
dependencies:
  external:
    - asyncio
    - logging
    - time
    - psutil
    - pydantic
    - collections
    - enum
    - typing
    - uuid_extensions
    - importlib.resources
    - datetime
    - json
    - os
    - tempfile
    - traceback
    - signal
  internal:
    - browser_use.agent.events
    - browser_use.agent.state_manager
    - browser_use.agent.views
    - browser_use.agent.settings
    - browser_use.agent.perception
    - browser_use.agent.decision_maker
    - browser_use.agent.actuator
    - browser_use.agent.prompts
    - browser_use.agent.supervisor
    - browser_use.agent.message_manager.service
    - browser_use.controller.service
    - browser_use.dom.history_tree_processor.service
    - browser_use.browser
    - browser_use.browser.views
    - browser_use.browser.session
    - browser_use.browser.types
    - browser_use.exceptions
    - browser_use.filesystem.file_system
    - browser_use.tokens.views
    - browser_use.llm.base
    - browser_use.llm.messages
    - browser_use.observability
    - browser_use.utils
public_api:
  - name: "Agent"
    type: "Class"
    properties:
      - "settings: AgentSettings"
      - "supervisor: Supervisor"
    methods:
      - "run(self) -> Awaitable[AgentHistoryList]: Launches a full agent run and returns history."
      - "pause(self): Pauses execution."
      - "resume(self): Resumes execution."
      - "stop(self): Stops execution."
      - "inject_human_guidance(self, text: str): Adds human feedback asynchronously."
      - "add_new_task(self, new_task: str): Changes the agent's current goal/task."
      - "close(self): Cleans up browser and agent resources."
      - "state (property): Returns current AgentState."
      - "browser_session (property): Returns the current BrowserSession."
      - "_get_agent_output_schema(self, include_done: bool = False): Internal LLM action schema helper."
      - "load_and_rerun(self, history_file: str, **kwargs): Load and replay agent history."
      - "rerun_history(self, history: AgentHistoryList, ...): Replay all steps in a history log."
data_contracts:
  - name: "AgentState"
    schema:
      agent_id: "str"
      task: "str"
      current_goal: "str"
      status: "AgentStatus"
      load_status: "LoadStatus"
      n_steps: "int"
      consecutive_failures: "int"
      last_error: "Optional[str]"
      accumulated_output: "Optional[str]"
      history: "AgentHistoryList"
      message_manager_state: "Dict[str, Any]"
      file_system_state: "Optional[Dict[str, Any]]"
      human_guidance_queue: "asyncio.Queue[str]"
  - name: "PerceptionOutput"
    schema:
      browser_state: "BrowserStateSummary"
      new_downloaded_files: "Optional[List[str]]"
      step_start_time: "float"
  - name: "Decision"
    schema:
      messages_to_llm: "List[BaseMessage]"
      llm_output: "Optional[AgentOutput]"
      action_results: "List[ActionResult]"
      step_metadata: "Optional[StepMetadata]"
      browser_state: "Optional[BrowserStateSummary]"
  - name: "ActuationResult"
    schema:
      action_results: "List[ActionResult]"
      llm_output: "Optional[AgentOutput]"
      browser_state: "Optional[BrowserStateSummary]"
      step_metadata: "StepMetadata"
  - name: "AgentHistory"
    schema:
      model_output: "Optional[AgentOutput]"
      result: "List[ActionResult]"
      state: "Optional[BrowserStateHistory]"
      metadata: "Optional[StepMetadata]"
  - name: "AgentOutput"
    schema:
      thinking: "Optional[str]"
      evaluation_previous_goal: "str"
      memory: "str"
      next_goal: "str"
      action: "List[ActionModel]"
  - name: "StepMetadata"
    schema:
      step_start_time: "float"
      step_end_time: "float"
      step_number: "int"
  - name: "ReflectionPlannerOutput"
    schema:
      memory_summary: "str"
      next_goal: "str"
      effective_strategy: "Optional[str]"
  - name: "AgentHistoryList"
    schema:
      history: "List[AgentHistory] | deque[AgentHistory]"
      usage: "Optional[UsageSummary]"

3. PHASE 1: Static Component & Dependency Graph Construction
Primary Logical Constructs

The agent module is composed of the following central classes:

    Agent (public API/shell): Provides a unified, type-safe, and async interface for running, pausing, resuming, stopping, and introspecting an agent; delegates execution to the Supervisor.

    Supervisor: Central orchestrator that wires together and supervises Perception, DecisionMaker, Actuator, and StateManager, each with their own distinct responsibilities, with internal queues and error handling.

    StateManager: Atomic, lock-protected state and history manager, providing safe concurrent access and bounded histories.

    Perception: Async, I/O-bound observer, continually gathering the environment state (BrowserSession), producing PerceptionOutput.

    DecisionMaker: Synchronous core, takes PerceptionOutput, prepares LLM prompts, decides next actions or reflection via LLM.

    Actuator: Async I/O executor, translates decisions into concrete browser actions via a Controller.

    Data contracts and models: Richly-typed data transfer objects (DTOs) like AgentState, Decision, ActuationResult, PerceptionOutput, AgentOutput, AgentHistory, AgentHistoryList, and others.

Direct Relationships and Internal Entry Points

    Agent instantiates and holds a Supervisor.

    Supervisor composes all other subsystems (Perception, DecisionMaker, Actuator, StateManager, MessageManager) and manages their lifecycles and orchestration.

    All cross-component communication leverages shared DTOs, message queues, and atomic property accessors (e.g., asyncio.Queue, StateManager.state).

    Public entry points for orchestration (Agent.run, Agent.pause, Agent.resume, Agent.stop, etc.) are mapped to supervisor methods, which in turn manipulate agent status via state transitions.

    Reflection, error, and recovery logic is governed by StateManager state and error transitions, and by control logic in Supervisor.

External Dependencies

    Relies on psutil for CPU load-based throttling.

    Heavy usage of pydantic for validation, type-safety, and runtime enforcement of DTO schemas.

    Delegates browser automation to types and classes from browser_use.browser.* and browser_use.controller.*.

    Interoperates with multiple LLM classes for action and reflection (llm, planner_llm).

Internal Data Model: Dependency Graph

    Agent (API surface)

        ⬇

    Supervisor

        ⬇

            Perception ⟶ produces PerceptionOutput

            DecisionMaker ⟶ consumes PerceptionOutput, produces Decision

            Actuator ⟶ consumes Decision, produces ActuationResult

            StateManager ⟶ central state object, atomic transitions, status/goal history

            MessageManager ⟶ prompt engineering, action and reflection message preparation

            Orchestrates async event loops via task groups for concurrent perception, decision, actuation, load monitoring, and error handling.

4. PHASE 2: Data & Control Flow Analysis
Primary Execution Paths

    Agent Lifecycle: User calls Agent.run(). This triggers the Supervisor.run() coroutine, which enters the orchestration loop:

        Perception Loop: Perception.run() monitors browser state, emits PerceptionOutput to a queue.

        Decision Loop: Consumes PerceptionOutput, prepares and sends LLM prompt, receives Decision, which is queued.

        Actuation Loop: Consumes Decision, actuates browser actions using Controller, emits ActuationResult, which finalizes the step and updates state/history.

        Reflection Path: If status changes to REFLECTING, DecisionMaker triggers the planner LLM for strategic reevaluation.

        Pausing/Stopping: All state transitions are managed by StateManager using atomic locks, driven by Supervisor.

Data Transformation & State Management

    The AgentState object (and its properties) is the nucleus for all persistent state: step number, current goal, status, failures, error messages, and the complete AgentHistoryList.

    State is always mutated inside lock-protected critical sections (using the _bulletproof_lock context).

    Step metadata (StepMetadata) records precise timing, enabling performance and duration tracking.

    Error and reflection handling escalate based on failure counts, feeding back into state transitions.

Interaction Patterns & Contracts

    Producer–Consumer: Perception, DecisionMaker, and Actuator interact via asyncio queues (perception_queue, decision_queue), decoupling I/O-bound and CPU-bound flows.

    Request/Response: LLM invocation and browser actuation conform to defined schemas, with rich DTOs for contract validation.

    Pub/Sub: Status and guidance updates propagate via atomic queues (e.g., human guidance).

    State Machine: Agent progresses through well-defined statuses (RUNNING, REFLECTING, FAILED, etc.) as a classic state machine.

    Error Handling: All flows include structured error reporting, retry logic, and history/logging.

5. PHASE 3: Integrity & Contradiction Audit
Dependency Validation

    All internal references to required types, methods, and fields are present within the analyzed code and expected imports. No circular dependencies detected at module-level.

    All key DTOs are fully defined and imported; dynamic LLM output schemas are built at runtime via helpers.

Contract Cohesion

    All public and internal method signatures (for primary flows) are consistent across instantiations and usage sites. DTO contract usage is type-checked and enforced with pydantic.

    Data passed between perception, decision, and actuation stages matches their signatures and expected fields (e.g., Decision.llm_output always checked for existence before access).

Execution Path Viability

    All main execution flows are covered for both normal operation and error/edge conditions. Agent status and error escalation are fully handled with atomic transitions.

    All exceptions in concurrent loops are caught, logged, and escalated via StateManager.

    CPU load shedding is robust: no deadlocks or race conditions due to atomic lock management and queue bounds.

    No observed dead-ends or infinite loops; terminal states are properly recognized and handled.

    Reflection and error recovery are only invoked when preconditions are met.

Architectural Consistency

    Clear separation of concerns between perception (environment sensing), decision (LLM-driven planning), and actuation (effecting environment change).

    State, event, and error handling patterns are uniformly applied.

    Adapter patterns are used for dynamic LLM output schemas, ensuring runtime flexibility.

    All lifecycle transitions are centralized in Supervisor, minimizing logic duplication and risk of inconsistent state.

ASSUMPTIONS & EXTERNAL DEPENDENCIES

    The module depends on correct implementation of browser session/control, LLM adapters, and file system abstractions as provided by external/internal modules (browser_use.browser, browser_use.controller, etc.).

    All referenced LLM models and schema adapters must implement expected async interfaces.

    Correct operation requires that all dependencies (especially for browser control and LLM I/O) are available and compatible with the provided APIs.
	

----------------------

Module Analysis: agent/message_manager
1. Core Identity

    Purpose: The agent/message_manager module constructs and maintains prompt-optimized, context-aware representations of agent history and state for large language model (LLM) interaction, manages history compression, and prepares both action and planner/reflection prompts for robust, iterative agent operation.

    Key Patterns: Adapter, Builder, State Holder, History Buffer, Factory Method.

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "agent/message_manager"
dependencies:
  external:
    - logging
    - itertools
    - typing
    - pydantic
  internal:
    - browser_use.agent.prompts
    - browser_use.agent.views
    - browser_use.llm.messages
    - browser_use.utils
    - .views
public_api:
  - name: "MessageManager"
    type: "Class"
    properties:
      - "task: str"
      - "system_message: SystemMessage"
      - "settings: MessageManagerSettings"
      - "state: MessageManagerState"
      - "file_system: FileSystem | None"
      - "sensitive_data: dict"
      - "last_input_messages: list"
    methods:
      - "__init__(self, task, system_message, settings, state=None, file_system=None, sensitive_data=None): Initializes the manager and bootstraps system message."
      - "add_new_task(self, new_task: str): Updates agent's task and appends system note."
      - "update_history_representation(self, agent_history: AgentHistoryList): Rebuilds history from canonical agent records."
      - "prepare_messages_for_llm(self, browser_state, current_goal, last_error, page_filtered_actions=None, agent_history_list=None): Returns full LLM message list."
      - "prepare_messages_for_planner(self, browser_state, current_plan, last_error): Returns LLM planner/reflection prompt."
      - "agent_history_description (property): Provides string serialization of history (with truncation)."
  - name: "HistoryItem"
    type: "Class"
    properties:
      - "step_number: int | None"
      - "evaluation_previous_goal: str | None"
      - "memory: str | None"
      - "next_goal: str | None"
      - "action_results: str | None"
      - "error: str | None"
      - "system_message: str | None"
    methods:
      - "to_string(self): Returns canonical text representation."
  - name: "MessageHistory"
    type: "Class"
    properties:
      - "system_message: BaseMessage | None"
      - "state_message: BaseMessage | None"
      - "consistent_messages: list[BaseMessage]"
    methods:
      - "get_messages(self): Returns current sequence of messages for LLM."
  - name: "MessageManagerSettings"
    type: "Class"
    properties:
      - "max_input_tokens: int"
      - "include_attributes: list[str]"
      - "message_context: Optional[str]"
      - "available_file_paths: List[str]"
      - "max_history_items: Optional[int]"
      - "max_history_for_planner: Optional[int]"
      - "images_per_step: int"
      - "use_vision: bool"
      - "use_vision_for_planner: bool"
      - "use_thinking: bool"
      - "image_tokens: int"
      - "recent_message_window_priority: int"
  - name: "MessageManagerState"
    type: "Class"
    properties:
      - "history: MessageHistory"
      - "tool_id: int"
      - "agent_history_items: list[HistoryItem]"
      - "read_state_description: str"
      - "local_system_notes: list[HistoryItem]"
data_contracts:
  - name: "LLMMessage"
    schema:
      role: "string"
      text: "string"
      images: "optional[list[string]]"
      state: "dict"
  - name: "HistoryItem"
    schema:
      step_number: "int | None"
      evaluation_previous_goal: "str | None"
      memory: "str | None"
      next_goal: "str | None"
      action_results: "str | None"
      error: "str | None"
      system_message: "str | None"
  - name: "MessageManagerSettings"
    schema:
      max_input_tokens: "int"
      include_attributes: "list[str]"
      message_context: "str | None"
      available_file_paths: "list[str]"
      max_history_items: "int | None"
      max_history_for_planner: "int | None"
      images_per_step: "int"
      use_vision: "bool"
      use_vision_for_planner: "bool"
      use_thinking: "bool"
      image_tokens: "int"
      recent_message_window_priority: "int"
  - name: "MessageManagerState"
    schema:
      history: "MessageHistory"
      tool_id: "int"
      agent_history_items: "list[HistoryItem]"
      read_state_description: "str"
      local_system_notes: "list[HistoryItem]"

3. PHASE 1: Static Component & Dependency Graph Construction

MessageManager is the module’s core construct, encapsulating the logic for converting agent state/history into LLM-friendly message sequences and summary blocks. It depends on DTOs and adapters defined in .views, as well as prompt factories (AgentMessagePrompt, PlannerPrompt). Internal relationships are based on composition: MessageManager holds a MessageManagerState, which aggregates a canonical MessageHistory and a bounded list of HistoryItem records. MessageManagerSettings parameterizes truncation, attributes, screenshot policies, and message formatting behaviors.

Public entry points are the MessageManager constructor and its prepare_messages_for_llm and prepare_messages_for_planner methods. State transitions occur via add_new_task (task update), update_history_representation (bulk state import), and _add_history_item (step append). All methods work by composing or transforming DTOs defined in the local scope or imported.

External dependencies are strictly for message structure (pydantic), message text formatting (logging, itertools), and prompt assembly utilities. All browser/file/LLM specifics are abstracted behind interfaces provided by higher layers or via dependency injection.
4. PHASE 2: Data & Control Flow Analysis

The principal execution paths all originate from MessageManager public methods. When an agent step occurs, update_history_representation is invoked with a fresh AgentHistoryList, clearing and reconstructing agent_history_items. Each AgentHistory entry is unpacked into a HistoryItem that canonically encodes LLM output, memory, action names, and error states. The prepare_messages_for_llm method orchestrates the creation of a context-appropriate user message (including screenshots if enabled) via the AgentMessagePrompt factory. The system and state messages are added to the canonical message sequence in MessageHistory.

Message serialization for the LLM is governed by truncation rules (see agent_history_description), ensuring bounded context and omission of oldest steps when limits are reached. The actual data objects—the LLM prompt messages—are held in BaseMessage subclasses, and are always filtered for sensitive data before dispatch to the LLM. Planner/reflection messages are constructed using the PlannerPrompt class, which synthesizes recent agent state into a JSON-only system prompt.

State is held within MessageManagerState, a pydantic structure providing history, the agent’s tool id, agent-readable action notes, and other persistent prompt-specific data. Mutation is atomic at the method level (each method clears or appends to state buffers in sequence).

The contract between MessageManager and the LLM is strictly defined: each message returned from prepare_messages_for_llm is a BaseMessage (with role, content, possibly images/state), and the message sequence always starts with the canonical system message. The module is adapter-oriented: any upstream change in history or settings can be rapidly reflected in the LLM prompt structure.
5. PHASE 3: Integrity & Contradiction Audit

All imports and data flow within the module are valid; every reference resolves to a construct within the same folder or an explicitly imported submodule. There are no circular dependencies. DTO field accesses and function/method signatures are consistent with their definitions, and all LLM-bound contracts are strictly enforced by pydantic models.

Error handling is robust: history mismatches, missing fields, or partial results are gracefully handled (e.g., zip_longest in action/result pairing, fallback strings in the absence of evaluation or memory). No observable risk of race conditions or deadlocks exists, as mutation occurs only in direct, synchronous method calls, and no shared mutable state is exposed outside the manager’s scope.

The design is architecturally consistent, applying the adapter and buffer patterns throughout. History truncation and prompt assembly rules are clear, and there is no contradiction in data lifecycle or memory policy. All state management occurs inside the MessageManagerState instance, which is replaced wholesale or mutated as an atomic unit.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    The system prompt templates (system_prompt.md, system_prompt_no_thinking.md) are static and are only read/interpreted elsewhere (not directly in this module).

    All BaseMessage, AgentHistory, and AgentMessagePrompt subclasses referenced are fully compliant with the browser_use LLM interface, as implied by the imports.

    Browser and file system state are always passed in from outside (never generated internally).



--------------------------------

Module Analysis: browser
1. Core Identity

    Purpose: The browser module defines the abstractions, state management, and connection logic for agent-driven Chromium-based browser automation, providing a unified interface for launching, configuring, and controlling browser sessions, profiles, and their runtime environments.

    Key Patterns: Adapter, Factory, State Holder, Resource Manager, Enum Registry, Data Transfer Object (DTO), Configuration Aggregator, Facade.

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "browser"
dependencies:
  external:
    - asyncio
    - atexit
    - dataclasses
    - enum
    - functools
    - logging
    - os
    - pathlib
    - pydantic
    - psutil
    - re
    - shutil
    - sys
    - tempfile
    - time
    - typing
    - uuid_extensions
    - playwright.async_api
    - patchright.async_api
    - httpx
  internal:
    - browser_use.browser.profile
    - browser_use.browser.session
    - browser_use.browser.types
    - browser_use.browser.views
    - browser_use.browser.extensions
    - browser_use.browser.context
    - browser_use.config
    - browser_use.observability
    - browser_use.utils
    - browser_use.dom.history_tree_processor.service
    - browser_use.dom.views
public_api:
  - name: "BrowserProfile"
    type: "Class"
    properties:
      - "all profile configuration fields: channel, user_data_dir, headless, stealth, traces_dir, etc."
    methods:
      - "detect_display_configuration(self): Detects display and viewport settings."
      - "model_copy(self, update=None): Returns a copy with updates."
      - "model_dump(self, exclude=None): Returns a dict representation."
  - name: "BrowserSession"
    type: "Class"
    properties:
      - "id: str"
      - "browser_profile: BrowserProfile"
      - "browser_pid: int | None"
      - "playwright: PlaywrightOrPatchright | None"
      - "browser: Browser | None"
      - "browser_context: BrowserContext | None"
      - "agent_current_page: Page | None"
      - "human_current_page: Page | None"
      - "initialized: bool"
    methods:
      - "start(self) -> Awaitable[Self]: Bootstraps or connects a browser session."
      - "stop(self, _hint=''): Shuts down the browser session and all resources."
      - "kill(self): Forcibly terminates the browser, regardless of keep_alive."
      - "close(self): Deprecated alias for stop."
      - "model_copy(self, **kwargs) -> Self: Shallow-copy session without closing resources."
      - "setup_playwright(self): Initializes Playwright/Patchright global state."
      - "setup_browser_via_passed_objects(self): Connects to existing Playwright objects."
      - "setup_browser_via_browser_pid(self): Connects via existing browser PID."
      - "setup_browser_via_wss_url(self): Connects via WebSocket endpoint."
      - "setup_browser_via_cdp_url(self): Connects via Chrome DevTools Protocol."
      - "setup_new_browser_context(self): Launches a new browser context."
      - "_take_screenshot_hybrid(self, clip=None) -> str: Screenshot utility."
      - "__aenter__/__aexit__: Async context manager for lifecycle."
      - "__del__: Cleanup."
  - name: "Browser"
    type: "TypeAlias"
    description: "Alias for BrowserSession."
  - name: "BrowserConfig"
    type: "TypeAlias"
    description: "Alias for BrowserProfile."
  - name: "BrowserContext"
    type: "TypeAlias"
    description: "Alias for BrowserSession."
  - name: "BrowserContextConfig"
    type: "TypeAlias"
    description: "Alias for BrowserProfile."
  - name: "TabInfo"
    type: "Class"
    properties:
      - "page_id: int"
      - "url: str"
      - "title: str"
      - "parent_page_id: int | None"
  - name: "PageInfo"
    type: "Class"
    properties:
      - "viewport_width: int"
      - "viewport_height: int"
      - "page_width: int"
      - "page_height: int"
      - "scroll_x: int"
      - "scroll_y: int"
      - "pixels_above: int"
      - "pixels_below: int"
      - "pixels_left: int"
      - "pixels_right: int"
  - name: "BrowserStateSummary"
    type: "Class"
    properties:
      - "url: str"
      - "title: str"
      - "tabs: list[TabInfo]"
      - "screenshot: str | None"
      - "page_info: PageInfo | None"
      - "pixels_above: int"
      - "pixels_below: int"
      - "browser_errors: list[str]"
  - name: "BrowserStateHistory"
    type: "Class"
    properties:
      - "url: str"
      - "title: str"
      - "tabs: list[TabInfo]"
      - "interacted_element: list[DOMHistoryElement | None] | list[None]"
      - "screenshot: str | None"
    methods:
      - "to_dict(self) -> dict[str, Any]"
  - name: "TargetClosedError"
    type: "Exception"
    description: "Unified error for browser target closure (Patchright or Playwright)."
data_contracts:
  - name: "BrowserProfile"
    schema:
      channel: "BrowserChannel"
      user_data_dir: "str | None"
      headless: "bool"
      stealth: "bool"
      traces_dir: "str | None"
      ... (and all other config fields)
  - name: "BrowserSession"
    schema:
      id: "str"
      browser_profile: "BrowserProfile"
      browser_pid: "int | None"
      playwright: "PlaywrightOrPatchright | None"
      browser: "Browser | None"
      browser_context: "BrowserContext | None"
      agent_current_page: "Page | None"
      human_current_page: "Page | None"
      initialized: "bool"
  - name: "BrowserStateSummary"
    schema:
      url: "str"
      title: "str"
      tabs: "list[TabInfo]"
      screenshot: "str | None"
      page_info: "PageInfo | None"
      pixels_above: "int"
      pixels_below: "int"
      browser_errors: "list[str]"
  - name: "BrowserStateHistory"
    schema:
      url: "str"
      title: "str"
      tabs: "list[TabInfo]"
      interacted_element: "list[DOMHistoryElement | None] | list[None]"
      screenshot: "str | None"

3. PHASE 1: Static Component & Dependency Graph Construction

The module aggregates a hierarchy of objects representing the browser’s configuration, session state, and real-time environment. BrowserProfile encapsulates all launch and runtime parameters for a browser instance (channel, user data dir, headless/stealth modes, viewport, etc.), serving as the source of truth for session configuration. BrowserSession orchestrates the lifecycle of a browser instance, including launching, connecting, context management, error handling, and resource cleanup, and composes its state by referencing BrowserProfile, various Playwright/Patchright handles, and real-time properties. Type aliases (Browser, BrowserConfig, BrowserContext, BrowserContextConfig) offer polymorphic facades for compatibility across the agent and DOM layers. DTOs such as TabInfo, PageInfo, BrowserStateSummary, and BrowserStateHistory serialize state and context for LLMs or downstream consumers.

Direct relationships include composition (sessions own profiles, state summaries, and handles), and dynamic references (e.g., async start/stop/kill, context switching, event handling). Internal entry points include all public methods on BrowserSession, as well as DTO factories. Imports span pydantic models, enum definitions, resource managers, and Playwright/Patchright/HTTPX types for interoperability.
4. PHASE 2: Data & Control Flow Analysis

Session lifecycle begins by instantiating a BrowserSession with an optional BrowserProfile. Upon invoking start, the session determines whether to launch or connect, detecting the proper configuration, resolving environment, and acquiring Playwright/Patchright objects. The session attempts connection by multiple prioritized vectors: passed Playwright/Page/Browser objects, PID, WSS, CDP URLs, or, as a fallback, launching a new browser with the current profile.

All mutable state is centrally held in the BrowserSession instance, protected by pydantic’s assignment controls and explicit event loop checks. Session teardown (stop/kill/close) cascades cleanup, releasing resources and closing browser contexts, and—if the session is owner—killing browser processes. Additional resource control is enforced in __del__, ensuring no zombie processes persist after GC. Async context management (__aenter__/__aexit__) provides safe, atomic lifecycle handling for concurrent agent operation.

DTOs serve as adapters and serialization points. BrowserStateSummary captures LLM-facing browser state, including DOM snapshots, tab info, and error logs, while BrowserStateHistory serializes session state for replay or logging. Type aliases and cross-module imports allow seamless polymorphism between Patchright and Playwright implementations.

Interaction patterns are classic request/response for browser commands, observer patterns for event handling, and resource management for process/lifecycle control. API contracts are enforced via pydantic, type annotations, and structured DTOs.
5. PHASE 3: Integrity & Contradiction Audit

All class and type references resolve correctly within the module, with type aliases providing clean adaptation between Patchright and Playwright implementations. There are no circular dependencies. All public method signatures match usage and documentation. DTO and configuration classes are consistent across submodules, ensuring no mismatches during instantiation or serialization.

Lifecycle edge cases (such as reconnection, unexpected browser exit, process cleanup, orphan contexts) are handled with explicit retries, cascading teardown, and resource auditing. Logging is comprehensive, reducing risk of dead-end failure or resource leaks. There are no architectural contradictions: state, configuration, and resource management are all modular and uniform.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    Playwright and Patchright are assumed to be present and API-compatible; the types system is designed to abstract over either.

    All LLM or agent modules consuming this API are expected to use DTOs and factory methods as specified.

    Browser process and context management rely on correct environment setup (valid Chrome/Chromium binaries, valid networking for remote protocols).
	
	
-----------------------------------

Module Analysis: controller
1. Core Identity

    Purpose: The controller module manages the registration, normalization, and execution of browser agent actions, enabling structured, schema-driven, and context-aware orchestration of both atomic and batch browser automation tasks for an LLM-powered agent.

    Key Patterns: Registry, Command, Adapter, Decorator, Strategy, Factory, Dependency Injection, Context Object, DTO (Data Transfer Object).

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "controller"
dependencies:
  external:
    - asyncio
    - functools
    - inspect
    - logging
    - re
    - typing
    - pydantic
  internal:
    - browser_use.browser
    - browser_use.browser.types
    - browser_use.controller.registry.service
    - browser_use.controller.registry.views
    - browser_use.agent.views
    - browser_use.filesystem.file_system
    - browser_use.llm.base
    - browser_use.observability
    - browser_use.utils
    - browser_use.telemetry.service
public_api:
  - name: "Controller"
    type: "Class"
    properties:
      - "registry: Registry"
      - "display_files_in_done_text: bool"
    methods:
      - "__init__(self, exclude_actions: Optional[list[str]] = None, output_model: Optional[type[T]] = None, display_files_in_done_text: bool = True): Initializes Controller and its action registry."
      - "action(self, description: str, **kwargs): Decorator for registering custom actions."
      - "use_structured_output_action(self, output_model: type[T]): Registers a custom structured done action."
      - "act(self, action: ActionModel, browser_session: BrowserSession, ...): Executes a single action using the registry."
      - "multi_act(self, actions: list[ActionModel], browser_session: BrowserSession, ...): Executes a sequence of actions with optional UI stability check."
  - name: "Registry"
    type: "Class"
    properties:
      - "registry: ActionRegistry"
      - "telemetry: ProductTelemetry"
      - "exclude_actions: list[str]"
    methods:
      - "action(description: str, ...): Decorator for registering actions."
      - "execute_action(self, action_name: str, params: dict, ...): Executes a registered action with simplified parameter handling."
      - "create_action_model(self, ...): Builds a Union of available action models for LLM tool-calling."
      - "get_prompt_description(self, ...): Returns a description of actions for prompting."
      - "_normalize_action_function_signature(self, ...): Normalizes action signature to accept only kwargs."
      - "_replace_sensitive_data(self, params, sensitive_data, current_url): Recursively replaces sensitive data placeholders in params."
      - "_get_special_param_types(self): Returns expected types for controller-injected parameters."
  - name: "ActionRegistry"
    type: "Class"
    properties:
      - "actions: dict[str, RegisteredAction]"
    methods:
      - "get_prompt_description(self, page: Page | None = None) -> str"
      - "_match_domains(domains: list[str] | None, url: str) -> bool"
      - "_match_page_filter(page_filter: Callable | None, page: Page) -> bool"
  - name: "RegisteredAction"
    type: "Class"
    properties:
      - "name: str"
      - "description: str"
      - "function: Callable"
      - "param_model: type[BaseModel]"
      - "domains: list[str] | None"
      - "page_filter: Callable[[Page], bool] | None"
    methods:
      - "prompt_description(self) -> str"
  - name: "ActionModel"
    type: "Class"
    methods:
      - "get_index(self) -> int | None"
      - "set_index(self, index: int)"
      - "model_dump(self, **kwargs)"
  - name: "SpecialActionParameters"
    type: "Class"
    properties:
      - "context: Any | None"
      - "browser_session: BrowserSession | None"
      - "browser: BrowserSession | None"
      - "browser_context: BrowserSession | None"
      - "page: Page | None"
      - "page_extraction_llm: BaseChatModel | None"
      - "file_system: FileSystem | None"
      - "available_file_paths: list[str] | None"
      - "has_sensitive_data: bool"
    methods:
      - "get_browser_requiring_params(cls) -> set[str]"
data_contracts:
  - name: "ActionModel"
    schema:
      fields: "Dynamically generated per action; all registered action parameters as BaseModel fields."
  - name: "DoneAction"
    schema:
      text: "str"
      success: "bool"
      files_to_display: "Optional[List[str]]"
  - name: "StructuredOutputAction"
    schema:
      success: "bool"
      data: "T"
  - name: "RegisteredAction"
    schema:
      name: "str"
      description: "str"
      function: "Callable"
      param_model: "type[BaseModel]"
      domains: "list[str] | None"
      page_filter: "Callable[[Page], bool] | None"

3. PHASE 1: Static Component & Dependency Graph Construction

The controller module centers on the Controller class, which composes a Registry for action registration and lookup. The Registry manages an ActionRegistry, which stores RegisteredAction records for each available agent action. Actions are registered via the Controller.action() or Registry.action() decorator, which normalizes function signatures (using _normalize_action_function_signature) and enforces pydantic parameter models. The act and multi_act methods on Controller serve as public entry points for action execution, invoking Registry.execute_action or sequencing multiple actions, with strict error handling and context parameter injection. DTOs (ActionModel, DoneAction, StructuredOutputAction, etc.) define all data contracts for LLM-driven control, with each action receiving validated, schema-driven parameters.

Direct relationships include dependency injection for browser and file system context, dynamic output model wiring, and UI stability checks for batched actions. All submodules—browser_use.controller.registry.service, browser_use.controller.registry.views, and the DTO sublayer—are composed via imports and runtime registry assembly.
4. PHASE 2: Data & Control Flow Analysis

All agent actions are registered at runtime, either as default or user-defined, with their function signatures normalized to accept validated pydantic parameter models plus a suite of "special" context parameters (e.g., browser session, page, file system, etc.). Execution begins when the agent invokes Controller.act, which receives an ActionModel instance and forwards it to the registry for lookup, validation, and async invocation. Each action is executed in a coroutine context, and all parameters (action + context) are explicitly passed and checked, ensuring deterministic invocation and robust error handling. Sensitive data placeholders are recursively replaced at runtime prior to execution.

The multi_act method executes action batches, maintaining page and DOM state hashes for UI stability; if elements mutate during execution, the process halts with a descriptive message. Results from each action are aggregated into an array of ActionResult DTOs, with early exit on error or completion signal (is_done). DTO contracts are strictly enforced at every stage via pydantic validation and the dynamic creation of per-action parameter schemas.

State management is isolated: the controller maintains no persistent state except for the runtime action registry and action DTOs. All transient state (input models, output models, execution context, UI hashes, etc.) is constructed and consumed within the call stack. Action registration and lookup are handled dynamically at runtime.

Interaction patterns include classic command dispatch, registry lookup, async task execution, context injection, and error bubbling. Contracts are governed by DTO schemas, action decorators, and explicit type validation. Prompt descriptions for LLM prompting are generated by introspecting registered actions and their input/output contracts.
5. PHASE 3: Integrity & Contradiction Audit

All internal and external references resolve; class and function signatures are consistent across registration, lookup, and execution. Decorators enforce input contract normalization and block ambiguous **kwargs usage. Registry and context wiring prevent accidental leakage or contamination of execution state. Parameter mismatches or missing context objects are handled with descriptive exceptions at runtime. Sensitive data replacement and domain filtering mechanisms are robust and precisely delimited.

There are no circular dependencies in the registry-service-view contract chain. The registration and invocation machinery is uniform and fully adapter-driven. The architectural pattern—registry-lookup-invoke—is never violated, and context passing is consistently enforced throughout. The module is architecturally and semantically consistent.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    Agent actions are assumed to be idempotent and safe to retry.

    The browser session, file system, and LLM base classes provided as context must conform to required async interfaces.

    Downstream consumers (agent, browser, LLM) are responsible for maintaining contract compliance when registering or invoking new actions.
	
	
--------------------------------

Module Analysis: dom
1. Core Identity

    Purpose: The dom module and its sibling submodules implement the full pipeline for DOM (Document Object Model) acquisition, parsing, normalization, history tracking, clickable element hashing, and browser agent interaction, enabling LLM-driven and programmatic manipulation and interpretation of web page structures.

    Key Patterns: Composite, DTO (Data Transfer Object), Visitor, Builder, Hash Adapter, Decorator, Registry, Context Object.

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "dom"
dependencies:
  external:
    - asyncio
    - logging
    - importlib.resources
    - dataclasses
    - functools
    - typing
    - pydantic
    - hashlib
    - urllib.parse
    - anyio
    - json
    - os
    - time
    - pyperclip
    - tiktoken
  internal:
    - browser_use.browser
    - browser_use.browser.types
    - browser_use.dom.views
    - browser_use.dom.utils
    - browser_use.dom.history_tree_processor.service
    - browser_use.dom.history_tree_processor.view
    - browser_use.dom.clickable_element_processor.service
    - browser_use.dom.dom_tree
    - browser_use.filesystem.file_system
    - browser_use.agent.prompts
    - browser_use.utils
public_api:
  - name: "DomService"
    type: "Class"
    properties:
      - "page: Page"
      - "logger: logging.Logger"
    methods:
      - "get_clickable_elements(self, ...): Returns DOMState with clickable elements."
      - "get_cross_origin_iframes(self): Returns list of cross-origin iframe URLs."
      - "_build_dom_tree(self, ...): Builds and parses DOM using injected JS."
      - "_construct_dom_tree(self, ...): Transforms JS output to Python DOM tree."
      - "_parse_node(self, ...): Converts node dicts to DOM*Node objects."
  - name: "ClickableElementProcessor"
    type: "Class"
    methods:
      - "get_clickable_elements_hashes(dom_element: DOMElementNode) -> set[str]"
      - "get_clickable_elements(dom_element: DOMElementNode) -> list[DOMElementNode]"
      - "hash_dom_element(dom_element: DOMElementNode) -> str"
  - name: "HistoryTreeProcessor"
    type: "Class"
    methods:
      - "convert_dom_element_to_history_element(dom_element: DOMElementNode) -> DOMHistoryElement"
      - "find_history_element_in_tree(dom_history_element: DOMHistoryElement, tree: DOMElementNode) -> DOMElementNode | None"
      - "compare_history_element_and_dom_element(dom_history_element: DOMHistoryElement, dom_element: DOMElementNode) -> bool"
      - "_hash_dom_history_element(dom_history_element: DOMHistoryElement) -> HashedDomElement"
      - "_hash_dom_element(dom_element: DOMElementNode) -> HashedDomElement"
  - name: "DOMBaseNode"
    type: "Class"
    properties:
      - "is_visible: bool"
      - "parent: Optional[DOMElementNode]"
    methods:
      - "__json__(self) -> dict"
  - name: "DOMTextNode"
    type: "Class"
    properties:
      - "text: str"
      - "type: str"
      - "parent: Optional[DOMElementNode]"
    methods:
      - "__json__(self) -> dict"
  - name: "DOMElementNode"
    type: "Class"
    properties:
      - "tag_name: str"
      - "xpath: str"
      - "attributes: dict[str, str]"
      - "children: list[DOMBaseNode]"
      - "is_interactive: bool"
      - "is_top_element: bool"
      - "is_in_viewport: bool"
      - "shadow_root: bool"
      - "highlight_index: int | None"
      - "viewport_coordinates: CoordinateSet | None"
      - "page_coordinates: CoordinateSet | None"
      - "viewport_info: ViewportInfo | None"
      - "is_new: bool | None"
    methods:
      - "__json__(self) -> dict"
      - "clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str"
      - "get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str"
  - name: "DOMHistoryElement"
    type: "Class"
    properties:
      - "tag_name: str"
      - "xpath: str"
      - "highlight_index: int | None"
      - "entire_parent_branch_path: list[str]"
      - "attributes: dict[str, str]"
      - "shadow_root: bool"
      - "css_selector: str | None"
      - "page_coordinates: CoordinateSet | None"
      - "viewport_coordinates: CoordinateSet | None"
      - "viewport_info: ViewportInfo | None"
    methods:
      - "to_dict(self) -> dict"
  - name: "HashedDomElement"
    type: "Class"
    properties:
      - "branch_path_hash: str"
      - "attributes_hash: str"
      - "xpath_hash: str"
  - name: "CoordinateSet"
    type: "Class"
    properties:
      - "top_left: Coordinates"
      - "top_right: Coordinates"
      - "bottom_left: Coordinates"
      - "bottom_right: Coordinates"
      - "center: Coordinates"
      - "width: int"
      - "height: int"
  - name: "ViewportInfo"
    type: "Class"
    properties:
      - "scroll_x: int | None"
      - "scroll_y: int | None"
      - "width: int"
      - "height: int"
data_contracts:
  - name: "DOMState"
    schema:
      element_tree: "DOMElementNode"
      selector_map: "dict[int, DOMElementNode]"
  - name: "SelectorMap"
    schema:
      mapping: "int -> DOMElementNode"
  - name: "HistoryTree"
    schema:
      elements: "list[DOMHistoryElement]"
  - name: "HashedDomElement"
    schema:
      branch_path_hash: "str"
      attributes_hash: "str"
      xpath_hash: "str"
  - name: "DOMHistoryElement"
    schema:
      tag_name: "str"
      xpath: "str"
      highlight_index: "int | None"
      entire_parent_branch_path: "list[str]"
      attributes: "dict[str, str]"
      shadow_root: "bool"
      css_selector: "str | None"
      page_coordinates: "CoordinateSet | None"
      viewport_coordinates: "CoordinateSet | None"
      viewport_info: "ViewportInfo | None"

3. PHASE 1: Static Component & Dependency Graph Construction

The module is structured around composable node and DTO types (DOMBaseNode, DOMElementNode, DOMTextNode, DOMHistoryElement, HashedDomElement), resource-managed DOM pipeline services (DomService, ClickableElementProcessor, HistoryTreeProcessor), and orchestration utilities (CoordinateSet, ViewportInfo). DomService is the primary orchestrator, taking a browser page and producing parsed, validated, and structured DOM trees, clickable element sets, and selector maps via injected JavaScript and post-processing routines. ClickableElementProcessor isolates all logic for identifying, traversing, and hashing clickable nodes. HistoryTreeProcessor provides conversion between current DOM and history records, as well as element matching and hash comparison. extraction.py and process_dom.py offer test harnesses for pipeline invocation. DTOs are unified by hash adapters and context-aware serialization.

Direct relationships are strictly compositional: nodes reference parents and children; processors and services operate on or emit DTOs. All methods either transform, hash, or traverse node graphs. No inheritance beyond base dataclasses for node polymorphism. External dependencies are limited to JS resource injection, async browser control, file I/O, and minor third-party utilities (e.g., tiktoken, pyperclip for test harnesses).
4. PHASE 2: Data & Control Flow Analysis

The canonical execution flow starts by instantiating a DomService with a browser Page and logger, which can then asynchronously retrieve clickable elements or DOM state. _build_dom_tree injects and executes bundled JS (via importlib.resources), returning a raw structure which is parsed into a canonical tree (DOMElementNode) and selector map. Clickable elements are extracted and indexed by highlight, supporting direct agent interaction. The processor classes can traverse and hash any DOM tree, supporting rapid delta detection and action correlation.

History management is accomplished by converting live nodes to DOMHistoryElement records (including branch paths, selectors, coordinates, and hashes) via HistoryTreeProcessor. Comparison, matching, and deduplication of DOM nodes is performed by deep hash and attribute analysis. All hashing logic (attribute, xpath, parent-branch) is centralized in ClickableElementProcessor and HistoryTreeProcessor, using SHA-256 for stability.

State is always stored in and mutated on per-instance DTOs. No mutable global state exists. Selector maps and tree roots are ephemeral, computed per invocation and tied to a single browser state. All I/O, including clipboard or file operations, is isolated to test harnesses or explicit pipeline calls.

Contracts are formalized via pydantic, dataclasses, and type annotations; methods expect and return strongly typed node or DTO objects. String formatting, attribute selection, and attribute deduplication are deeply optimized for LLM token usage and context windows. Prompt-friendly string representations are generated for agent consumption.
5. PHASE 3: Integrity & Contradiction Audit

All imports and relationships resolve; no circular dependencies detected between node types, services, or processors. All cross-class method calls, including async flows, type checks, and hash comparisons, are signature-aligned and consistently implemented. DTOs maintain strict field congruence; node-to-history or node-to-string transformations match their respective schemas and contracts.

Node parent/child relationships and selector map constructions are robust against missing or cyclic nodes. Exception handling exists for failed JS evals or parsing errors, and all errors are logged or raised with clear tracebacks. Test harnesses, while rich in utility, are side-effect free for the module core.

No contradictions or architectural inconsistencies are present: node composition, pipeline invocation, and hash-based identification are uniform throughout. State and history management are centralized, and serialization is deterministic and lossless.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    Browser session, Playwright/Patchright types, and JS resources (e.g., buildDomTree.js) must be present and API-compatible.

    File system and clipboard integration are limited to test harnesses.

    Agent and LLM consumers must use only the contractually exposed DTOs and string representations.
	
	
	
-------------------------------------

Module Analysis: /filesystem, /integrations/gmail/, /llm/google/, /mcp, /tokens
1. Core Identity

    Purpose:
    This suite orchestrates the major infrastructure for: a) local and agent-facing file system management with in-memory and disk-backed state; b) Gmail integration for agent-accessible email authentication and retrieval; c) Google LLM (Large Language Model) and Gemini API chat integration, including message serialization, invocation, and retry logic; d) MCP (Model Context Protocol) server/client/tool registry for tool discovery, action wrapping, and tool execution; e) Token cost and usage tracking, including pricing fetching and per-LLM usage breakdowns.

    Key Patterns:
    Abstract Base, Service-Oriented, Adapter, Singleton (for resource/service instances), Factory (for file/document types and tool wrappers), DTO (Data Transfer Object), Registry, Proxy (for action/method wrapping), Observer (for usage/cost tracking), Command.

2. Component & API Surface (YAML)

module: "filesystem_gmail_llm_mcp_tokens"
dependencies:
  external:
    - asyncio
    - aiofiles
    - httpx
    - pydantic
    - dotenv
    - google-auth
    - google-auth-oauthlib
    - google-api-python-client
    - mcp
    - base64
    - logging
    - json
    - dataclasses
    - typing
    - os
    - pathlib
    - time
    - http
    - shutil
    - tempfile
    - concurrent.futures
  internal:
    - browser_use.agent.views
    - browser_use.controller.service
    - browser_use.controller.registry.service
    - browser_use.telemetry
    - browser_use.config
    - browser_use.llm.base
    - browser_use.llm.exceptions
    - browser_use.llm.messages
    - browser_use.llm.schema
    - browser_use.llm.views
    - browser_use.llm.google.serializer
    - browser_use.utils
    - browser_use.tokens.views
public_api:
  - name: "FileSystem"
    type: "Class"
    properties:
      - "base_dir: Path"
      - "files: dict"
    methods:
      - "get_allowed_extensions(self) -> list[str]"
      - "list_files(self) -> list[str]"
      - "get_file(self, full_filename: str) -> BaseFile | None"
      - "read_file(self, full_filename: str, external_file: bool = False) -> Awaitable[str]"
      - "write_file(self, full_filename: str, content: str) -> Awaitable[str]"
      - "display_file(self, full_filename: str) -> str | None"
  - name: "GmailService"
    type: "Class"
    properties:
      - "config_dir: Path"
      - "creds: Credentials | None"
      - "service: Any | None"
      - "_authenticated: bool"
    methods:
      - "is_authenticated(self) -> bool"
      - "authenticate(self) -> Awaitable[bool]"
      - "get_recent_emails(self, max_results: int = 10, query: str = '', time_filter: str = '1h') -> Awaitable[list[dict[str, Any]]]"
  - name: "ChatGoogle"
    type: "Class"
    properties:
      - "model: str"
      - "temperature: float | None"
      - "api_key: str | None"
      - "vertexai: bool | None"
    methods:
      - "ainvoke(self, messages: list[BaseMessage], output_format: type[T] | None = None) -> Awaitable[ChatInvokeCompletion[T] | ChatInvokeCompletion[str]]"
      - "get_client(self) -> genai.Client"
  - name: "GoogleMessageSerializer"
    type: "Class"
    methods:
      - "serialize_messages(messages: list[BaseMessage]) -> tuple[ContentListUnion, str | None]"
  - name: "MCPClient"
    type: "Class"
    properties:
      - "server_name: str"
      - "command: str"
      - "args: list[str]"
      - "session: ClientSession | None"
    methods:
      - "connect(self) -> Awaitable[None]"
      - "disconnect(self) -> Awaitable[None]"
      - "register_to_controller(self, controller: Controller, tool_filter: list[str] | None = None, prefix: str | None = None) -> Awaitable[None]"
  - name: "MCPToolWrapper"
    type: "Class"
    methods:
      - "connect(self) -> Awaitable[None]"
  - name: "BrowserUseServer"
    type: "Class"
    properties:
      - "server: Server"
      - "controller: Controller | None"
    methods:
      - "_execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Awaitable[str]"
  - name: "TokenCost"
    type: "Class"
    properties:
      - "include_cost: bool"
      - "usage_history: list[TokenUsageEntry]"
      - "registered_llms: dict[str, BaseChatModel]"
    methods:
      - "initialize(self) -> Awaitable[None]"
      - "get_model_pricing(self, model_name: str) -> Awaitable[ModelPricing | None]"
      - "calculate_cost(self, model: str, usage: ChatInvokeUsage) -> Awaitable[TokenCostCalculated | None]"
      - "add_usage(self, model: str, usage: ChatInvokeUsage) -> TokenUsageEntry"
data_contracts:
  - name: "BaseFile"
    schema:
      name: "str"
      content: "str"
  - name: "FileSystemState"
    schema:
      files: "dict[str, dict[str, Any]]"
      base_dir: "str"
      extracted_content_count: "int"
  - name: "GetRecentEmailsParams"
    schema:
      keyword: "str"
      max_results: "int"
  - name: "ActionResult"
    schema:
      extracted_content: "str"
      long_term_memory: "str"
  - name: "TokenUsageEntry"
    schema:
      model: "str"
      timestamp: "datetime"
      usage: "ChatInvokeUsage"

PHASE 1: Static Component & Dependency Graph Construction

The FileSystem submodule defines in-memory and disk-backed file types (MarkdownFile, TxtFile, JsonFile, CsvFile) via a BaseFile abstract base, with the primary FileSystem service managing file objects keyed by filename. State is serialized through FileSystemState.
Gmail integration exposes actions via actions.py (for agent registry) and a GmailService for OAuth authentication, token refresh, and async email retrieval, using the Google API client and pydantic DTOs for inputs.
Google LLM/Gemini integration centers on the ChatGoogle dataclass (LLM proxy), invoking the Google genai client, using a GoogleMessageSerializer for content/schema adaption, and supporting schema-optimized output parsing, retries, and usage reporting.
MCP: Both server and client abstractions are present. MCPClient connects to an MCP server, discovers tools, and dynamically registers them as agent actions, using JSON Schema-to-Pydantic model conversion. The server registers tools (browser/automation) and dispatches tool execution, with telemetric instrumentation at every step.
Tokens: The TokenCost service tracks LLM usage, fetches and caches model pricing from the LiteLLM repository, and calculates/report costs via pydantic DTOs and async HTTP/file IO.
All modules rely on Pydantic for contracts, and use a dense, logging-driven service architecture with explicit async boundaries.
PHASE 2: Data & Control Flow Analysis

FileSystem methods are invoked via agent or service calls, maintaining state in a files dict, with I/O operations isolated via async methods. File operations mutate in-memory objects and sync to disk under a dedicated data directory.
Gmail actions are registered to a controller, exposing a parameterized action (get_recent_emails) that triggers authentication (with token file, OAuth, or direct access token), then queries Gmail with a keyword and time filter, returning formatted email DTOs.
LLM invocations route through ChatGoogle, which serializes conversation context, applies system instructions, then executes Gemini/gemai API calls with robust retry logic. Usage metadata is parsed and can be recorded via TokenCost, which on registration of an LLM, patches its invocation pipeline to log usage, update in-memory state, and if enabled, compute per-call cost using cached pricing.
MCP tools and actions use async client/server protocols: discovery (list_tools), dynamic Pydantic model generation for tool schemas, and wrapper functions that relay user-supplied parameters to the MCP server. The server (in server.py) exposes a tool set for browser and agent control, routing calls to appropriate internal methods and capturing telemetry.
All critical data flows are strictly mediated by DTOs. State (file, usage, token, action, session) is always stored in service-scoped objects and mutated through contract-bound methods. Pub/sub and registry patterns allow for extensible, dynamic action discovery.
PHASE 3: Integrity & Contradiction Audit

All module-local imports resolve; circular dependencies are avoided by separating registry, client/server, and data-contract logic. Method signatures match invocation and contract expectations across async boundaries.
No global mutable state exists except where explicitly guarded (e.g., global GmailService singleton).
Action and tool registration is idempotent, and session state is correctly managed through async connect/disconnect protocols.
OAuth, credential, and token files are checked for existence before use; errors are caught and reported via logging or return DTOs.
Schema mapping (JSON Schema -> Pydantic) is handled with dynamic model creation, robust against missing or extra fields.
File I/O and network ops are properly wrapped, and error reporting uses both exception handling and user-visible messages.
No architectural contradictions were detected in state management or API wrapping.
Telemetry and logging are integrated without interfering with control flow.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    Requires all external libraries (google-api-python-client, httpx, mcp, etc.) to be installed and at compatible versions.

    Requires valid Google OAuth credentials and token storage for Gmail API usage.

    Token pricing depends on external LiteLLM repo schema.

    MCP tools assume a running and schema-compliant MCP server.

    OS environment and file system must be writable in configured directories.
	
	
	
------------------------------------

Module Analysis: /
1. Core Identity

    Purpose:
    This module provides the global entry points, foundational configuration, exception handling, logging/observability, and utility functions required to initialize, run, and monitor the broader "browser-use" system; it defines CLI (Command-Line Interface) bootstrapping, persistent environment/configuration logic, global error/reporting patterns, and essential runtime control constructs.

    Key Patterns:
    Service Locator, Singleton, Factory, Adapter, Decorator (for observability/logging), Command, DTO (Data Transfer Object), Facade (for CLI bootstrapping).

2. Component & API Surface (YAML)

# Static analysis of module components and contracts.
module: "/"
dependencies:
  external:
    - asyncio
    - os
    - sys
    - json
    - time
    - signal
    - platform
    - pathlib
    - typing
    - logging
    - dotenv
    - pydantic
    - pydantic_settings
    - uuid
    - psutil
    - click
    - textual
    - rich
  internal:
    - browser_use
    - browser_use.llm.anthropic.chat
    - browser_use.llm.google.chat
    - browser_use.llm.openai.chat
    - browser_use.agent.views
    - browser_use.browser
    - browser_use.config
    - browser_use.logging_config
    - browser_use.telemetry
    - browser_use.utils
    - browser_use.exceptions
    - browser_use.observability
public_api:
  - name: "get_default_config"
    type: "Function"
    signature: "() -> dict[str, Any]"
    description: "Loads the default configuration from the unified config system."
  - name: "load_user_config"
    type: "Function"
    signature: "() -> dict[str, Any]"
    description: "Loads the user configuration, including command history."
  - name: "save_user_config"
    type: "Function"
    signature: "(config: dict[str, Any]) -> None"
    description: "Persists the user’s command history to disk."
  - name: "update_config_with_click_args"
    type: "Function"
    signature: "(config: dict[str, Any], ctx: click.Context) -> dict[str, Any]"
    description: "Updates the configuration dictionary with CLI arguments from a Click context."
  - name: "setup_readline_history"
    type: "Function"
    signature: "(history: list[str]) -> None"
    description: "Initializes command-line history for readline."
  - name: "get_llm"
    type: "Function"
    signature: "(config: dict[str, Any]) -> ChatOpenAI | ChatAnthropic | ChatGoogle"
    description: "Returns the appropriate LLM (Large Language Model) client based on config and API keys."
  - name: "RichLogHandler"
    type: "Class"
    properties:
      - "rich_log: RichLog"
    methods:
      - "emit(record: logging.LogRecord) -> None"
    description: "Custom logging handler that directs log output to a RichLog widget."
  - name: "BrowserUseApp"
    type: "Class"
    properties:
      - "config: dict"
      - "browser_session: BrowserSession | None"
      - "controller: Controller | None"
      - "agent: Agent | None"
      - "llm: Any | None"
      - "task_history: list"
    methods:
      - "setup_richlog_logging() -> None"
      - "on_mount() -> None"
      - "on_input_key_up(event: events.Key) -> None"
      - "on_input_key_down(event: events.Key) -> None"
      - "on_key(event: events.Key) -> Awaitable[None]"
      - "on_input_submitted(event: Input.Submitted) -> None"
      - "hide_intro_panels() -> None"
      - "update_info_panels() -> None"
      - "update_browser_panel() -> None"
      - "update_model_panel() -> None"
      - "update_tasks_panel() -> None"
    description: "Main Textual TUI application entrypoint for browser-use."
  - name: "SignalHandler"
    type: "Class"
    properties:
      - "loop: asyncio.AbstractEventLoop"
      - "pause_callback: Callable | None"
      - "resume_callback: Callable | None"
      - "custom_exit_callback: Callable | None"
    methods:
      - "register() -> None"
      - "unregister() -> None"
      - "sigint_handler() -> None"
      - "sigterm_handler() -> None"
      - "wait_for_resume() -> None"
      - "reset() -> None"
    description: "Handles OS signals for safe pausing, resuming, and exiting in async CLI workflows."
  - name: "observe"
    type: "Function"
    signature: "(name: str | None = None, ...) -> Callable[[F], F]"
    description: "Decorator providing optional function tracing using Laminar (lmnr) if available; no-op otherwise."
  - name: "observe_debug"
    type: "Function"
    signature: "(name: str | None = None, ...) -> Callable[[F], F]"
    description: "Debug-only tracing decorator, using Laminar if present and debug mode is enabled."
  - name: "addLoggingLevel"
    type: "Function"
    signature: "(levelName: str, levelNum: int, methodName: str | None = None) -> None"
    description: "Dynamically adds a new log level to the Python logging module."
  - name: "setup_logging"
    type: "Function"
    signature: "(stream: Any = None, log_level: str | None = None, force_setup: bool = False) -> logging.Logger"
    description: "Configures root and app-specific loggers, with support for custom log levels and rich output."
  - name: "AgentException"
    type: "Class"
    description: "Base exception for agent-related errors."
  - name: "AgentConfigurationError"
    type: "Class"
    description: "Raised for invalid agent configuration."
  - name: "AgentInterruptedError"
    type: "Class"
    description: "Raised when the agent is interrupted by user or system."
  - name: "LLMException"
    type: "Class"
    description: "Raised for non-recoverable LLM invocation errors."
  - name: "RateLimitError"
    type: "Class"
    description: "Specific LLMException for rate-limiting."
  - name: "LockTimeoutError"
    type: "Class"
    description: "Raised when state lock acquisition times out."
  - name: "time_execution_sync"
    type: "Function"
    signature: "(additional_text: str = '') -> Callable"
    description: "Decorator for measuring execution time of synchronous functions, with logging."
  - name: "time_execution_async"
    type: "Function"
    signature: "(additional_text: str = '') -> Callable"
    description: "Async decorator for measuring execution time of coroutines."
  - name: "singleton"
    type: "Function"
    signature: "(cls: type) -> Callable"
    description: "Decorator to make a class a singleton."
  - name: "check_env_variables"
    type: "Function"
    signature: "(keys: list[str], any_or_all=all) -> bool"
    description: "Utility to check for presence of required environment variables."
data_contracts:
  - name: "DBStyleEntry"
    schema:
      id: "str"
      default: "bool"
      created_at: "str"
  - name: "BrowserProfileEntry"
    schema:
      id: "str"
      default: "bool"
      created_at: "str"
      headless: "bool | None"
      user_data_dir: "str | None"
      allowed_domains: "list[str] | None"
      downloads_path: "str | None"
  - name: "LLMEntry"
    schema:
      id: "str"
      default: "bool"
      created_at: "str"
      api_key: "str | None"
      model: "str | None"
      temperature: "float | None"
      max_tokens: "int | None"
  - name: "AgentEntry"
    schema:
      id: "str"
      default: "bool"
      created_at: "str"
      max_steps: "int | None"
      use_vision: "bool | None"
      system_prompt: "str | None"
  - name: "DBStyleConfigJSON"
    schema:
      browser_profile: "dict[str, BrowserProfileEntry]"
      llm: "dict[str, LLMEntry]"
      agent: "dict[str, AgentEntry]"

PHASE 1: Static Component & Dependency Graph Construction

The / segment aggregates core runtime initialization and bootstrapping logic for the system. The CLI interface (cli.py) composes configuration from the environment and user overrides, supports history and interactive TUI with Textual, and launches the primary application (BrowserUseApp) using service and LLM factories. The configuration logic (config.py) exposes both legacy (environment variable–driven) and "database-style" (Pydantic-backed, versioned JSON) config surfaces, with migration and defaulting behaviors. Exceptions are centralized in exceptions.py with a typed error hierarchy for all agent and LLM execution contexts. Logging is orchestrated through logging_config.py, supporting dynamic log levels and RichLog/Textual handlers for both console and in-app streams. Observability is provided by a decorator toolkit in observability.py that conditionally integrates with "lmnr" (Laminar) tracing, defaulting to no-op stubs if not present. Utility logic in utils.py supplies safe OS signal management for async and cross-platform control, singleton decorators, execution time measurement for both sync and async, and basic environment validation. All data contracts are realized as Pydantic models to guarantee forward-compatible, structured configuration.
PHASE 2: Data & Control Flow Analysis

The system's lifecycle begins in CLI invocation, which loads environment variables via dotenv, parses config files, applies overrides, and initializes telemetry. User actions are driven via Textual input widgets in a TUI, routing through handler methods (on_mount, on_input_submitted, etc.), which update or mutate in-memory config and dispatch tasks to the agent or controller via the BrowserUseApp context. Exception handling surfaces at every async boundary via custom exception classes. Logging flows through dynamic log level handlers, enabling structured output for both application and third-party modules. Signal handling is fully stateful and cross-platform, tracking first/second Ctrl+C events and pausing or exiting accordingly. Observability/tracing is optionally injected into all major async and sync control points if the "lmnr" library is installed and debugging is active, but safely no-ops if not. All persistent configuration changes (e.g., command history, browser/LLM/agent settings) are saved to disk via JSON serialization. All TUI state and task histories are maintained in the app context and regularly synchronized. Configuration management, environment checks, and error surfaces are guarded and contractually consistent via Pydantic schemas and explicit method signatures.
PHASE 3: Integrity & Contradiction Audit

All internal and external dependencies are present and resolved, with dynamic loading for optional libraries (e.g., textual, lmnr, readline). CLI/TUI bootstrapping is robust against missing dependencies, providing user guidance for installation if modules are missing. Configuration migrations are strictly guarded with detection and replacement of old formats. Logging is safely initialized with log level name collisions avoided by defensive dynamic method registration. All custom exceptions subclass a central root and surface actionable error messages. Signal management is portable and correct, with Windows and Unix paths handled explicitly. Observability decorators degrade gracefully. All config DTOs and runtime models are type-checked. No circular dependencies or unreachable code paths were detected. All public APIs have correct invocation signatures. No race conditions or dead-ends are present in the main app logic or config save/restore flows.
ASSUMPTIONS & EXTERNAL DEPENDENCIES

    The system presumes that required external modules (click, textual, rich, pydantic, dotenv, psutil) are available; CLI disables or exits with instructions if any are missing.

    Environment variable and JSON config file locations are presumed to be readable/writable.

    Optional dependencies (lmnr, readline) are handled with run-time checks and do not block core functionality.

    User's OS and terminal must support ANSI sequences for best CLI/TUI experience.
