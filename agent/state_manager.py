from __future__ import annotations

import asyncio
import logging
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

from browser_use.agent.views import AgentHistory, AgentHistoryList

if TYPE_CHECKING:
    from browser_use.filesystem.file_system import FileSystem

logger = logging.getLogger(__name__)

class LockTimeoutError(Exception):
    """Custom exception for lock acquisition timeouts."""
    pass

@asynccontextmanager
async def _bulletproof_lock(lock: asyncio.Lock, timeout: float):
    """Acquires a lock with a timeout, raising a custom error on failure."""
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        yield
    except asyncio.TimeoutError:
        raise LockTimeoutError(f"Lock acquisition failed after {timeout}s: potential deadlock detected")
    finally:
        if lock.locked():
            lock.release()

class AgentStatus(Enum):
    PENDING = "PENDING"; RUNNING = "RUNNING"; PAUSED = "PAUSED"; REFLECTING = "REFLECTING"
    STOPPED = "STOPPED"; COMPLETED = "COMPLETED"; FAILED = "FAILED"; MAX_STEPS_REACHED = "MAX_STEPS_REACHED"


class LoadStatus(Enum):
    NORMAL = "NORMAL"; SHEDDING = "SHEDDING"


TERMINAL_STATES = {AgentStatus.STOPPED, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_STEPS_REACHED}
STATE_PRIORITY = {
    AgentStatus.STOPPED: 5, AgentStatus.FAILED: 4, AgentStatus.PAUSED: 3, AgentStatus.REFLECTING: 2,
    AgentStatus.RUNNING: 1, AgentStatus.COMPLETED: 0, AgentStatus.MAX_STEPS_REACHED: 0, AgentStatus.PENDING: -1,
}


def agent_log(level: int, agent_id: str, step: int, message: str, **kwargs):
    log_extras = {'agent_id': agent_id, 'step': step}
    logger.log(level, message, extra=log_extras, **kwargs)


class AgentState(BaseModel):
    agent_id: str = Field(default_factory=uuid7str)
    task: str
    current_goal: str = ""
    status: AgentStatus = AgentStatus.PENDING
    load_status: LoadStatus = LoadStatus.NORMAL # New: Track system load status
    n_steps: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    accumulated_output: Optional[str] = None
    history: AgentHistoryList = Field(default_factory=AgentHistoryList)
    message_manager_state: Dict[str, Any] = Field(default_factory=dict)
    file_system_state: Optional[Dict[str, Any]] = None
    human_guidance_queue: asyncio.Queue[str] = Field(default_factory=asyncio.Queue, exclude=True)
    class Config: arbitrary_types_allowed = True


class StateManager:
    """Manages the agent's state with a lock to ensure atomic, safe operations."""

    def __init__(
        self, 
        initial_state: AgentState, 
        file_system: Optional[FileSystem],
        max_failures: int,
        lock_timeout_seconds: float,
        use_planner: bool,
        reflect_on_error: bool,
        max_history_items: int
    ):
        self._state = initial_state
        self._lock = asyncio.Lock()
        self.lock_timeout_seconds = lock_timeout_seconds
        self._file_system = file_system
        self.max_failures = max_failures
        self.use_planner = use_planner
        self.reflect_on_error = reflect_on_error
        # Convert history to a bounded deque to prevent memory leaks
        initial_state.history.history = deque(initial_state.history.history, maxlen=max_history_items)
        self._state = initial_state
        
        if not self._state.current_goal:
            self._state.current_goal = self._state.task

    @property
    def state(self) -> AgentState:
        return self._state

    async def get_status(self) -> AgentStatus:
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):          
            return self._state.status

    async def get_load_status(self) -> LoadStatus:
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.load_status

    async def set_load_status(self, new_status: LoadStatus):
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            if self._state.load_status != new_status:
                self._state.load_status = new_status
                agent_log(logging.WARNING, self._state.agent_id, self._state.n_steps, f"System load status changed to: {new_status.value}")

    def _set_status_internal(self, new_status: AgentStatus, force: bool = False):
        """Internal, non-locking version of set_status. Must be called from within a held lock."""
        current_priority = STATE_PRIORITY.get(self._state.status, -1)
        new_priority = STATE_PRIORITY.get(new_status, -1)
        if force or (new_priority >= current_priority and self._state.status != new_status):
            agent_log(logging.DEBUG, self._state.agent_id, self._state.n_steps,
                      f"State transition: {self._state.status.value} -> {new_status.value}")
            self._state.status = new_status

    async def set_status(self, new_status: AgentStatus, force: bool = False):
        """Public method that acquires the lock before setting status."""
        async with self._lock:
            self._set_status_internal(new_status, force=force)

    async def add_history_item(self, item: AgentHistory):
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.history.history.append(item) # deque handles maxlen
            if self._file_system:
                self._state.file_system_state = self._file_system.get_state()

    async def update_after_step(self, results: list[Any], max_steps: int, planner_interval: int):
        next_status = None
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.n_steps += 1
            if any(r.is_done for r in results):
                next_status = AgentStatus.COMPLETED
            elif any(not r.success for r in results):
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.max_failures:
                    next_status = AgentStatus.FAILED
                elif self.use_planner and self.reflect_on_error:
                    next_status = AgentStatus.REFLECTING
            else:
                self._state.consecutive_failures = 0
                self._state.last_error = None
                if self._state.n_steps >= max_steps:
                    next_status = AgentStatus.MAX_STEPS_REACHED
                elif self.use_planner and (self._state.n_steps % planner_interval == 0):
                    next_status = AgentStatus.REFLECTING
                else:
                    next_status = AgentStatus.RUNNING
        
            if next_status:
                self._set_status_internal(next_status)

    async def record_error(self, error_msg: str, is_critical: bool = False):
        next_status = None
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.last_error = error_msg
            if is_critical:
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.max_failures:
                    next_status = AgentStatus.FAILED
                elif self.use_planner and self.reflect_on_error:
                    next_status = AgentStatus.REFLECTING
            
            if next_status:
                self._set_status_internal(next_status, force=True)

    async def update_task(self, new_task: str):
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.task = new_task
            self._state.current_goal = new_task

    async def clear_error_and_failures(self):
        """Resets the error and failure counters, typically after reflection."""
        async with _bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.last_error = None
            self._state.consecutive_failures = 0

    async def add_human_guidance(self, text: str):
        await self._state.human_guidance_queue.put(text)

    async def get_human_guidance(self) -> Optional[str]:
        try:
            guidance = await asyncio.wait_for(self._state.human_guidance_queue.get(), timeout=0.5)
            self._state.human_guidance_queue.task_done()
            return guidance
        except asyncio.TimeoutError:
            return None