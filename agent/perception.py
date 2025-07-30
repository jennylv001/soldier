from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from browser_use.agent.events import PerceptionOutput
from browser_use.agent.state_manager import AgentStatus, LoadStatus, agent_log, TERMINAL_STATES
from browser_use.browser.views import BrowserStateSummary

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.browser import BrowserSession

logger = logging.getLogger(__name__)

class Perception:
    """ "Perception ≠ Paralysis" & "The Canary Protocol" """

    def __init__(
        self,
        browser_session: BrowserSession,
        state_manager: StateManager,
        settings: AgentSettings,
        perception_queue: asyncio.Queue,
    ):
        self.browser_session = browser_session
        self.state_manager = state_manager
        self.settings = settings
        self.perception_queue = perception_queue
        self._last_signal_time = 0
        self.has_downloads_path = self.browser_session.browser_profile.downloads_path is not None
        if self.has_downloads_path:
            self._last_known_downloads = []

    async def run(self):
        logger.info("Perception component started.")
        while await self.state_manager.get_status() not in TERMINAL_STATES:
            status = await self.state_manager.get_status()
            if status == AgentStatus.RUNNING or status == AgentStatus.REFLECTING:
                # Dynamic Load Shedding Check
                load_status = await self.state_manager.get_load_status()
                if load_status == LoadStatus.SHEDDING:
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                              "System under high load, throttling perception.")
                    await asyncio.sleep(2.0) # Throttle by waiting

                try:
                    new_files = await self._check_and_update_downloads()
                    browser_state = await self._get_browser_state_with_recovery()
                    perception_data = PerceptionOutput(browser_state=browser_state, new_downloaded_files=new_files)
                    
                    try:
                        self.perception_queue.put_nowait(perception_data)
                        self._last_signal_time = time.monotonic()
                    except asyncio.QueueFull:
                        agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                  "Perception overload: signal dropped. Core is not processing fast enough.")

                    await asyncio.sleep(0.1) 
                    while await self.state_manager.get_status() == status:
                        await asyncio.sleep(0.2)

                except Exception as e:
                    error_msg = f"Perception loop failed: {e}"
                    agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
                    await self.state_manager.record_error(error_msg, is_critical=True)
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(0.5)
        logger.info("Perception component stopped.")

    async def watchdog(self, timeout: int = 15):
        # Lenient startup: Wait for the first signal before enforcing the strict timeout.
        # This gives the browser time to initialize, which can be slow under heavy system load.
        while self._last_signal_time == 0:
            if await self.state_manager.get_status() in TERMINAL_STATES:
                return  # Agent stopped before perception could start.
            await asyncio.sleep(0.5)

        while await self.state_manager.get_status() not in TERMINAL_STATES:
            if time.monotonic() - self._last_signal_time > timeout:
                error_msg = f"Perception flatlined: No signal for {timeout}s. Sensor death detected."
                agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg)
                await self.state_manager.set_status(AgentStatus.FAILED)
                raise RuntimeError(error_msg)
            await asyncio.sleep(timeout / 2)

    async def _get_browser_state_with_recovery(self) -> BrowserStateSummary:
        try:
            return await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
        except Exception as e:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Full state retrieval failed: {e}")
            return await self.browser_session.get_minimal_state_summary()

    async def _check_and_update_downloads(self) -> Optional[list[str]]:
        if not self.has_downloads_path: return None
        try:
            current_downloads = self.browser_session.downloaded_files
            if current_downloads != self._last_known_downloads:
                new_files = list(set(current_downloads) - set(self._last_known_downloads))
                if new_files:
                    self._last_known_downloads = current_downloads
                    return new_files
        except Exception as e:
            agent_log(logging.DEBUG, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Failed to check downloads: {e}")
        return None