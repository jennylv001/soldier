from __future__ import annotations

import asyncio
import json
import enum
import logging
from contextlib import nullcontext
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser import BrowserSession
from browser_use.controller.default_actions import BROWSER_USE_DEFAULT_ACTIONS
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import DoneAction, StructuredOutputAction
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.base import BaseChatModel
from browser_use.observability import observe_debug
from browser_use.utils import time_execution_sync

try:
    from lmnr import Laminar
except ImportError:
    Laminar = None

logger = logging.getLogger(__name__)

ContextT = TypeVar('ContextT')
T = TypeVar('T', bound=BaseModel)


class Controller(Generic[ContextT]):
    """
    Orchestrates the registration and execution of agent actions.

    This class uses the powerful `Registry` to manage individual actions.
    Its primary role is to execute single actions via `act` and sequences
    of actions via `multi_act`.
    """

    registry: Registry[ContextT]
    display_files_in_done_text: bool

    def __init__(
        self,
        exclude_actions: Optional[list[str]] = None,
        output_model: Optional[type[T]] = None,
        display_files_in_done_text: bool = True,
    ):
        """Initializes the Controller and registers all default actions."""
        self.registry = Registry[ContextT](exclude_actions or [])
        self.display_files_in_done_text = display_files_in_done_text
        self._register_default_actions()
        if output_model:
            self.use_structured_output_action(output_model)

    def _register_default_actions(self) -> None:
        """Registers all default actions from the default_actions module."""
        for action_definition in BROWSER_USE_DEFAULT_ACTIONS:
            self.registry.action(
                description=action_definition.description,
                param_model=action_definition.param_model,
                domains=action_definition.domains,
            )(action_definition.function)

        self._register_done_action(None, self.display_files_in_done_text)

    # Custom done action for structured output
    def _register_done_action(self, output_model: type[T] | None, display_files_in_done_text: bool = True):
        if output_model is not None:
            self.display_files_in_done_text = display_files_in_done_text

            @self.registry.action(

                description='Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached',
                param_model=StructuredOutputAction[output_model],      
            )
 

            async def done(params: StructuredOutputAction[output_model]):
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=json.dumps(output_dict),
                    long_term_memory=f'Task completed. Success Status: {params.success}',
                )

        else:

            @self.registry.action(

                description='Complete task - provide a summary of results for the user. Set success=True if task completed successfully, false otherwise. Text should be your response to the user summarizing results. Include files you would like to display to the user in files_to_display.',
                param_model=DoneAction,       
            )
            
            async def done(params: DoneAction):
                user_message = params.text

                len_text = len(params.text)
                len_max_memory = 100
                memory = f'Task completed: {params.success} - {params.text[:len_max_memory]}'
                if len_text > len_max_memory:
                    memory += f' - {len_text - len_max_memory} more characters'

                attachments = []
                if params.files_to_display and file_system:
                    if self.display_files_in_done_text:
                        file_msg = ''
                        for file_name in params.files_to_display:
                            if file_name == 'todo.md':
                                continue
                            file_content = file_system.display_file(file_name)
                            if file_content:
                                file_msg += f'\n\n{file_name}:\n{file_content}'
                                attachments.append(file_name)
                        if file_msg:
                            user_message += '\n\nAttachments:'
                            user_message += file_msg
                        else:
                            logger.warning('Agent wanted to display files but none were found')
                    else:
                        for file_name in params.files_to_display:
                            if file_name == 'todo.md':
                                continue
                            file_content = file_system.display_file(file_name)
                            if file_content:
                                attachments.append(file_name)

                if file_system:
                    attachments = [str(file_system.get_dir() / file_name) for file_name in attachments]

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=user_message,
                    long_term_memory=memory,
                    attachments=attachments,
                )

    def use_structured_output_action(self, output_model: type[T]):
        self._register_done_action(output_model, self.display_files_in_done_text)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description=description, **kwargs)
        
    @observe_debug(name='act')
    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        browser_session: BrowserSession,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[dict[str, str | dict[str, str]]] = None,
        available_file_paths: Optional[list[str]] = None,
        file_system: Optional[FileSystem] = None,
        context: Optional[ContextT] = None,
    ) -> ActionResult:
        """Executes a single action using the registry."""
        action_dump = action.model_dump(exclude_unset=True)
        if not action_dump:
            return ActionResult(success=False, error="Received an empty action model.")

        action_name = next(iter(action_dump))
        params = action_dump.get(action_name, {}) or {}

        span_context = Laminar.start_as_current_span(
            name=action_name, input={'action': action_name, 'params': params}, span_type='TOOL'
        ) if Laminar else nullcontext()

        with span_context:
            try:
                result = await self.registry.execute_action(
                    action_name=action_name,
                    params=params,
                    browser_session=browser_session,
                    page_extraction_llm=page_extraction_llm,
                    file_system=file_system,
                    sensitive_data=sensitive_data,
                    available_file_paths=available_file_paths,
                    context=context,
                )
                if Laminar and result:
                    Laminar.set_span_output(result)
            except Exception as e:
                logger.error(f"Error executing action '{action_name}': {e}", exc_info=True)
                result = ActionResult(error=str(e))
                if Laminar:
                    Laminar.set_span_output({'error': str(e)})

        if isinstance(result, ActionResult):
            result.action = action
            return result
        elif isinstance(result, str):
            return ActionResult(extracted_content=result, action=action)
        elif result is None:
            return ActionResult(action=action)
        else:
            raise ValueError(f"Invalid action result type: {type(result)} of {result}")

    @observe_debug(name='multi_act')
    @time_execution_sync('--multi_act')
    async def multi_act(
        self,
        actions: list[ActionModel],
        browser_session: BrowserSession,
        check_ui_stability: bool = True, # Make this feature configurable
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[dict[str, str | dict[str, str]]] = None,
        available_file_paths: Optional[list[str]] = None,
        file_system: Optional[FileSystem] = None,
        context: Optional[ContextT] = None,
    ) -> list[ActionResult]:
        """
        Executes a sequence of actions, stopping on failure or 'done'.
        Includes an optional UI stability check to prevent acting on a stale DOM.
        """
        results: list[ActionResult] = []

        # Get the initial state of the page before any actions in this batch are taken.
        cached_selector_map = await browser_session.get_selector_map()
        cached_path_hashes = {e.hash.branch_path_hash for e in cached_selector_map.values()}

        for i, action_model_instance in enumerate(actions):
            action_name = next(iter(action_model_instance.model_dump(exclude_unset=True)), 'unknown')
            
            try:
                # --- UI Stability Check ---
                # For every action after the first, if it targets an element by index,
                # we can check if the page has changed unexpectedly.
                action_params = action_model_instance.model_dump(exclude_unset=True).get(action_name, {})
                target_index = action_params.get('index')

                if i > 0 and check_ui_stability and target_index is not None:
                    # Get the current state of the page to compare against our cached initial state.
                    new_browser_state_summary = await browser_session.get_state_summary(cache_clickable_elements_hashes=False)
                    new_selector_map = new_browser_state_summary.selector_map
                    
                    # Check 1: Has the specific element we are targeting changed?
                    orig_target = cached_selector_map.get(target_index)
                    new_target = new_selector_map.get(target_index)
                    if (orig_target and new_target and orig_target.hash.branch_path_hash != new_target.hash.branch_path_hash) or (not orig_target and new_target):
                        msg = f"Halting execution: Element at index {target_index} has changed since the start of the step. The agent should re-evaluate the page."
                        logger.info(msg)
                        results.append(ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg))
                        break # Stop processing further actions in this batch

                    # Check 2: Have any new, unexpected elements appeared on the page?
                    new_path_hashes = {e.hash.branch_path_hash for e in new_selector_map.values()}
                    if not new_path_hashes.issubset(cached_path_hashes):
                        msg = "Halting execution: New elements have appeared on the page since the start of the step. The agent should re-evaluate the page."
                        logger.info(msg)
                        results.append(ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg))
                        break # Stop processing further actions in this batch

                # --- Execute Action ---
                action_result = await self.act(
                    action=action_model_instance,
                    browser_session=browser_session,
                    page_extraction_llm=page_extraction_llm,
                    sensitive_data=sensitive_data,
                    available_file_paths=available_file_paths,
                    file_system=file_system,
                    context=context,
                )
                results.append(action_result)


                if action_result.error is not None or action_result.is_done:
                    reason = 'failed' if action_result.error is not None else "signaled 'done'"
                    logger.info(f"Action '{action_name}' {reason}. Halting further actions in this step.")
                    break

                # --- Wait Between Actions ---
                if i < len(actions) - 1 and browser_session.browser_profile.wait_between_actions > 0:
                    await asyncio.sleep(browser_session.browser_profile.wait_between_actions)

            except Exception as e:
                error_msg = f"Controller-level error executing action '{action_name}': {e}"
                logger.error(error_msg, exc_info=True)
                results.append(ActionResult(action=action_model_instance, success=False, error=error_msg))
                break

        return results