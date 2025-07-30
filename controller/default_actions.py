from __future__ import annotations

from browser_use.controller.registry.views import RegisteredAction
from browser_use.controller.views import (
    AppendFileAction,
    ClearCellContentsAction,
    ClickElementAction,
    CloseTabAction,
    ExtractStructuredDataAction,
    FallbackInputIntoSingleSelectedCellAction,
    GetDropdownOptionsAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    ReadCellContentsAction,
    ReadFileAction,
    ReadSheetContentsAction,
    ScrollAction,
    ScrollToTextAction,
    SearchGoogleAction,
    SelectCellOrRangeAction,
    SelectDropdownOptionAction,
    SendKeysAction,
    SwitchTabAction,
    UpdateCellContentsAction,
    UploadFileAction,
    WaitAction,
    WriteFileAction,
)

from . import actions

# This list is the single source of truth for all default actions.
# It unifies the action's implementation (the function) with its metadata
# (description, parameter model, and domain filters).
BROWSER_USE_DEFAULT_ACTIONS: list[RegisteredAction] = [
    RegisteredAction(name='search_google', function=actions.search_google, description='Search a query in Google.', param_model=SearchGoogleAction),
    RegisteredAction(name='go_to_url', function=actions.go_to_url, description='Navigate to a URL.', param_model=GoToUrlAction),
    RegisteredAction(name='go_back', function=actions.go_back, description='Navigate back in browser history.', param_model=NoParamsAction),
    RegisteredAction(name='wait', function=actions.wait, description='Wait for a specified number of seconds.', param_model=WaitAction),
    RegisteredAction(name='click_element_by_index', function=actions.click_element_by_index, description='Click an element by its index.', param_model=ClickElementAction),
    RegisteredAction(name='input_text', function=actions.input_text, description='Input text into an element.', param_model=InputTextAction),
    RegisteredAction(name='upload_file', function=actions.upload_file, description='Upload a file to an element.', param_model=UploadFileAction),
    RegisteredAction(name='switch_tab', function=actions.switch_tab, description='Switch to a different browser tab.', param_model=SwitchTabAction),
    RegisteredAction(name='close_tab', function=actions.close_tab, description='Close the current browser tab.', param_model=CloseTabAction),
    RegisteredAction(name='extract_structured_data', function=actions.extract_structured_data, description='Extract structured data from the page.', param_model=ExtractStructuredDataAction),
    RegisteredAction(name='scroll', function=actions.scroll, description='Scroll the page or an element.', param_model=ScrollAction),
    RegisteredAction(name='send_keys', function=actions.send_keys, description='Send special key presses to the page.', param_model=SendKeysAction),
    RegisteredAction(name='scroll_to_text', function=actions.scroll_to_text, description='Scroll to a specific text on the page.', param_model=ScrollToTextAction),
    RegisteredAction(name='write_file', function=actions.write_file, description='Write content to a file.', param_model=WriteFileAction),
    RegisteredAction(name='append_file', function=actions.append_file, description='Append content to a file.', param_model=AppendFileAction),
    RegisteredAction(name='read_file', function=actions.read_file, description='Read content from a file.', param_model=ReadFileAction),
    RegisteredAction(name='get_dropdown_options', function=actions.get_dropdown_options, description='Get all options from a native dropdown.', param_model=GetDropdownOptionsAction),
    RegisteredAction(name='select_dropdown_option', function=actions.select_dropdown_option, description='Select an option from a dropdown by its text.', param_model=SelectDropdownOptionAction),
    RegisteredAction(name='read_sheet_contents', function=actions.read_sheet_contents, description='Sheets: Get all sheet contents.', param_model=ReadSheetContentsAction, domains=['https://docs.google.com']),
    RegisteredAction(name='read_cell_contents', function=actions.read_cell_contents, description='Sheets: Get cell/range contents.', param_model=ReadCellContentsAction, domains=['https://docs.google.com']),
    RegisteredAction(name='update_cell_contents', function=actions.update_cell_contents, description='Sheets: Update cell/range contents.', param_model=UpdateCellContentsAction, domains=['https://docs.google.com']),
    RegisteredAction(name='clear_cell_contents', function=actions.clear_cell_contents, description='Sheets: Clear selected cells.', param_model=ClearCellContentsAction, domains=['https://docs.google.com']),
    RegisteredAction(name='select_cell_or_range', function=actions.select_cell_or_range, description='Sheets: Select a cell or range.', param_model=SelectCellOrRangeAction, domains=['https://docs.google.com']),
    RegisteredAction(name='fallback_input_into_single_selected_cell', function=actions.fallback_input_into_single_selected_cell, description='Sheets: Type text into selected cell.', param_model=FallbackInputIntoSingleSelectedCellAction, domains=['https://docs.google.com']),
]