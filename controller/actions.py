from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import markdownify

from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserError
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
from browser_use.llm.messages import UserMessage

if TYPE_CHECKING:
    from browser_use.browser import BrowserSession
    from browser_use.browser.types import Page
    from browser_use.filesystem.file_system import FileSystem
    from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)


async def search_google(params: SearchGoogleAction) -> ActionResult:
    search_url = f'https://www.google.com/search?q={params.query}&udm=14'
    page = await browser_session.get_current_page()
    await browser_session.navigate_to(search_url, new_tab=(page.url.strip('/') == 'https://www.google.com'))
    return ActionResult(extracted_content=f'🔍 Searched for "{params.query}" in Google.', include_in_memory=True)

async def go_to_url(params: GoToUrlAction) -> ActionResult:
    await browser_session.navigate_to(params.url, new_tab=params.new_tab)
    return ActionResult(extracted_content=f'🔗 Navigated to {params.url}.', include_in_memory=True)

async def go_back(_: NoParamsAction) -> ActionResult:
    await browser_session.go_back()
    return ActionResult(extracted_content='🔙 Navigated back.', include_in_memory=True)

async def wait(params: WaitAction) -> ActionResult:
    await asyncio.sleep(params.seconds)
    return ActionResult(extracted_content=f'🕒 Waited for {params.seconds} seconds.', include_in_memory=True)

async def click_element_by_index(params: ClickElementAction) -> ActionResult:
    selector_map = await browser_session.get_selector_map()
    if params.index not in selector_map:
        return ActionResult(success=False, error=f"Element with index {params.index} not found.")

    element_node = selector_map[params.index]
    if browser_session.is_file_input(element_node):
        return ActionResult(success=False, error="Element is a file input. Use 'upload_file' instead.")

    initial_tab_count = len(browser_session.tabs)
    download_path = await browser_session._click_element_node(element_node)
    
    msg = f'🖱️ Clicked element {params.index}: "{element_node.get_all_text_till_next_clickable_element(max_depth=2)}"'
    if download_path:
        msg = f'💾 Downloaded file to {download_path}.'
    elif len(browser_session.tabs) > initial_tab_count:
        await browser_session.switch_to_tab(-1)
        msg += " and switched to new tab."
        
    return ActionResult(extracted_content=msg, include_in_memory=True)

async def input_text(params: InputTextAction) -> ActionResult:
    element_node = await browser_session.get_dom_element_by_index(params.index)
    if not element_node:
        return ActionResult(success=False, error=f"Element with index {params.index} not found.")
    await browser_session._input_text_element_node(element_node, params.text)
    msg = f'⌨️ Input sensitive data into element {params.index}.' if has_sensitive_data else f'⌨️ Input "{params.text}" into element {params.index}.'
    return ActionResult(extracted_content=msg, include_in_memory=True)

async def upload_file(params: UploadFileAction) -> ActionResult:
    if params.path not in available_file_paths:
        return ActionResult(success=False, error=f'File path {params.path} is not available.')
    if not os.path.exists(params.path):
        return ActionResult(success=False, error=f'File {params.path} does not exist.')
    
    element_node = await browser_session.find_file_upload_element_by_index(params.index)
    if not element_node:
        return ActionResult(success=False, error=f"No file upload element found at index {params.index}.")
    
    playwright_element = await browser_session.get_locate_element(element_node)
    if not playwright_element:
        return ActionResult(success=False, error=f"Could not locate playwright element for index {params.index}.")
    
    await playwright_element.set_input_files(params.path)
    return ActionResult(extracted_content=f'📁 Uploaded file "{params.path}" to element {params.index}.', include_in_memory=True)

async def switch_tab(params: SwitchTabAction) -> ActionResult:
    await browser_session.switch_to_tab(params.page_id)
    page = await browser_session.get_current_page()
    return ActionResult(extracted_content=f'🔄 Switched to tab #{params.page_id} ({page.url}).', include_in_memory=True)

async def close_tab(params: CloseTabAction) -> ActionResult:
    await browser_session.switch_to_tab(params.page_id)
    page = await browser_session.get_current_page()
    url = page.url
    await page.close()
    new_page = await browser_session.get_current_page()
    return ActionResult(extracted_content=f'❌ Closed tab #{params.page_id} ({url}), now on tab #{browser_session.tabs.index(new_page)}.', include_in_memory=True)

async def extract_structured_data(params: ExtractStructuredDataAction) -> ActionResult:
    loop = asyncio.get_event_loop()
    page_html = await page.content()
    markdownify_func = partial(markdownify.markdownify, strip=([] if params.extract_links else ['a', 'img']))
    content = await loop.run_in_executor(None, markdownify_func, page_html)
    content = re.sub(r'\n+', '\n', content)
    
    max_chars = 30000
    if len(content) > max_chars:
        content = content[:max_chars//2] + "\n... [CONTENT TRUNCATED] ...\n" + content[-max_chars//2:]

    prompt = f"Extract information from this webpage based on the query. Respond in JSON format.\nQuery: {params.query}\n\nWebsite Content:\n{content}"
    response = await page_extraction_llm.ainvoke([UserMessage(content=prompt)])
    extracted_content = f'Query: {params.query}\nExtracted Content:\n{response.completion}'
    
    if len(extracted_content) > 600:
        save_result = await file_system.save_extracted_content(extracted_content)
        memory = f"Extracted content from {page.url} for query '{params.query}'. See file: {save_result}"
        return ActionResult(extracted_content=extracted_content, include_extracted_content_only_once=True, long_term_memory=memory)
    
    return ActionResult(extracted_content=extracted_content, include_in_memory=True, long_term_memory=extracted_content)

async def scroll(params: ScrollAction) -> ActionResult:
    page = await browser_session.get_current_page()
    window_height = await page.evaluate('() => window.innerHeight') or 720
    dy = int(window_height * params.num_pages) * (1 if params.down else -1)
    direction = "down" if params.down else "up"
    
    await page.evaluate(f'window.scrollBy(0, {dy})')
    return ActionResult(extracted_content=f'🔍 Scrolled page {direction} by {params.num_pages} pages.', include_in_memory=True)

async def send_keys(params: SendKeysAction) -> ActionResult:
    await page.keyboard.press(params.keys)
    return ActionResult(extracted_content=f'⌨️ Sent keys: {params.keys}.', include_in_memory=True)

async def scroll_to_text(params: ScrollToTextAction) -> ActionResult:
    locator = page.get_by_text(params.text, exact=False)
    try:
        if await locator.count() > 0:
            await locator.first.scroll_into_view_if_needed()
            await asyncio.sleep(0.5)
            return ActionResult(extracted_content=f'🔍 Scrolled to text: {params.text}.', include_in_memory=True)
    except Exception:
        pass
    return ActionResult(success=False, error=f"Text '{params.text}' not found or not visible.")

async def write_file(params: WriteFileAction) -> ActionResult:
    result = await file_system.write_file(params.file_name, params.content)
    return ActionResult(extracted_content=result, include_in_memory=True)

async def append_file(params: AppendFileAction) -> ActionResult:
    result = await file_system.append_file(params.file_name, params.content)
    return ActionResult(extracted_content=result, include_in_memory=True)

async def read_file(params: ReadFileAction) -> ActionResult:
    content = await file_system.read_file(params.file_name, external_file=(params.file_name in available_file_paths))
    return ActionResult(extracted_content=content, include_in_memory=True, include_extracted_content_only_once=True)

async def get_dropdown_options(params: GetDropdownOptionsAction) -> ActionResult:
    page = await browser_session.get_current_page()
    dom_element = (await browser_session.get_selector_map()).get(params.index)
    if not dom_element:
        return ActionResult(success=False, error=f"Dropdown with index {params.index} not found.")

    options_data = await page.evaluate(f"""(xpath) => {{
        const select = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (!select || select.tagName.toLowerCase() !== 'select') return null;
        return Array.from(select.options).map(opt => ({{ text: opt.text, value: opt.value, index: opt.index }}));
    }}""", dom_element.xpath)
    
    if not options_data:
        return ActionResult(success=False, error="No options found for dropdown.")
    
    all_options = [f"{opt['index']}: text={json.dumps(opt['text'])}" for opt in options_data]
    return ActionResult(extracted_content='\n'.join(all_options), include_in_memory=True)

async def select_dropdown_option(params: SelectDropdownOptionAction) -> ActionResult:
    page = await browser_session.get_current_page()
    dom_element = (await browser_session.get_selector_map()).get(params.index)
    if not dom_element or dom_element.tag_name.lower() != 'select':
        return ActionResult(success=False, error=f"Element with index {params.index} is not a dropdown.")

    try:
        await page.locator(f"xpath=//{dom_element.xpath}").select_option(label=params.text, timeout=2000)
        return ActionResult(extracted_content=f"Selected option '{params.text}' from dropdown {params.index}.", include_in_memory=True)
    except Exception:
        return ActionResult(success=False, error=f"Could not select option '{params.text}' from dropdown {params.index}.")

async def read_sheet_contents(_: ReadSheetContentsAction) -> ActionResult:
    await page.keyboard.press('ControlOrMeta+A'); await page.keyboard.press('ControlOrMeta+C')
    return ActionResult(extracted_content=await page.evaluate('navigator.clipboard.readText()'), include_in_memory=True)

async def select_cell_or_range(params: SelectCellOrRangeAction) -> ActionResult:
    await page.keyboard.press('Escape'); await page.keyboard.press('Control+G'); await page.keyboard.type(params.cell_or_range, delay=20); await page.keyboard.press('Enter'); await page.keyboard.press('Escape')
    return ActionResult(extracted_content=f'Selected cells: {params.cell_or_range}.', include_in_memory=False)

async def read_cell_contents(params: ReadCellContentsAction) -> ActionResult:
    page = await browser_session.get_current_page()
    await select_cell_or_range(SelectCellOrRangeAction(cell_or_range=params.cell_or_range), page=page)
    await page.keyboard.press('ControlOrMeta+C')
    return ActionResult(extracted_content=await page.evaluate('navigator.clipboard.readText()'), include_in_memory=True)

async def update_cell_contents(params: UpdateCellContentsAction) -> ActionResult:
    page = await browser_session.get_current_page()
    await select_cell_or_range(SelectCellOrRangeAction(cell_or_range=params.cell_or_range), page=page)
    await page.evaluate("text => navigator.clipboard.writeText(text)", params.new_contents_tsv)
    await page.keyboard.press('ControlOrMeta+V')
    return ActionResult(extracted_content=f'Updated cells: {params.cell_or_range}.', include_in_memory=False)

async def clear_cell_contents(params: ClearCellContentsAction) -> ActionResult:
    page = await browser_session.get_current_page()
    await select_cell_or_range(SelectCellOrRangeAction(cell_or_range=params.cell_or_range), page=page)
    await page.keyboard.press('Backspace')
    return ActionResult(extracted_content=f'Cleared cells: {params.cell_or_range}.', include_in_memory=False)

async def fallback_input_into_single_selected_cell(params: FallbackInputIntoSingleSelectedCellAction) -> ActionResult:
    await page.keyboard.type(params.text, delay=20)
    await page.keyboard.press('Enter'); await page.keyboard.press('ArrowUp')
    return ActionResult(extracted_content=f'Inputted text: {params.text}.', include_in_memory=False)