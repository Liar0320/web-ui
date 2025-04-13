import pdb
import logging

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr
import inspect
from functools import wraps

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot, MissingAPIKeyError
from src.utils import utils
from src.utils.report_manager import get_report_manager

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

# webui config
webui_config_manager = utils.ConfigManager()

# 获取全局报表管理器
_global_report_manager = get_report_manager()


def scan_and_register_components(blocks):
    """扫描一个Blocks对象并注册其中的所有交互式组件，但不包括按钮
    
    该函数递归遍历Gradio的Blocks对象，找到所有交互式组件并注册到全局配置管理器中，
    以便后续可以保存和加载UI配置。
    
    Args:
        blocks: Gradio Blocks对象，包含UI组件
        
    Returns:
        int: 注册的组件总数
    """
    global webui_config_manager

    def traverse_blocks(block, prefix=""):
        registered = 0

        # 处理 Blocks 自身的组件
        if hasattr(block, "children"):
            for i, child in enumerate(block.children):
                if isinstance(child, gr.components.Component):
                    # 排除按钮 (Button) 组件
                    if getattr(child, "interactive", False) and not isinstance(child, gr.Button):
                        name = f"{prefix}component_{i}"
                        if hasattr(child, "label") and child.label:
                            # 使用标签作为名称的一部分
                            label = child.label
                            name = f"{prefix}{label}"
                        logger.debug(f"Registering component: {name}")
                        webui_config_manager.register_component(name, child)
                        registered += 1
                elif hasattr(child, "children"):
                    # 递归处理嵌套的 Blocks
                    new_prefix = f"{prefix}block_{i}_"
                    registered += traverse_blocks(child, new_prefix)

        return registered

    total = traverse_blocks(blocks)
    logger.info(f"Total registered components: {total}")


def save_current_config():
    """保存当前WebUI配置到JSON文件
    
    将UI中所有注册组件的当前状态和值保存到配置文件中，
    以便稍后可以恢复相同的UI状态。
    
    Returns:
        str: 保存状态信息
    """
    return webui_config_manager.save_current_config()


def update_ui_from_config(config_file):
    """从配置文件更新WebUI界面
    
    读取上传的配置文件，并将其中的值应用到当前UI的所有注册组件。
    
    Args:
        config_file: 上传的配置文件对象
        
    Returns:
        str: 更新状态信息
    """
    return webui_config_manager.update_ui_from_config(config_file)


def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result


async def stop_agent():
    """请求代理停止并更新UI状态
    
    向全局代理实例发送停止请求，并立即更新UI中的按钮状态，
    提供视觉反馈表明停止命令已发出，代理将在下一个安全点停止。
    
    对于循环执行的任务，这将停止整个循环；对于单次任务，则只停止当前正在执行的任务。
    
    Returns:
        tuple: 包含停止按钮和运行按钮的更新状态
    """
    global _global_agent, _global_agent_state

    try:
        # 同时使用两种方式发送停止信号，确保能够停止任何类型的代理和循环
        if _global_agent is not None:
            # 请求停止代理
            _global_agent.stop()
            logger.info("已发送停止代理请求")
            
        # 设置全局停止标志，这将终止循环执行
        if _global_agent_state is not None:
            _global_agent_state.request_stop()
            logger.info("已设置全局停止标志")
            
        # 更新UI立即反馈
        message = "停止请求已发送 - 代理将在下一个安全点停止"
        logger.info(f"🛑 {message}")

        # 返回UI更新信息
        return (
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"停止过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def stop_research_agent():
    """请求研究代理停止并更新UI状态
    
    专门用于停止深度研究代理，通过设置全局停止标志来终止代理的执行。
    提供增强的反馈信息。
    
    Returns:
        tuple: 包含UI更新信息
    """
    global _global_agent_state

    try:
        # 请求停止
        _global_agent_state.request_stop()

        # 更新UI立即反馈
        message = "停止请求已发送 - 研究代理将在下一个安全点停止"
        logger.info(f"🛑 {message}")

        # 返回UI更新信息
        return (
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"停止过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    """运行浏览器代理执行任务
    
    根据提供的配置参数初始化并运行浏览器代理。支持多种LLM模型、浏览器配置选项，
    并能够保存执行记录、历史和跟踪信息。
    
    Args:
        agent_type: 代理类型，可以是'org'或'custom'
        llm_provider: 语言模型提供商(如openai, anthropic)
        llm_model_name: 使用的模型名称
        llm_num_ctx: 上下文窗口大小
        llm_temperature: 生成温度参数
        llm_base_url: API基础URL
        llm_api_key: API密钥
        use_own_browser: 是否使用自有浏览器
        keep_browser_open: 任务间保持浏览器打开
        headless: 无头模式
        disable_security: 是否禁用安全特性
        window_w: 窗口宽度
        window_h: 窗口高度
        save_recording_path: 录制视频保存路径
        save_agent_history_path: 代理历史保存路径
        save_trace_path: 跟踪信息保存路径
        enable_recording: 是否启用录制
        task: 任务描述
        add_infos: 附加信息
        max_steps: 最大步骤数
        use_vision: 是否使用视觉能力
        max_actions_per_step: 每步最大动作数
        tool_calling_method: 工具调用方法
        chrome_cdp: Chrome CDP URL
        max_input_tokens: 最大输入token数
        
    Returns:
        tuple: 包含执行结果、错误信息、模型动作、模型思考、GIF路径、跟踪文件、历史文件和UI更新信息
    """
    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise gr.Error(str(e), print_exception=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # history_file
            None,  # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    """运行官方浏览器代理
    
    使用原始browser-use代码库中的Agent类初始化并运行浏览器代理。
    负责创建浏览器实例、上下文环境并执行指定任务。
    
    Args:
        llm: 语言模型实例
        use_own_browser: 是否使用自有浏览器
        keep_browser_open: 是否在任务间保持浏览器打开
        headless: 是否使用无头模式
        disable_security: 是否禁用安全特性
        window_w: 窗口宽度
        window_h: 窗口高度
        save_recording_path: 录像保存路径
        save_agent_history_path: 代理历史保存路径
        save_trace_path: 跟踪信息保存路径
        task: 任务描述
        max_steps: 最大执行步骤数
        use_vision: 是否使用视觉功能
        max_actions_per_step: 每步最大动作数
        tool_calling_method: 工具调用方法
        chrome_cdp: Chrome CDP URL
        max_input_tokens: 最大输入token数
        
    Returns:
        tuple: 包含执行结果、错误信息、模型动作、模型思考、跟踪文件和历史文件路径
    """
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    """运行自定义浏览器代理
    
    使用自定义代理类初始化并运行浏览器代理，提供了对原始Agent的扩展功能。
    支持更多自定义提示词和控制器设置。
    
    Args:
        llm: 语言模型实例
        use_own_browser: 是否使用自有浏览器
        keep_browser_open: 是否在任务间保持浏览器打开
        headless: 是否使用无头模式
        disable_security: 是否禁用安全特性
        window_w: 窗口宽度
        window_h: 窗口高度
        save_recording_path: 录像保存路径
        save_agent_history_path: 代理历史保存路径
        save_trace_path: 跟踪信息保存路径
        task: 任务描述
        add_infos: 附加信息
        max_steps: 最大执行步骤数
        use_vision: 是否使用视觉功能
        max_actions_per_step: 每步最大动作数
        tool_calling_method: 工具调用方法
        chrome_cdp: Chrome CDP URL
        max_input_tokens: 最大输入token数
        
    Returns:
        tuple: 包含执行结果、错误信息、模型动作、模型思考、跟踪文件和历史文件路径
    """
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)

            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        # if chrome_cdp not empty string nor None
        if (_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None or (chrome_cdp and cdp_url != "" and cdp_url != None):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_with_stream(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
        loop
):
    """运行代理并流式更新UI
    
    运行浏览器代理并实时向UI发送更新，包括截图和状态信息。
    在无头模式下，通过定期截图提供可视化反馈。
    支持循环执行任务，直到用户手动停止。
    
    Args:
        与run_browser_agent函数参数相同
        loop: 是否循环执行任务，True为循环执行直到手动停止，False为执行一次后停止
        
    Yields:
        list: 包含HTML内容和各种结果数据，用于实时更新UI
    """
    global _global_agent, _global_browser, _global_browser_context, _global_agent_state, _global_report_manager

    # 在开始任务前清除停止标志
    if _global_agent_state:
        _global_agent_state.clear_stop()
        logger.info("已重置停止标志，准备开始任务")

    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    
    # 初始化视图状态
    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
    
    # 记录循环状态
    first_execution = True
    
    # 初始日志输出
    if loop:
        logger.info("循环模式：任务将循环执行直到手动停止")
    else:
        logger.info("非循环模式：任务将只执行一次")

    # 主循环：循环模式下一直执行，非循环模式只执行一次
    while loop or first_execution:
        # 标记此次为已执行
        first_execution = False
        
        # 检查是否已经请求停止
        if _global_agent_state and _global_agent_state.is_stop_requested():
            logger.info("任务循环被用户手动停止")
            yield [
                gr.HTML(value=html_content, visible=True),
                "", "", "", "", None, None, None,
                gr.update(value="Stop", interactive=True),
                gr.update(interactive=True)
            ]
            break
            
        try:
            logger.info("开始执行新的任务循环...")
            
            # 创建新的任务记录
            current_task_record = _global_report_manager.start_task_record(task)
            logger.info(f"已开始记录任务: {current_task_record.task_id}")
            
            if not headless:
                # 非无头模式下直接运行
                try:
                    result = await run_browser_agent(
                        agent_type=agent_type,
                        llm_provider=llm_provider,
                        llm_model_name=llm_model_name,
                        llm_num_ctx=llm_num_ctx,
                        llm_temperature=llm_temperature,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        use_own_browser=use_own_browser,
                        keep_browser_open=keep_browser_open,
                        headless=headless,
                        disable_security=disable_security,
                        window_w=window_w,
                        window_h=window_h,
                        save_recording_path=save_recording_path,
                        save_agent_history_path=save_agent_history_path,
                        save_trace_path=save_trace_path,
                        enable_recording=enable_recording,
                        task=task,
                        add_infos=add_infos,
                        max_steps=max_steps,
                        use_vision=use_vision,
                        max_actions_per_step=max_actions_per_step,
                        tool_calling_method=tool_calling_method,
                        chrome_cdp=chrome_cdp,
                        max_input_tokens=max_input_tokens
                    )
                    
                    # 初始化结果变量
                    final_result = ""
                    errors = ""
                    
                    # 安全地解析返回结果
                    if isinstance(result, tuple):
                        # 提取返回结果
                        results_list = list(result)
                        if len(results_list) > 0:
                            final_result = results_list[0]
                        if len(results_list) > 1:
                            errors = results_list[1]
                            
                        # 更新UI
                        yield [gr.update(visible=False)] + results_list
                    else:
                        # 处理意外的返回值格式
                        logger.warning(f"代理任务返回值格式异常: {type(result)} - {result}")
                        final_result = str(result) if result else ""
                        errors = "意外的返回值格式"
                        yield [
                            gr.update(visible=False),
                            final_result,
                            errors,
                            "",  # model_actions
                            "",  # model_thoughts
                            None,  # recording_gif
                            None,  # trace
                            None,  # history_file
                            gr.update(value="Stop", interactive=True),  # stop_button
                            gr.update(interactive=True)  # run_button
                        ]
                    
                    # 任务完成，记录结果
                    # 判断成功条件：errors为空或者final_result不为空
                    success_condition = (errors == "" or errors is None) or (final_result and isinstance(final_result, str) and len(final_result.strip()) > 0)
                    success = bool(success_condition)
                    _global_report_manager.end_task_record(success, errors if not success else None)
                    logger.info(f"任务已完成: {'成功' if success else '失败'}")
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error during task execution: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    
                    # 记录任务异常
                    _global_report_manager.end_task_record(False, error_msg)
                    
                    yield [
                        gr.update(visible=False),
                        "",  # final_result
                        error_msg,  # errors
                        "",  # model_actions
                        "",  # model_thoughts
                        None,  # recording_gif
                        None,  # trace
                        None,  # history_file
                        gr.update(value="Stop", interactive=True),  # stop_button
                        gr.update(interactive=True)  # run_button
                    ]
                    
                    # 非无头模式下的循环控制
                    is_stopped = False
                    if _global_agent_state and _global_agent_state.is_stop_requested():
                        is_stopped = True
                        logger.info("任务已手动停止 (全局状态)")
                    
                    if not is_stopped and _global_agent is not None:
                        try:
                            if getattr(_global_agent, "state", None) is not None:
                                if getattr(_global_agent.state, "stopped", False):
                                    is_stopped = True
                                    logger.info("任务已停止 (代理状态)")
                        except (AttributeError, Exception):
                            pass
                            
                    # 如果是非循环模式或者用户主动停止，退出循环
                    if is_stopped or not loop:
                        loop_msg = "用户手动停止" if is_stopped else "非循环模式，任务已完成"
                        logger.info(f"循环终止: {loop_msg}")
                        break
                    
                    # 只有循环模式才继续执行
                    logger.info("循环模式：准备开始下一轮任务...")
                    await asyncio.sleep(1)
            else:
                # 无头模式下
                # 在后台运行代理
                agent_task = asyncio.create_task(
                    run_browser_agent(
                        agent_type=agent_type,
                        llm_provider=llm_provider,
                        llm_model_name=llm_model_name,
                        llm_num_ctx=llm_num_ctx,
                        llm_temperature=llm_temperature,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        use_own_browser=use_own_browser,
                        keep_browser_open=keep_browser_open,
                        headless=headless,
                        disable_security=disable_security,
                        window_w=window_w,
                        window_h=window_h,
                        save_recording_path=save_recording_path,
                        save_agent_history_path=save_agent_history_path,
                        save_trace_path=save_trace_path,
                        enable_recording=enable_recording,
                        task=task,
                        add_infos=add_infos,
                        max_steps=max_steps,
                        use_vision=use_vision,
                        max_actions_per_step=max_actions_per_step,
                        tool_calling_method=tool_calling_method,
                        chrome_cdp=chrome_cdp,
                        max_input_tokens=max_input_tokens
                    )
                )

                # 周期性更新流，直到代理任务完成
                while not agent_task.done():
                    try:
                        encoded_screenshot = await capture_screenshot(_global_browser_context)
                        if encoded_screenshot is not None:
                            html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        else:
                            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                            
                        # 尝试获取当前URL并记录
                        try:
                            if _global_browser_context:
                                # 记录是否已成功记录URL
                                recorded_url = False
                                
                                # 尝试获取页面URL
                                try:
                                    # 检查get_current_page是否为方法
                                    get_page_attr = getattr(_global_browser_context, "get_current_page", None)
                                    if callable(get_page_attr):
                                        page = await get_page_attr()
                                        if page:
                                            url = await page.url()
                                            if url and url != "about:blank":
                                                _global_report_manager.record_url(url)
                                                logger.debug(f"记录URL(方法1): {url}")
                                                recorded_url = True
                                except Exception as e:
                                    logger.debug(f"尝试方法1获取URL失败: {str(e)}")
                                
                                # 如果无法通过上述方法获取，尝试捕获任何可能的错误
                                if not recorded_url:
                                    try:
                                        # 尝试最简单的截图方法，可能会更新当前页面信息
                                        await capture_screenshot(_global_browser_context)
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.debug(f"访问浏览器上下文时出错: {str(e)}")
                            
                    except Exception as e:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                    # 检查停止标志
                    if _global_agent_state and _global_agent_state.is_stop_requested():
                        logger.info("任务循环被用户手动停止")
                        yield [
                            gr.HTML(value=html_content, visible=True),
                            "", "", "", "", None, None, None,
                            gr.update(value="Stop", interactive=True),
                            gr.update(interactive=True)
                        ]
                        break
                    else:
                        yield [
                            gr.HTML(value=html_content, visible=True),
                            "", "", "", "", None, None, None,
                            gr.update(),  # Re-enable stop button
                            gr.update()  # Re-enable run button
                        ]
                    await asyncio.sleep(0.1)

                # 如果任务已取消，则退出循环
                if agent_task.cancelled():
                    break
                    
                # 一旦代理任务完成，获取结果
                try:
                    result = await agent_task
                    
                    # 初始化结果变量
                    final_result = ""
                    errors = ""
                    model_actions = ""
                    model_thoughts = ""
                    recording_gif = None
                    trace = None
                    history_file = None
                    
                    # 安全地解析返回结果
                    if isinstance(result, tuple):
                        # 提取最多前7个元素
                        results_list = list(result)
                        if len(results_list) > 0:
                            final_result = results_list[0]
                        if len(results_list) > 1:
                            errors = results_list[1]
                        if len(results_list) > 2:
                            model_actions = results_list[2]
                        if len(results_list) > 3:
                            model_thoughts = results_list[3]
                        if len(results_list) > 4:
                            recording_gif = results_list[4]
                        if len(results_list) > 5:
                            trace = results_list[5]
                        if len(results_list) > 6:
                            history_file = results_list[6]
                        # 忽略可能的其他元素（比如按钮更新）
                    else:
                        # 处理意外的返回值格式
                        logger.warning(f"代理任务返回值格式异常: {type(result)} - {result}")
                        final_result = str(result) if result else ""
                        errors = "意外的返回值格式"

                    # 记录任务完成
                    # 判断成功条件：errors为空或者final_result不为空
                    success_condition = (errors == "" or errors is None) or (final_result and isinstance(final_result, str) and len(final_result.strip()) > 0)
                    success = bool(success_condition)
                    _global_report_manager.end_task_record(success, errors if not success else None)
                    logger.info(f"任务已完成: {'成功' if success else '失败'}")
                    
                except gr.Error:
                    final_result = ""
                    model_actions = ""
                    model_thoughts = ""
                    recording_gif = trace = history_file = None
                    
                    # 记录任务失败
                    _global_report_manager.end_task_record(False, "Gradio错误")
                    logger.info("任务因Gradio错误而失败")
                    
                except Exception as e:
                    errors = f"Agent error: {str(e)}"
                    
                    # 记录任务失败
                    _global_report_manager.end_task_record(False, str(e))
                    logger.info(f"任务失败: {str(e)}")

                # 更新UI以显示当前任务结果
                yield [
                    gr.HTML(value=html_content, visible=True),
                    final_result,
                    errors,
                    model_actions,
                    model_thoughts,
                    recording_gif,
                    trace,
                    history_file,
                    gr.update(value="Stop", interactive=True),  # stop_button - 确保可以停止
                    gr.update(interactive=True)  # run_button - 确保可以重新运行
                ]
                
                # 检查是否需要停止循环
                is_stopped = False
                if _global_agent_state and _global_agent_state.is_stop_requested():
                    is_stopped = True
                    logger.info("任务已手动停止 (全局状态)")
                
                if not is_stopped and _global_agent is not None:
                    try:
                        if getattr(_global_agent, "state", None) is not None:
                            if getattr(_global_agent.state, "stopped", False):
                                is_stopped = True
                                logger.info("任务已停止 (代理状态)")
                    except (AttributeError, Exception):
                        pass
                        
                # 如果是非循环模式或者用户主动停止，退出循环
                if is_stopped or not loop:
                    loop_msg = "用户手动停止" if is_stopped else "非循环模式，任务已完成"
                    logger.info(f"循环终止: {loop_msg}")
                    break
                
                # 只有循环模式才继续执行
                logger.info("循环模式：准备开始下一轮任务...")
                await asyncio.sleep(1)
                
                # 重置代理状态，准备下一轮（保留代理实例但清除某些状态）
                if _global_agent:
                    # 重置代理状态但不停止它
                    try:
                        # 安全地重置consecutive_failures计数器
                        if getattr(_global_agent, "state", None) is not None:
                            agent_state = _global_agent.state
                            # 检查agent_state的类型，确保它是CustomAgentState或包含consecutive_failures属性
                            if hasattr(agent_state, "consecutive_failures"):
                                # 使用setattr而不是直接赋值，避免类型检查错误
                                setattr(agent_state, "consecutive_failures", 0)
                                logger.info("已重置代理状态consecutive_failures计数器，准备下一轮任务")
                    except (AttributeError, Exception) as e:
                        logger.warning(f"重置代理状态时发生错误: {str(e)}")
                
        except Exception as e:
            import traceback
            error_msg = f"Error during task execution: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # 记录任务异常
            if _global_report_manager.current_task:
                _global_report_manager.end_task_record(False, error_msg)
                logger.info("任务因异常而失败")
            
            yield [
                gr.HTML(
                    value=f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Error encountered...</h1>",
                    visible=True),
                "",
                error_msg,
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)  # Re-enable run button
            ]
            
            # 如果不是循环模式，立即退出循环
            if not loop:
                logger.info("任务执行完毕，非循环模式，退出循环")
                break
                
            # 即使发生错误，也延迟一下并继续下一轮
            await asyncio.sleep(2)


# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}


async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                          llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                          use_own_browser, headless, chrome_cdp):
    """运行深度搜索研究功能
    
    使用浏览器代理执行深度研究任务，可以迭代搜索并汇总信息，
    生成完整的研究报告。
    
    Args:
        research_task: 研究任务描述
        max_search_iteration_input: 最大搜索迭代次数
        max_query_per_iter_input: 每次迭代的最大查询数
        llm_provider: 语言模型提供商
        llm_model_name: 模型名称
        llm_num_ctx: 上下文窗口大小
        llm_temperature: 生成温度
        llm_base_url: API基础URL
        llm_api_key: API密钥
        use_vision: 是否使用视觉功能
        use_own_browser: 是否使用自有浏览器
        headless: 是否使用无头模式
        chrome_cdp: Chrome CDP URL
        
    Returns:
        tuple: 包含研究报告Markdown内容、文件路径和UI更新信息
    """
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        num_ctx=llm_num_ctx,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                      max_search_iterations=max_search_iteration_input,
                                                      max_query_num=max_query_per_iter_input,
                                                      use_vision=use_vision,
                                                      headless=headless,
                                                      use_own_browser=use_own_browser,
                                                      chrome_cdp=chrome_cdp
                                                      )

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)


def create_ui(theme_name="Ocean"):
    """创建浏览器代理Web用户界面
    
    创建完整的Gradio Web界面，包括多个选项卡用于配置模型、浏览器设置、运行代理和查看结果。
    支持切换主题、保存/加载配置等功能。
    
    Args:
        theme_name: 界面主题名称，默认为"Ocean"
        
    Returns:
        gr.Blocks: Gradio界面对象
    """
    css = """
    .gradio-container {
        width: 60vw !important; 
        max-width: 60% !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use",
                        interactive=True
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=100,
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                            interactive=True
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                            interactive=True
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=True,
                            info="Enable visual processing capabilities",
                            interactive=True
                        )
                        max_input_tokens = gr.Number(
                            label="Max Input Tokens",
                            value=128000,
                            precision=0,
                            interactive=True
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value="auto",
                            interactive=True,
                            allow_custom_value=True,  # Allow users to input custom model names
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Funtion Name",
                            visible=False
                        )

            with gr.TabItem("🔧 LLM Settings", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider, model in utils.model_names.items()],
                        label="LLM Provider",
                        value="deepseek",
                        info="Select your preferred language model provider",
                        interactive=True
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names['openai'],
                        value="deepseek/deepseek-chat-v3-0324:free",
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select a model in the dropdown options or directly type a custom model name"
                    )
                    ollama_num_ctx = gr.Slider(
                        minimum=2 ** 8,
                        maximum=2 ** 16,
                        value=16000,
                        step=1,
                        label="Ollama Context Length",
                        info="Controls max context length model needs to handle (less = faster)",
                        visible=False,
                        interactive=True
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs",
                        interactive=True
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value="",
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value="",
                            info="Your API key (leave blank to use .env)"
                        )

            # Change event to update context length slider
            def update_llm_num_ctx_visibility(llm_provider):
                return gr.update(visible=llm_provider == "ollama")

            # Bind the change event of llm_provider to update the visibility of context length slider
            llm_provider.change(
                fn=update_llm_num_ctx_visibility,
                inputs=llm_provider,
                outputs=ollama_num_ctx
            )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=False,
                            info="Use your existing browser instance",
                            interactive=True
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=False,
                            info="Keep Browser Open between Tasks",
                            interactive=True
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=False,
                            info="Run browser without GUI",
                            interactive=True
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=True,
                            info="Disable browser security features",
                            interactive=True
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=True,
                            info="Enable saving browser recordings",
                            interactive=True
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1280,
                            info="Browser window width",
                            interactive=True
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1100,
                            info="Browser window height",
                            interactive=True
                        )

                    chrome_cdp = gr.Textbox(
                        label="CDP URL",
                        placeholder="http://localhost:9222",
                        value="",
                        info="CDP for google remote debugging",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value="./tmp/traces",
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value="./tmp/agent_history",
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value="go to google.com and type 'OpenAI' click search and give me the first url",
                    info="Describe what you want the agent to do",
                    interactive=True
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                    value="",
                    interactive=True
                )
                
                with gr.Row():
                    loop = gr.Checkbox(
                        label="循环执行",
                        value=False,
                        info="启用后任务会循环执行，直到手动停止",
                        interactive=True
                    )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    reset_button = gr.Button("🔄 Reset", variant="secondary", scale=1)

                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                        visible=False
                    )

                gr.Markdown("### Results")
                with gr.Row():
                    with gr.Column():
                        final_result_output = gr.Textbox(
                            label="Final Result", lines=3, show_label=True
                        )
                    with gr.Column():
                        errors_output = gr.Textbox(
                            label="Errors", lines=3, show_label=True
                        )
                with gr.Row():
                    with gr.Column():
                        model_actions_output = gr.Textbox(
                            label="Model Actions", lines=3, show_label=True, visible=False
                        )
                    with gr.Column():
                        model_thoughts_output = gr.Textbox(
                            label="Model Thoughts", lines=3, show_label=True, visible=False
                        )
                recording_gif = gr.Image(label="Result GIF", format="gif")
                trace_file = gr.File(label="Trace File")
                agent_history_file = gr.File(label="Agent History")

            with gr.TabItem("🧐 Deep Research", id=5):
                research_task_input = gr.Textbox(label="Research Task", lines=5,
                                                 value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.",
                                                 interactive=True)
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3,
                                                           precision=0,
                                                           interactive=True)  # precision=0 确保是整数
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1,
                                                         precision=0,
                                                         interactive=True)  # precision=0 确保是整数
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹ Stop", variant="stop", scale=1)
                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

            # 添加重置功能，清除停止标志
            async def reset_agent_state():
                global _global_agent_state
                if _global_agent_state:
                    _global_agent_state.clear_stop()
                    logger.info("已手动重置代理状态，清除停止标志")
                return (
                    gr.update(interactive=True),  # run_button
                    "已重置代理状态，可以重新运行任务"  # 状态消息
                )

            # Bind the stop button click event after errors_output is defined
            stop_button.click(
                fn=stop_agent,
                inputs=[],
                outputs=[stop_button, run_button],
            )
            
            # Bind the reset button click event
            reset_button.click(
                fn=reset_agent_state,
                inputs=[],
                outputs=[run_button, errors_output]
            )

            # Run button click handler
            run_button.click(
                fn=run_with_stream,
                inputs=[
                    agent_type, llm_provider, llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url,
                    llm_api_key,
                    use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                    save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                    enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step,
                    tool_calling_method, chrome_cdp, max_input_tokens, loop
                ],
                outputs=[
                    browser_view,  # Browser view
                    final_result_output,  # Final result
                    errors_output,  # Errors
                    model_actions_output,  # Model actions
                    model_thoughts_output,  # Model thoughts
                    recording_gif,  # Latest recording
                    trace_file,  # Trace file
                    agent_history_file,  # Agent history file
                    stop_button,  # Stop button
                    run_button  # Run button
                ],
            )

            # Run Deep Research
            research_button.click(
                fn=run_deep_search,
                inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                        llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                        use_own_browser, headless, chrome_cdp],
                outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
            )
            # Bind the stop button click event after errors_output is defined
            stop_research_button.click(
                fn=stop_research_agent,
                inputs=[],
                outputs=[stop_research_button, research_button],
            )

            with gr.TabItem("🎥 Recordings", id=7, visible=True):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(
                        os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

            with gr.TabItem("📊 任务报表", id=9, visible=True):
                with gr.Row():
                    refresh_report_button = gr.Button("🔄 刷新报表数据", variant="secondary")
                    export_excel_button = gr.Button("📑 导出Excel报表", variant="primary")
                
                # 预加载提示
                auto_refresh_info = gr.Markdown("*页面加载完成后会自动刷新数据*")
                
                report_status = gr.Textbox(label="状态", interactive=False)
                
                gr.Markdown("### 📈 统计信息")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基础统计")
                        total_executions = gr.Textbox(label="总执行次数", interactive=False)
                        successful_executions = gr.Textbox(label="成功次数", interactive=False)
                        failed_executions = gr.Textbox(label="失败次数", interactive=False)
                        success_rate = gr.Textbox(label="成功率", interactive=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 时间统计")
                        total_duration = gr.Textbox(label="总运行时长", interactive=False)
                        avg_duration = gr.Textbox(label="总任务平均每次耗时", interactive=False)
                        avg_successful_duration = gr.Textbox(label="完成任务平均每次耗时", interactive=False)
                
                gr.Markdown("### 📅 最近任务")
                recent_tasks_output = gr.DataFrame(
                    headers=["任务ID", "任务描述", "开始时间", "执行时长", "是否成功"]
                )
                
                report_file_output = gr.File(label="下载Excel报表", visible=False)
                
                # 函数：获取报表数据
                def get_report_data():
                    stats = _global_report_manager.get_statistics()
                    
                    # 获取最近任务
                    recent_tasks = []
                    for task in reversed(_global_report_manager.task_history[-10:]):  # 最近10条
                        recent_tasks.append([
                            task.task_id[:8] + "...",  # 截断ID
                            task.task_description[:50] + "..." if len(task.task_description) > 50 else task.task_description,
                            task.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            task.format_duration() if task.duration else "未完成",
                            "✅" if task.success else "❌"
                        ])
                    
                    # 返回详细的统计信息
                    return (
                        str(stats["total_executions"]),
                        str(stats["successful_executions"]),
                        str(stats["failed_executions"]),
                        stats["success_rate"],
                        stats["total_duration_formatted"],
                        stats["avg_duration_formatted"],
                        stats["avg_successful_duration_formatted"],
                        recent_tasks
                    )
                
                # 函数：导出Excel报表
                def export_excel_report():
                    try:
                        excel_path = _global_report_manager.export_excel()
                        if excel_path and os.path.exists(excel_path):
                            return f"报表已导出至: {excel_path}", excel_path
                        else:
                            return "导出报表失败，请查看日志", None
                    except Exception as e:
                        return f"导出报表时出错: {str(e)}", None
                
                # 绑定刷新按钮
                refresh_report_button.click(
                    fn=get_report_data,
                    inputs=[],
                    outputs=[
                        total_executions,
                        successful_executions,
                        failed_executions,
                        success_rate,
                        total_duration,
                        avg_duration,
                        avg_successful_duration,
                        recent_tasks_output
                    ]
                )
                
                # 绑定导出按钮
                export_excel_button.click(
                    fn=export_excel_report,
                    inputs=[],
                    outputs=[report_status, report_file_output]
                )

            with gr.TabItem("📁 UI Configuration", id=8):
                config_file_input = gr.File(
                    label="Load UI Settings from Config File",
                    file_types=[".json"],
                    interactive=True
                )
                with gr.Row():
                    load_config_button = gr.Button("Load Config", variant="primary")
                    save_config_button = gr.Button("Save UI Settings", variant="primary")

                config_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                save_config_button.click(
                    fn=save_current_config,
                    inputs=[],  # 不需要输入参数
                    outputs=[config_status]
                )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

        scan_and_register_components(demo)
        global webui_config_manager
        all_components = webui_config_manager.get_all_components()

        load_config_button.click(
            fn=update_ui_from_config,
            inputs=[config_file_input],
            outputs=all_components + [config_status]
        )
        
        # 添加应用启动时自动加载报表数据
        demo.load(
            fn=get_report_data,
            inputs=None,
            outputs=[
                total_executions,
                successful_executions,
                failed_executions,
                success_rate,
                total_duration,
                avg_duration,
                avg_successful_duration,
                recent_tasks_output
            ]
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()
