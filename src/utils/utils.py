import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests
import json
import gradio as gr
import uuid
import logging

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot",
    "unbound": "Unbound AI"
}


def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider not in ["ollama"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            raise MissingAPIKeyError(provider, env_var)
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        return ChatAnthropic(
            model=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == 'mistral':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("MISTRAL_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            api_key=api_key,
        )
    elif provider == "ollama":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        else:
            base_url = kwargs.get("base_url")

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        api_version = kwargs.get("api_version", "") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "moonshot":
        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("MOONSHOT_ENDPOINT"),
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    elif provider == "unbound":
        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.0),
            base_url = os.getenv("UNBOUND_ENDPOINT", "https://api.getunbound.ai"),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Predefined model names for common providers
model_names = {
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner", "deepseek/deepseek-chat-v3-0324:free"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest",
               "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b", "llama2:7b",
               "deepseek-r1:7b","deepseek-r1:14b", "deepseek-r1:32b","deepseek/deepseek-chat-v3-0324:free"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
    "unbound": ["gemini-2.0-flash","gpt-4o-mini", "gpt-4o", "gpt-4.5-preview"]
}


# Callback to update the model name dropdown based on the selected provider
def update_model_dropdown(llm_provider, api_key=None, base_url=None, current_value=None):
    """
    更新模型下拉列表的选项，根据用户选择的模型提供商。
    
    Args:
        llm_provider: A模型提供商名称
        api_key: 可选的API密钥
        base_url: 可选的基础URL
        current_value: 当前选中的值，如果提供则保留此值
        
    Returns:
        gr.update对象，包含更新后的下拉菜单选项和值
    """
    import gradio as gr
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"更新模型下拉菜单，提供商: {llm_provider}, 当前值: {current_value}")
    
    # 使用环境变量中的API密钥（如果未提供）
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # 使用预定义的模型列表
    if llm_provider in model_names:
        model_list = model_names[llm_provider]
        
        # 如果有传入的当前值且该值在列表中，使用它
        # 否则，如果有传入的当前值但不在列表中，仍然使用它（可能是自定义值）
        # 如果没有传入值，使用列表的第一个值作为默认值
        if current_value:
            default_model = current_value
        else:
            default_model = model_list[0] if model_list else ""
            
        logger.info(f"找到提供商 {llm_provider} 的 {len(model_list)} 个模型，使用值: {default_model}")
        return gr.update(choices=model_list, value=default_model)
    else:
        logger.warning(f"未找到提供商 {llm_provider} 的模型列表")
        return gr.update(choices=[], value=current_value or "", allow_custom_value=True)


class MissingAPIKeyError(Exception):
    """Custom exception for missing API key."""

    def __init__(self, provider: str, env_var: str):
        provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
        super().__init__(f"💥 {provider_display} API key not found! 🔑 Please set the "
                         f"`{env_var}` environment variable or provide it in the UI.")


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None


class ConfigManager:
    def __init__(self):
        self.components = {}
        self.component_order = []

    def register_component(self, name: str, component):
        """Register a gradio component for config management."""
        self.components[name] = component
        if name not in self.component_order:
            self.component_order.append(name)
        return component

    def save_current_config(self):
        """Save the current configuration of all registered components."""
        current_config = {}
        for name in self.component_order:
            component = self.components[name]
            # Get the current value from the component
            current_config[name] = getattr(component, "value", None)

        return save_config_to_file(current_config)

    def update_ui_from_config(self, config_file):
        """Update UI components from a loaded configuration file."""
        if config_file is None:
            return [gr.update() for _ in self.component_order] + ["No file selected."]

        loaded_config = load_config_from_file(config_file.name)

        if not isinstance(loaded_config, dict):
            return [gr.update() for _ in self.component_order] + ["Error: Invalid configuration file."]

        # Prepare updates for all components
        updates = []
        for name in self.component_order:
            if name in loaded_config:
                updates.append(gr.update(value=loaded_config[name]))
            else:
                updates.append(gr.update())

        updates.append("Configuration loaded successfully.")
        return updates

    def get_all_components(self):
        """Return all registered components in the order they were registered."""
        return [self.components[name] for name in self.component_order]


def load_config_from_file(config_file):
    """Load settings from a config file (JSON format)."""
    try:
        with open(config_file, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        return f"Error loading configuration: {str(e)}"


def save_config_to_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a UUID.json file with a UUID name."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, f"{uuid.uuid4()}.json")
    with open(config_file, 'w') as f:
        json.dump(settings, f, indent=2)
    return f"Configuration saved to {config_file}"


def save_config_to_default_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a default.json file that will be loaded automatically on startup."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, "default.json")
    with open(config_file, 'w') as f:
        json.dump(settings, f, indent=2)
    return f"Configuration saved to {config_file}"


class ConfigAutosaveManager(ConfigManager):
    """扩展ConfigManager类，添加自动保存功能"""
    
    def __init__(self, autosave_dir="./tmp/webui_settings", default_config_name="default.json"):
        super().__init__()
        self.autosave_dir = autosave_dir
        self.default_config_name = default_config_name
        self.default_config_path = os.path.join(autosave_dir, default_config_name)
        self.component_change_handlers = {}
        
        # 确保保存目录存在
        os.makedirs(autosave_dir, exist_ok=True)
    
    def register_component(self, name: str, component):
        """注册组件并添加变更监听器"""
        super().register_component(name, component)
        
        # 为支持change事件的组件添加自动保存功能
        if hasattr(component, "change"):
            # 存储原始change事件
            self.component_change_handlers[name] = getattr(component, "_event_triggers", {}).get("change", [])
            
            # 添加自动保存回调
            component.change(
                fn=lambda value, _name=name: self._on_component_change(_name, value),
                inputs=[component],
                outputs=[]
            )
            
        return component
    
    def _on_component_change(self, component_name, new_value):
        """组件值变更时自动保存配置"""
        logger = logging.getLogger(__name__)
        try:
            # 记录组件变更
            if "Base URL" in component_name or "API Key" in component_name:
                # 对敏感信息不显示具体内容
                logger.info(f"组件 {component_name} 值已变更")
            else:
                logger.info(f"组件 {component_name} 值已变更为: {new_value}")
                
            # 更新组件的值
            if component_name in self.components:
                component = self.components[component_name]
                if hasattr(component, "value"):
                    old_value = component.value
                    component.value = new_value
                    logger.debug(f"组件 {component_name} 值已从 {old_value} 更新为 {new_value}")
            else:
                logger.warning(f"组件 {component_name} 不在注册列表中，无法更新值")
            
            # 检查是否启用了自动保存 (通过UI组件的值)
            auto_save_enabled = True  # 默认启用
            
            # 遍历所有组件查找自动保存开关
            for name, component in self.components.items():
                if name.endswith("自动保存配置") and hasattr(component, "value"):
                    auto_save_enabled = component.value
                    break
            
            if auto_save_enabled:
                result = self.autosave_current_config()
                logger.info(f"自动保存配置结果: {result}")
            else:
                logger.info("自动保存已禁用，跳过保存")
                
            return None
        except Exception as e:
            logger.error(f"自动保存配置失败: {str(e)}")
            return None
    
    def autosave_current_config(self):
        """自动保存当前配置到默认文件"""
        current_config = {}
        for name in self.component_order:
            component = self.components[name]
            # 获取组件当前值
            current_config[name] = getattr(component, "value", None)
        
        # 保存到默认配置文件
        return save_config_to_default_file(current_config, self.autosave_dir)
    
    def load_default_config(self):
        """加载默认配置文件"""
        if os.path.exists(self.default_config_path):
            return load_config_from_file(self.default_config_path)
        return None
    
    def set_autosave_enabled(self, enabled: bool):
        """启用或禁用自动保存功能"""
        # 查找并更新自动保存组件
        for name, component in self.components.items():
            if name.endswith("自动保存配置") and hasattr(component, "value"):
                component.value = enabled
                
        # 如果启用，立即保存当前配置
        if enabled:
            self.autosave_current_config()
            
        return f"自动保存已{'启用' if enabled else '禁用'}"
    
    def apply_default_config(self):
        """应用默认配置到UI组件"""
        logger = logging.getLogger(__name__)
        
        config = self.load_default_config()
        if not config or not isinstance(config, dict):
            logger.warning("没有找到默认配置或配置格式不正确")
            return None
            
        logger.info(f"正在应用默认配置，共 {len(config)} 个配置项")
        
        # 准备所有组件的更新
        updates = []
        for name in self.component_order:
            if name in config:
                component = self.components[name]
                saved_value = config[name]
                
                # 如果是LLM Provider，打印更详细的信息
                if "LLM Provider" in name:
                    logger.info(f"处理LLM Provider配置: {name} = {saved_value}, 组件类型: {type(component).__name__}")
                    
                # 直接设置组件值
                if hasattr(component, "value"):
                    old_value = component.value
                    component.value = saved_value
                    logger.debug(f"设置组件值: {name} 从 {old_value} 更新为 {saved_value}")
                    
                # 返回更新指令
                updates.append(gr.update(value=saved_value))
            else:
                logger.debug(f"配置中未找到组件: {name}")
                updates.append(gr.update())
                
        return updates
