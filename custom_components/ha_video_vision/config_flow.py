"""Config flow for HA Video Vision - Video Only."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    CONF_PROVIDER_CONFIGS,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    ALL_PROVIDERS,
    PROVIDER_NAMES,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # vLLM
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    # Facial Recognition
    CONF_FACIAL_REC_URL,
    CONF_FACIAL_REC_ENABLED,
    CONF_FACIAL_REC_CONFIDENCE,
    DEFAULT_FACIAL_REC_URL,
    DEFAULT_FACIAL_REC_ENABLED,
    DEFAULT_FACIAL_REC_CONFIDENCE,
    # RTSP
    CONF_RTSP_HOST,
    CONF_RTSP_PORT,
    CONF_RTSP_USERNAME,
    CONF_RTSP_PASSWORD,
    CONF_RTSP_STREAM_TYPE,
    DEFAULT_RTSP_HOST,
    DEFAULT_RTSP_PORT,
    DEFAULT_RTSP_USERNAME,
    DEFAULT_RTSP_PASSWORD,
    DEFAULT_RTSP_STREAM_TYPE,
    # Cameras
    CONF_CAMERAS,
    DEFAULT_CAMERAS,
    # Video
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    CONF_VIDEO_CRF,
    CONF_FRAME_FOR_FACIAL,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_CRF,
    DEFAULT_FRAME_FOR_FACIAL,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_DIR,
    # Notifications
    CONF_NOTIFY_SERVICES,
    CONF_IOS_DEVICES,
    CONF_COOLDOWN_SECONDS,
    CONF_CRITICAL_ALERTS,
    DEFAULT_NOTIFY_SERVICES,
    DEFAULT_IOS_DEVICES,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_CRITICAL_ALERTS,
)

_LOGGER = logging.getLogger(__name__)


class VideoVisionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA Video Vision."""

    VERSION = 3

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - Provider selection."""
        if user_input is not None:
            self._data[CONF_PROVIDER] = user_input[CONF_PROVIDER]
            return await self.async_step_credentials()

        # Build provider options
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_PROVIDER, default=PROVIDER_OPENROUTER): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }),
        )

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials step based on provider."""
        errors = {}
        provider = self._data.get(CONF_PROVIDER, PROVIDER_OPENROUTER)

        if user_input is not None:
            valid = await self._test_provider_connection(provider, user_input)
            if valid:
                self._data.update(user_input)
                return await self.async_step_rtsp()
            else:
                errors["base"] = "cannot_connect"

        # Build schema based on provider
        if provider == PROVIDER_LOCAL:
            schema = vol.Schema({
                vol.Required(CONF_VLLM_URL, default=DEFAULT_VLLM_URL): str,
                vol.Required(CONF_VLLM_MODEL, default=PROVIDER_DEFAULT_MODELS[provider]): str,
            })
        else:
            schema = vol.Schema({
                vol.Required(CONF_API_KEY): str,
                vol.Optional(CONF_VLLM_MODEL, default=PROVIDER_DEFAULT_MODELS[provider]): str,
            })

        return self.async_show_form(
            step_id="credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "provider_name": PROVIDER_NAMES[provider],
                "default_model": PROVIDER_DEFAULT_MODELS[provider],
            },
        )

    async def _test_provider_connection(
        self, provider: str, config: dict[str, Any]
    ) -> bool:
        """Test connection to the selected provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_LOCAL:
                    url = f"{config[CONF_VLLM_URL]}/models"
                    headers = {}
                elif provider == PROVIDER_GOOGLE:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config[CONF_API_KEY]}"
                    headers = {}
                elif provider == PROVIDER_OPENROUTER:
                    url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config[CONF_API_KEY]}"}
                else:
                    return False

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception as e:
            _LOGGER.warning("Provider connection test failed: %s", e)
            return False

    async def async_step_rtsp(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle RTSP configuration step."""
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_cameras()

        return self.async_show_form(
            step_id="rtsp",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_RTSP_HOST, default=DEFAULT_RTSP_HOST): str,
                    vol.Required(CONF_RTSP_PORT, default=DEFAULT_RTSP_PORT): int,
                    vol.Required(CONF_RTSP_USERNAME, default=DEFAULT_RTSP_USERNAME): str,
                    vol.Required(CONF_RTSP_PASSWORD, default=DEFAULT_RTSP_PASSWORD): str,
                    vol.Required(CONF_RTSP_STREAM_TYPE, default=DEFAULT_RTSP_STREAM_TYPE): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"label": "Sub Stream (Lower Quality, Faster)", "value": "sub"},
                                {"label": "Main Stream (Full Quality)", "value": "main"},
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera configuration step."""
        if user_input is not None:
            cameras = {}
            camera_text = user_input.get("camera_config", "")
            
            for line in camera_text.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":")
                if len(parts) >= 2:
                    name = parts[0].strip().lower()
                    channel = parts[1].strip()
                    friendly = parts[2].strip() if len(parts) > 2 else name.title()
                    cameras[name] = {
                        "channel": channel.zfill(2),
                        "friendly_name": friendly,
                        "ha_entity": "",
                    }
            
            self._data[CONF_CAMERAS] = cameras
            return await self.async_step_facial_rec()

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema(
                {
                    vol.Required("camera_config", default="camera1:01:Camera 1"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                        )
                    ),
                }
            ),
        )

    async def async_step_facial_rec(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle facial recognition configuration step."""
        if user_input is not None:
            self._data.update(user_input)
            
            # Build provider configs
            provider = self._data.get(CONF_PROVIDER, PROVIDER_OPENROUTER)
            provider_configs = {}
            
            if provider == PROVIDER_LOCAL:
                provider_configs[PROVIDER_LOCAL] = {
                    "api_key": "",
                    "model": self._data.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS[provider]),
                    "base_url": self._data.get(CONF_VLLM_URL, DEFAULT_VLLM_URL),
                }
            else:
                provider_configs[provider] = {
                    "api_key": self._data.get(CONF_API_KEY, ""),
                    "model": self._data.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS[provider]),
                    "base_url": PROVIDER_BASE_URLS[provider],
                }
            
            self._data[CONF_PROVIDER_CONFIGS] = provider_configs
            
            return self.async_create_entry(
                title="HA Video Vision",
                data=self._data,
            )

        return self.async_show_form(
            step_id="facial_rec",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_FACIAL_REC_ENABLED, default=DEFAULT_FACIAL_REC_ENABLED): bool,
                    vol.Required(CONF_FACIAL_REC_URL, default=DEFAULT_FACIAL_REC_URL): str,
                    vol.Required(CONF_FACIAL_REC_CONFIDENCE, default=DEFAULT_FACIAL_REC_CONFIDENCE): vol.All(
                        vol.Coerce(int), vol.Range(min=0, max=100)
                    ),
                }
            ),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return VideoVisionOptionsFlow(config_entry)


class VideoVisionOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for HA Video Vision."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "providers": "Vision Providers",
                "rtsp": "RTSP/Camera Connection",
                "cameras": "Camera Channels",
                "facial_rec": "Facial Recognition",
                "video": "Video Recording Settings",
                "notifications": "Notification Settings",
            },
        )

    async def async_step_providers(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle all provider settings on one page."""
        if user_input is not None:
            active_provider = user_input.get("active_provider", PROVIDER_OPENROUTER)
            provider_configs = {}
            
            # Local vLLM
            if user_input.get("local_url"):
                provider_configs[PROVIDER_LOCAL] = {
                    "api_key": "",
                    "model": user_input.get("local_model") or PROVIDER_DEFAULT_MODELS[PROVIDER_LOCAL],
                    "base_url": user_input.get("local_url"),
                }
            
            # Google Gemini
            if user_input.get("google_key"):
                provider_configs[PROVIDER_GOOGLE] = {
                    "api_key": user_input.get("google_key"),
                    "model": user_input.get("google_model") or PROVIDER_DEFAULT_MODELS[PROVIDER_GOOGLE],
                    "base_url": PROVIDER_BASE_URLS[PROVIDER_GOOGLE],
                }
            
            # OpenRouter
            if user_input.get("openrouter_key"):
                provider_configs[PROVIDER_OPENROUTER] = {
                    "api_key": user_input.get("openrouter_key"),
                    "model": user_input.get("openrouter_model") or PROVIDER_DEFAULT_MODELS[PROVIDER_OPENROUTER],
                    "base_url": PROVIDER_BASE_URLS[PROVIDER_OPENROUTER],
                }
            
            new_options = {
                **self._entry.options,
                CONF_PROVIDER: active_provider,
                CONF_PROVIDER_CONFIGS: provider_configs,
            }
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        configs = current.get(CONF_PROVIDER_CONFIGS, {})
        active = current.get(CONF_PROVIDER, PROVIDER_OPENROUTER)

        # Build provider options for dropdown
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="providers",
            data_schema=vol.Schema(
                {
                    # Active provider dropdown
                    vol.Required(
                        "active_provider",
                        default=active,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    # Local vLLM
                    vol.Optional(
                        "local_url",
                        default=configs.get(PROVIDER_LOCAL, {}).get("base_url", current.get(CONF_VLLM_URL, "")),
                    ): str,
                    vol.Optional(
                        "local_model",
                        default=configs.get(PROVIDER_LOCAL, {}).get("model", current.get(CONF_VLLM_MODEL, "")),
                    ): str,
                    # Google Gemini
                    vol.Optional(
                        "google_key",
                        default=configs.get(PROVIDER_GOOGLE, {}).get("api_key", ""),
                    ): str,
                    vol.Optional(
                        "google_model",
                        default=configs.get(PROVIDER_GOOGLE, {}).get("model", PROVIDER_DEFAULT_MODELS[PROVIDER_GOOGLE]),
                    ): str,
                    # OpenRouter
                    vol.Optional(
                        "openrouter_key",
                        default=configs.get(PROVIDER_OPENROUTER, {}).get("api_key", ""),
                    ): str,
                    vol.Optional(
                        "openrouter_model",
                        default=configs.get(PROVIDER_OPENROUTER, {}).get("model", PROVIDER_DEFAULT_MODELS[PROVIDER_OPENROUTER]),
                    ): str,
                }
            ),
        )

    async def async_step_rtsp(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle RTSP settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="rtsp",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_RTSP_HOST,
                        default=current.get(CONF_RTSP_HOST, DEFAULT_RTSP_HOST),
                    ): str,
                    vol.Required(
                        CONF_RTSP_PORT,
                        default=current.get(CONF_RTSP_PORT, DEFAULT_RTSP_PORT),
                    ): int,
                    vol.Required(
                        CONF_RTSP_USERNAME,
                        default=current.get(CONF_RTSP_USERNAME, DEFAULT_RTSP_USERNAME),
                    ): str,
                    vol.Required(
                        CONF_RTSP_PASSWORD,
                        default=current.get(CONF_RTSP_PASSWORD, DEFAULT_RTSP_PASSWORD),
                    ): str,
                    vol.Required(
                        CONF_RTSP_STREAM_TYPE,
                        default=current.get(CONF_RTSP_STREAM_TYPE, DEFAULT_RTSP_STREAM_TYPE),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"label": "Sub Stream (Lower Quality, Faster)", "value": "sub"},
                                {"label": "Main Stream (Full Quality)", "value": "main"},
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera settings."""
        if user_input is not None:
            cameras = {}
            camera_text = user_input.get("camera_config", "")
            
            for line in camera_text.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":")
                if len(parts) >= 2:
                    name = parts[0].strip().lower()
                    channel = parts[1].strip()
                    friendly = parts[2].strip() if len(parts) > 2 else name.title()
                    cameras[name] = {
                        "channel": channel.zfill(2),
                        "friendly_name": friendly,
                        "ha_entity": "",
                    }
            
            new_options = {**self._entry.options, CONF_CAMERAS: cameras}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        cameras = current.get(CONF_CAMERAS, DEFAULT_CAMERAS)
        
        camera_text = "\n".join([
            f"{name}:{config['channel']}:{config['friendly_name']}"
            for name, config in cameras.items()
        ])

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema(
                {
                    vol.Required("camera_config", default=camera_text): selector.TextSelector(
                        selector.TextSelectorConfig(
                            multiline=True,
                        )
                    ),
                }
            ),
        )

    async def async_step_facial_rec(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle facial recognition settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="facial_rec",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_FACIAL_REC_ENABLED,
                        default=current.get(CONF_FACIAL_REC_ENABLED, DEFAULT_FACIAL_REC_ENABLED),
                    ): bool,
                    vol.Required(
                        CONF_FACIAL_REC_URL,
                        default=current.get(CONF_FACIAL_REC_URL, DEFAULT_FACIAL_REC_URL),
                    ): str,
                    vol.Required(
                        CONF_FACIAL_REC_CONFIDENCE,
                        default=current.get(CONF_FACIAL_REC_CONFIDENCE, DEFAULT_FACIAL_REC_CONFIDENCE),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
                }
            ),
        )

    async def async_step_video(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle video recording settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="video",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_VIDEO_DURATION,
                        default=current.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
                    vol.Optional(
                        CONF_VIDEO_WIDTH,
                        default=current.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH),
                    ): vol.All(vol.Coerce(int), vol.Range(min=320, max=1920)),
                    vol.Optional(
                        CONF_VIDEO_CRF,
                        default=current.get(CONF_VIDEO_CRF, DEFAULT_VIDEO_CRF),
                    ): vol.All(vol.Coerce(int), vol.Range(min=18, max=35)),
                    vol.Optional(
                        CONF_FRAME_FOR_FACIAL,
                        default=current.get(CONF_FRAME_FOR_FACIAL, DEFAULT_FRAME_FOR_FACIAL),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=90)),
                    vol.Optional(
                        CONF_SNAPSHOT_DIR,
                        default=current.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR),
                    ): str,
                }
            ),
        )

    async def async_step_notifications(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle notification settings with iOS/Android support."""
        if user_input is not None:
            # Parse notify services (all devices)
            services = [s.strip() for s in user_input.get("notify_services_text", "").split(",") if s.strip()]
            user_input[CONF_NOTIFY_SERVICES] = services
            del user_input["notify_services_text"]
            
            # Parse iOS devices (for proper image handling)
            ios_devices = [s.strip() for s in user_input.get("ios_devices_text", "").split(",") if s.strip()]
            user_input[CONF_IOS_DEVICES] = ios_devices
            del user_input["ios_devices_text"]
            
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        
        # Get current notify services
        notify_services = current.get(CONF_NOTIFY_SERVICES, DEFAULT_NOTIFY_SERVICES)
        notify_text = ", ".join(notify_services) if notify_services else ""
        
        # Get current iOS devices
        ios_devices = current.get(CONF_IOS_DEVICES, DEFAULT_IOS_DEVICES)
        ios_text = ", ".join(ios_devices) if ios_devices else ""

        return self.async_show_form(
            step_id="notifications",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "notify_services_text",
                        default=notify_text,
                    ): str,
                    vol.Optional(
                        "ios_devices_text",
                        default=ios_text,
                    ): str,
                    vol.Optional(
                        CONF_COOLDOWN_SECONDS,
                        default=current.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN_SECONDS),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=3600)),
                    vol.Optional(
                        CONF_CRITICAL_ALERTS,
                        default=current.get(CONF_CRITICAL_ALERTS, DEFAULT_CRITICAL_ALERTS),
                    ): bool,
                }
            ),
        )
