"""Config flow for HA Video Vision - Auto-Discovery."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    ALL_PROVIDERS,
    PROVIDER_NAMES,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # AI Settings
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    CONF_VLLM_MAX_TOKENS,
    CONF_VLLM_TEMPERATURE,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_MAX_TOKENS,
    DEFAULT_VLLM_TEMPERATURE,
    # Cameras
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    CONF_CAMERA_ALIASES,
    DEFAULT_CAMERA_ALIASES,
    # Facial Recognition
    CONF_FACIAL_REC_URL,
    CONF_FACIAL_REC_ENABLED,
    CONF_FACIAL_REC_CONFIDENCE,
    DEFAULT_FACIAL_REC_URL,
    DEFAULT_FACIAL_REC_ENABLED,
    DEFAULT_FACIAL_REC_CONFIDENCE,
    # Video Settings
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    CONF_SNAPSHOT_QUALITY,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_QUALITY,
)

_LOGGER = logging.getLogger(__name__)


class VideoVisionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA Video Vision."""

    VERSION = 4

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
                return await self.async_step_cameras()
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

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera selection step - Auto-discovered!"""
        if user_input is not None:
            self._data[CONF_SELECTED_CAMERAS] = user_input.get(CONF_SELECTED_CAMERAS, [])

            # Create the config entry
            return self.async_create_entry(
                title="HA Video Vision",
                data=self._data,
            )

        # Auto-discover all camera entities
        camera_entities = []
        for state in self.hass.states.async_all("camera"):
            friendly_name = state.attributes.get("friendly_name", state.entity_id)
            camera_entities.append(
                selector.SelectOptionDict(
                    value=state.entity_id,
                    label=f"{friendly_name} ({state.entity_id})"
                )
            )

        if not camera_entities:
            # No cameras found - show message
            return self.async_show_form(
                step_id="cameras",
                data_schema=vol.Schema({
                    vol.Optional(CONF_SELECTED_CAMERAS, default=[]): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[],
                            multiple=True,
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }),
                description_placeholders={
                    "camera_count": "0",
                    "camera_hint": "No cameras found. Add camera integrations first.",
                },
            )

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema({
                vol.Required(CONF_SELECTED_CAMERAS, default=[]): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=camera_entities,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }),
            description_placeholders={
                "camera_count": str(len(camera_entities)),
                "camera_hint": "Select which cameras to enable for AI analysis.",
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return VideoVisionOptionsFlow(config_entry)


class VideoVisionOptionsFlow(config_entries.OptionsFlow):
    """Handle options for HA Video Vision."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "provider",
                "cameras",
                "voice_aliases",
                "facial_rec",
                "video_quality",
                "ai_settings",
            ],
        )

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle provider settings."""
        errors = {}
        current = {**self._entry.data, **self._entry.options}

        if user_input is not None:
            provider = user_input.get(CONF_PROVIDER, current.get(CONF_PROVIDER))
            valid = await self._test_provider_connection(provider, user_input)
            if valid:
                new_options = {**self._entry.options, **user_input}
                return self.async_create_entry(title="", data=new_options)
            else:
                errors["base"] = "cannot_connect"

        provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        if provider == PROVIDER_LOCAL:
            schema = vol.Schema({
                vol.Required(CONF_PROVIDER, default=provider): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_VLLM_URL, default=current.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)): str,
                vol.Required(CONF_VLLM_MODEL, default=current.get(CONF_VLLM_MODEL, DEFAULT_VLLM_MODEL)): str,
            })
        else:
            schema = vol.Schema({
                vol.Required(CONF_PROVIDER, default=provider): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=provider_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_API_KEY, default=current.get(CONF_API_KEY, "")): str,
                vol.Optional(CONF_VLLM_MODEL, default=current.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(provider, ""))): str,
            })

        return self.async_show_form(
            step_id="provider",
            data_schema=schema,
            errors=errors,
        )

    async def _test_provider_connection(
        self, provider: str, config: dict[str, Any]
    ) -> bool:
        """Test connection to the selected provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_LOCAL:
                    url = f"{config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)}/models"
                    headers = {}
                elif provider == PROVIDER_GOOGLE:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config.get(CONF_API_KEY, '')}"
                    headers = {}
                elif provider == PROVIDER_OPENROUTER:
                    url = "https://openrouter.ai/api/v1/models"
                    headers = {"Authorization": f"Bearer {config.get(CONF_API_KEY, '')}"}
                else:
                    return False

                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def async_step_cameras(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera selection - Auto-discovered!"""
        if user_input is not None:
            new_options = {**self._entry.options}
            new_options[CONF_SELECTED_CAMERAS] = user_input.get(CONF_SELECTED_CAMERAS, [])
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        selected = current.get(CONF_SELECTED_CAMERAS, [])

        # Auto-discover all camera entities
        camera_entities = []
        for state in self.hass.states.async_all("camera"):
            friendly_name = state.attributes.get("friendly_name", state.entity_id)
            camera_entities.append(
                selector.SelectOptionDict(
                    value=state.entity_id,
                    label=f"{friendly_name} ({state.entity_id})"
                )
            )

        return self.async_show_form(
            step_id="cameras",
            data_schema=vol.Schema({
                vol.Required(CONF_SELECTED_CAMERAS, default=selected): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=camera_entities,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }),
            description_placeholders={
                "camera_count": str(len(camera_entities)),
            },
        )

    async def async_step_voice_aliases(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle voice alias configuration."""
        if user_input is not None:
            # Parse aliases from text
            aliases = {}
            alias_text = user_input.get("alias_config", "")

            for line in alias_text.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    voice_name = parts[0].strip().lower()
                    camera_id = parts[1].strip()
                    if voice_name and camera_id:
                        aliases[voice_name] = camera_id

            new_options = {**self._entry.options, CONF_CAMERA_ALIASES: aliases}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        aliases = current.get(CONF_CAMERA_ALIASES, DEFAULT_CAMERA_ALIASES)
        selected_cameras = current.get(CONF_SELECTED_CAMERAS, [])

        # Build current alias text
        alias_lines = [f"{name}:{camera}" for name, camera in aliases.items()]
        alias_text = "\n".join(alias_lines) if alias_lines else ""

        # Build hint showing available cameras
        camera_hints = []
        for entity_id in selected_cameras:
            state = self.hass.states.get(entity_id)
            if state:
                friendly = state.attributes.get("friendly_name", entity_id)
                camera_hints.append(f"{entity_id} ({friendly})")

        return self.async_show_form(
            step_id="voice_aliases",
            data_schema=vol.Schema({
                vol.Optional("alias_config", default=alias_text): selector.TextSelector(
                    selector.TextSelectorConfig(
                        multiline=True,
                    )
                ),
            }),
            description_placeholders={
                "available_cameras": "\n".join(camera_hints) if camera_hints else "No cameras selected",
            },
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

    async def async_step_video_quality(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle video/image quality settings."""
        if user_input is not None:
            # Convert width to int
            if CONF_VIDEO_WIDTH in user_input:
                user_input[CONF_VIDEO_WIDTH] = int(user_input[CONF_VIDEO_WIDTH])

            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="video_quality",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_VIDEO_DURATION,
                        default=current.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            step=1,
                            unit_of_measurement="seconds",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(
                        CONF_VIDEO_WIDTH,
                        default=str(current.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"label": "480p", "value": "480"},
                                {"label": "640p", "value": "640"},
                                {"label": "720p", "value": "720"},
                                {"label": "1080p", "value": "1080"},
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_SNAPSHOT_QUALITY,
                        default=current.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=50,
                            max=100,
                            step=5,
                            unit_of_measurement="%",
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_SNAPSHOT_DIR,
                        default=current.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR),
                    ): str,
                }
            ),
        )

    async def async_step_ai_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle AI response settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="ai_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_VLLM_MAX_TOKENS,
                        default=current.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=50,
                            max=500,
                            step=10,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Required(
                        CONF_VLLM_TEMPERATURE,
                        default=current.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            description_placeholders={
                "temp_hint": "Lower = more consistent/factual. Higher = more creative.",
            },
        )
