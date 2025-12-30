"""HA Video Vision - Video-Only AI Camera Analysis for Home Assistant."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Any

import aiofiles
import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    CONF_PROVIDER_CONFIGS,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # vLLM
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    CONF_VLLM_MAX_TOKENS,
    CONF_VLLM_TEMPERATURE,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_MAX_TOKENS,
    DEFAULT_VLLM_TEMPERATURE,
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
    # Services
    SERVICE_ANALYZE_CAMERA,
    SERVICE_RECORD_CLIP,
    SERVICE_IDENTIFY_FACES,
    # Attributes
    ATTR_CAMERA,
    ATTR_DURATION,
    ATTR_USER_QUERY,
    ATTR_NOTIFY,
    ATTR_IMAGE_PATH,
)

_LOGGER = logging.getLogger(__name__)

# Service schemas
SERVICE_ANALYZE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
        vol.Optional(ATTR_USER_QUERY, default=""): cv.string,
        vol.Optional(ATTR_NOTIFY, default=False): cv.boolean,
    }
)

SERVICE_RECORD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
    }
)

SERVICE_IDENTIFY_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_IMAGE_PATH): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the HA Video Vision component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.info("Migrating HA Video Vision config entry from version %s", config_entry.version)

    if config_entry.version < 3:
        new_data = {**config_entry.data}
        new_options = {**config_entry.options}
        
        # Migrate to video-only: default to OpenRouter
        current_provider = new_options.get(CONF_PROVIDER) or new_data.get(CONF_PROVIDER)
        if current_provider not in [PROVIDER_LOCAL, PROVIDER_GOOGLE, PROVIDER_OPENROUTER]:
            new_options[CONF_PROVIDER] = PROVIDER_OPENROUTER

        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            options=new_options,
            version=3,
        )
        _LOGGER.info("Migration to version 3 (video-only) successful")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Video Vision from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Merge data and options
    config = {**entry.data, **entry.options}
    
    # Create the video analyzer instance
    analyzer = VideoAnalyzer(hass, config)
    hass.data[DOMAIN][entry.entry_id] = {
        "config": config,
        "analyzer": analyzer,
    }
    
    # Register services
    async def handle_analyze_camera(call: ServiceCall) -> dict[str, Any]:
        """Handle analyze_camera service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        user_query = call.data.get(ATTR_USER_QUERY, "")
        notify = call.data.get(ATTR_NOTIFY, False)
        
        result = await analyzer.analyze_camera(camera, duration, user_query)
        
        if notify and result.get("success"):
            await analyzer.send_notification(result)
        
        return result

    async def handle_record_clip(call: ServiceCall) -> dict[str, Any]:
        """Handle record_clip service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        
        return await analyzer.record_clip(camera, duration)

    async def handle_identify_faces(call: ServiceCall) -> dict[str, Any]:
        """Handle identify_faces service call."""
        image_path = call.data[ATTR_IMAGE_PATH]
        
        return await analyzer.identify_faces_from_file(image_path)

    # Register services with response support
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_CAMERA,
        handle_analyze_camera,
        schema=SERVICE_ANALYZE_SCHEMA,
        supports_response=True,
    )
    
    hass.services.async_register(
        DOMAIN,
        SERVICE_RECORD_CLIP,
        handle_record_clip,
        schema=SERVICE_RECORD_SCHEMA,
        supports_response=True,
    )
    
    hass.services.async_register(
        DOMAIN,
        SERVICE_IDENTIFY_FACES,
        handle_identify_faces,
        schema=SERVICE_IDENTIFY_SCHEMA,
        supports_response=True,
    )

    # Listen for option updates
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _LOGGER.info("HA Video Vision (Video-Only) integration setup complete")
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    config = {**entry.data, **entry.options}
    hass.data[DOMAIN][entry.entry_id]["config"] = config
    hass.data[DOMAIN][entry.entry_id]["analyzer"].update_config(config)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Remove services
    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_CAMERA)
    hass.services.async_remove(DOMAIN, SERVICE_RECORD_CLIP)
    hass.services.async_remove(DOMAIN, SERVICE_IDENTIFY_FACES)
    
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


class VideoAnalyzer:
    """Class to handle video analysis with video-capable LLM providers and facial recognition."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self._session = async_get_clientsession(hass)
        self.update_config(config)

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        # Get active provider and all provider configs
        self.provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.provider_configs = config.get(CONF_PROVIDER_CONFIGS, {})
        
        # Get settings for the active provider from provider_configs
        active_config = self.provider_configs.get(self.provider, {})
        
        if active_config:
            # Use settings from provider_configs
            self.api_key = active_config.get("api_key", "")
            self.vllm_model = active_config.get("model", PROVIDER_DEFAULT_MODELS.get(self.provider, ""))
            self.base_url = active_config.get("base_url", PROVIDER_BASE_URLS.get(self.provider, ""))
        else:
            # Fallback to legacy config format
            self.api_key = config.get(CONF_API_KEY, "")
            self.vllm_model = config.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(self.provider, DEFAULT_VLLM_MODEL))
            
            # Set base URL based on provider
            if self.provider == PROVIDER_LOCAL:
                self.base_url = config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)
            else:
                self.base_url = PROVIDER_BASE_URLS.get(self.provider, config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL))
        
        # Other vLLM settings
        self.vllm_max_tokens = config.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS)
        self.vllm_temperature = config.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE)
        
        # Facial recognition settings
        self.facial_rec_url = config.get(CONF_FACIAL_REC_URL, DEFAULT_FACIAL_REC_URL)
        self.facial_rec_enabled = config.get(CONF_FACIAL_REC_ENABLED, DEFAULT_FACIAL_REC_ENABLED)
        self.facial_rec_confidence = config.get(CONF_FACIAL_REC_CONFIDENCE, DEFAULT_FACIAL_REC_CONFIDENCE)
        
        # RTSP settings
        self.rtsp_host = config.get(CONF_RTSP_HOST, DEFAULT_RTSP_HOST)
        self.rtsp_port = config.get(CONF_RTSP_PORT, DEFAULT_RTSP_PORT)
        self.rtsp_username = config.get(CONF_RTSP_USERNAME, DEFAULT_RTSP_USERNAME)
        self.rtsp_password = config.get(CONF_RTSP_PASSWORD, DEFAULT_RTSP_PASSWORD)
        self.rtsp_stream_type = config.get(CONF_RTSP_STREAM_TYPE, DEFAULT_RTSP_STREAM_TYPE)
        
        # Camera settings
        self.cameras = config.get(CONF_CAMERAS, DEFAULT_CAMERAS)
        
        # Video settings
        self.video_duration = config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)
        self.video_width = config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)
        self.video_crf = config.get(CONF_VIDEO_CRF, DEFAULT_VIDEO_CRF)
        self.frame_for_facial = config.get(CONF_FRAME_FOR_FACIAL, DEFAULT_FRAME_FOR_FACIAL)
        
        # Snapshot settings
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        
        # Notification settings - Parse service lists
        notify_services = config.get(CONF_NOTIFY_SERVICES, DEFAULT_NOTIFY_SERVICES)
        if isinstance(notify_services, str):
            self.notify_services = [s.strip() for s in notify_services.split(",") if s.strip()]
        else:
            self.notify_services = notify_services or []
        
        # iOS devices - Parse list
        ios_devices = config.get(CONF_IOS_DEVICES, DEFAULT_IOS_DEVICES)
        if isinstance(ios_devices, str):
            self.ios_devices = [s.strip() for s in ios_devices.split(",") if s.strip()]
        else:
            self.ios_devices = ios_devices or []
        
        self.cooldown_seconds = config.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN_SECONDS)
        self.critical_alerts = config.get(CONF_CRITICAL_ALERTS, DEFAULT_CRITICAL_ALERTS)
        
        _LOGGER.info(
            "HA Video Vision config updated - Provider: %s, Model: %s, Cameras: %d",
            self.provider, self.vllm_model, len(self.cameras)
        )

    def _get_rtsp_url(self, camera_name: str) -> str | None:
        """Build RTSP URL for a camera."""
        camera_config = self.cameras.get(camera_name.lower())
        if not camera_config:
            for name, cfg in self.cameras.items():
                if name == camera_name.lower() or cfg.get("friendly_name", "").lower() == camera_name.lower():
                    camera_config = cfg
                    break
        
        if not camera_config:
            return None
        
        channel = camera_config.get("channel", "01")
        stream = "sub" if self.rtsp_stream_type == "sub" else "main"
        
        return f"rtsp://{self.rtsp_username}:{self.rtsp_password}@{self.rtsp_host}:{self.rtsp_port}/h264Preview_{channel}_{stream}"

    async def record_clip(self, camera_name: str, duration: int = None) -> dict[str, Any]:
        """Record a video clip from camera."""
        duration = duration or self.video_duration
        
        rtsp_url = self._get_rtsp_url(camera_name)
        if not rtsp_url:
            return {"success": False, "error": f"Unknown camera: {camera_name}"}
        
        os.makedirs(self.snapshot_dir, exist_ok=True)
        video_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=self.snapshot_dir) as vf:
                video_path = vf.name
            
            cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-t", str(duration),
                "-vf", f"scale={self.video_width}:-2",
                "-r", "10",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", str(self.video_crf),
                "-an",
                video_path
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 15)
            
            if proc.returncode != 0:
                _LOGGER.error("FFmpeg error: %s", stderr.decode() if stderr else "Unknown error")
                return {"success": False, "error": "Failed to record video"}
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return {"success": False, "error": "Video file empty or not created"}
            
            final_path = os.path.join(self.snapshot_dir, f"{camera_name}_clip.mp4")
            os.rename(video_path, final_path)
            video_path = final_path
            
            return {
                "success": True,
                "camera": camera_name,
                "video_path": video_path,
                "duration": duration,
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Recording timed out"}
        except Exception as e:
            _LOGGER.error("Error recording clip: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            if video_path and os.path.exists(video_path) and "clip.mp4" not in video_path:
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    async def _record_video_bytes(self, camera_name: str, duration: int) -> tuple[bytes | None, bytes | None, bytes | None]:
        """Record video and extract frame, return (video_bytes, frame_bytes, facial_frame_bytes)."""
        rtsp_url = self._get_rtsp_url(camera_name)
        if not rtsp_url:
            return None, None, None
        
        video_path = None
        frame_path = None
        facial_frame_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as facial_f:
                facial_frame_path = facial_f.name
            
            # Record video (scaled down for LLM)
            video_cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-t", str(duration),
                "-vf", f"scale={self.video_width}:-2",
                "-r", "10",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", str(self.video_crf),
                "-an",
                video_path
            ]
            
            # Extract first frame (scaled)
            frame_cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-frames:v", "1",
                "-vf", f"scale={self.video_width}:-2",
                "-q:v", "2",
                frame_path
            ]
            
            # Extract HIGH-RES frame for facial recognition
            facial_cmd = [
                "ffmpeg", "-y", "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-frames:v", "1",
                "-vf", f"select=eq(n\\,{self.frame_for_facial})",
                "-q:v", "1",
                facial_frame_path
            ]
            
            video_proc = await asyncio.create_subprocess_exec(
                *video_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            frame_proc = await asyncio.create_subprocess_exec(
                *frame_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            facial_proc = await asyncio.create_subprocess_exec(
                *facial_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            await asyncio.wait_for(video_proc.communicate(), timeout=duration + 15)
            await asyncio.wait_for(frame_proc.wait(), timeout=10)
            await asyncio.wait_for(facial_proc.wait(), timeout=10)
            
            video_bytes = None
            frame_bytes = None
            facial_frame_bytes = None
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                async with aiofiles.open(video_path, 'rb') as f:
                    video_bytes = await f.read()
            
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                async with aiofiles.open(frame_path, 'rb') as f:
                    frame_bytes = await f.read()
            
            if os.path.exists(facial_frame_path) and os.path.getsize(facial_frame_path) > 0:
                async with aiofiles.open(facial_frame_path, 'rb') as f:
                    facial_frame_bytes = await f.read()
            
            return video_bytes, frame_bytes, facial_frame_bytes
            
        except Exception as e:
            _LOGGER.error("Error recording video: %s", e)
            return None, None, None
        finally:
            for path in [video_path, frame_path, facial_frame_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _identify_faces(self, image_bytes: bytes) -> list[dict]:
        """Send image to facial recognition server."""
        if not self.facial_rec_enabled or not self.facial_rec_url:
            return []
        
        try:
            image_b64 = base64.b64encode(image_bytes).decode()
            
            async with asyncio.timeout(15):
                async with self._session.post(
                    f"{self.facial_rec_url}/identify",
                    json={"image_base64": image_b64},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        people = result.get("people", [])
                        
                        identified = [
                            {"name": p["name"], "confidence": p["confidence"]}
                            for p in people
                            if p.get("name") != "Unknown" and p.get("confidence", 0) >= self.facial_rec_confidence
                        ]
                        
                        return identified
                    else:
                        error_text = await response.text()
                        _LOGGER.warning("Facial rec error %s: %s", response.status, error_text[:200])
                        return []
                        
        except asyncio.TimeoutError:
            _LOGGER.warning("Facial recognition timed out")
            return []
        except Exception as e:
            _LOGGER.warning("Facial recognition error: %s", e)
            return []

    async def identify_faces_from_file(self, image_path: str) -> dict[str, Any]:
        """Identify faces from an image file."""
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}
        
        try:
            async with aiofiles.open(image_path, 'rb') as f:
                image_bytes = await f.read()
            
            identified = await self._identify_faces(image_bytes)
            
            return {
                "success": True,
                "image_path": image_path,
                "identified_people": identified,
                "count": len(identified),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_video(
        self, 
        video_bytes: bytes, 
        camera_name: str, 
        identified_people: list[dict], 
        user_query: str = ""
    ) -> str:
        """Send video to LLM provider for analysis."""
        # Route to appropriate provider
        if self.provider == PROVIDER_GOOGLE:
            return await self._analyze_google(video_bytes, camera_name, identified_people, user_query)
        else:
            # OpenAI-compatible providers (Local vLLM, OpenRouter)
            return await self._analyze_openai_compatible(video_bytes, camera_name, identified_people, user_query)

    def _build_prompt(self, camera_name: str, identified_people: list[dict], user_query: str = "") -> str:
        """Build the analysis prompt."""
        identity_context = ""
        if identified_people:
            names = [f"{p['name']} ({p['confidence']}%)" for p in identified_people]
            identity_context = f"Identified: {', '.join(names)}. "
        
        camera_config = self.cameras.get(camera_name.lower(), {})
        friendly_name = camera_config.get("friendly_name", camera_name.title())
        
        if user_query:
            return f"""{identity_context}Based on this view from the {friendly_name} camera, answer: "{user_query}"

Answer directly, then briefly describe what you see. 1-2 sentences. No emojis."""
        else:
            return f"""{identity_context}Describe what you see on the {friendly_name} camera.

Focus on: people, vehicles, packages, unusual activity. If nothing notable, say so briefly. 1-2 sentences. No emojis."""

    async def _analyze_openai_compatible(
        self, 
        video_bytes: bytes, 
        camera_name: str, 
        identified_people: list[dict], 
        user_query: str = ""
    ) -> str:
        """Analyze with OpenAI-compatible API (Local vLLM, OpenRouter) - VIDEO ONLY."""
        try:
            prompt = self._build_prompt(camera_name, identified_people, user_query)
            
            # All providers in this integration support video directly
            video_b64 = base64.b64encode(video_bytes).decode()
            content = [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                {"type": "text", "text": prompt}
            ]
            
            payload = {
                "model": self.vllm_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.provider != PROVIDER_LOCAL:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with asyncio.timeout(90):
                async with self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        _LOGGER.error("API error %s: %s", response.status, error_text[:200])
                        return "Camera analysis unavailable"
                        
        except asyncio.TimeoutError:
            return "Camera analysis timed out"
        except Exception as e:
            _LOGGER.error("Analysis error: %s", e)
            return "Camera analysis failed"

    async def _analyze_google(
        self, 
        video_bytes: bytes, 
        camera_name: str, 
        identified_people: list[dict], 
        user_query: str = ""
    ) -> str:
        """Analyze with Google Gemini API (supports video directly)."""
        try:
            prompt = self._build_prompt(camera_name, identified_people, user_query)
            video_b64 = base64.b64encode(video_bytes).decode()
            
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "video/mp4",
                                "data": video_b64
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": self.vllm_max_tokens,
                    "temperature": self.vllm_temperature,
                }
            }
            
            url = f"{self.base_url}/models/{self.vllm_model}:generateContent?key={self.api_key}"
            
            async with asyncio.timeout(90):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        candidates = result.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    return part["text"].strip()
                        return "No response from Gemini"
                    else:
                        error_text = await response.text()
                        _LOGGER.error("Gemini error %s: %s", response.status, error_text[:200])
                        return "Camera analysis unavailable"
                        
        except asyncio.TimeoutError:
            return "Camera analysis timed out"
        except Exception as e:
            _LOGGER.error("Gemini analysis error: %s", e)
            return "Camera analysis failed"

    async def analyze_camera(
        self, 
        camera_name: str, 
        duration: int = None, 
        user_query: str = ""
    ) -> dict[str, Any]:
        """Full analysis: record video, facial recognition, LLM analysis."""
        duration = duration or self.video_duration
        
        # Check if camera exists
        camera_key = camera_name.lower()
        if camera_key not in self.cameras:
            found = False
            for name, cfg in self.cameras.items():
                if cfg.get("friendly_name", "").lower() == camera_key:
                    camera_key = name
                    found = True
                    break
            if not found:
                return {"success": False, "error": f"Unknown camera: {camera_name}"}
        
        camera_config = self.cameras[camera_key]
        friendly_name = camera_config.get("friendly_name", camera_key.title())
        
        _LOGGER.info("Analyzing %s camera (%s) with %s", camera_key, friendly_name, self.provider)
        
        # Record video and extract frames
        video_bytes, frame_bytes, facial_frame_bytes = await self._record_video_bytes(camera_key, duration)
        
        if not video_bytes:
            return {"success": False, "error": f"Could not record from {friendly_name} camera"}
        
        # Run facial recognition on HIGH-RES frame
        identified_people = []
        if facial_frame_bytes and self.facial_rec_enabled:
            identified_people = await self._identify_faces(facial_frame_bytes)
        
        # Analyze video with LLM provider
        description = await self._analyze_video(video_bytes, camera_key, identified_people, user_query)
        
        # Use high-res frame for snapshot if available
        snapshot_frame = facial_frame_bytes or frame_bytes
        
        # Save snapshot (overwrites previous)
        snapshot_path = None
        snapshot_url = None
        if snapshot_frame:
            try:
                os.makedirs(self.snapshot_dir, exist_ok=True)
                snapshot_filename = f"{camera_key}_latest.jpg"
                snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
                async with aiofiles.open(snapshot_path, 'wb') as f:
                    await f.write(snapshot_frame)
                snapshot_url = f"/media/local/ha_video_vision/{snapshot_filename}"
            except Exception as e:
                _LOGGER.warning("Failed to save snapshot: %s", e)
        
        return {
            "success": True,
            "camera": camera_key,
            "friendly_name": friendly_name,
            "description": description,
            "identified_people": identified_people,
            "snapshot_path": snapshot_path,
            "snapshot_url": snapshot_url,
            "provider_used": self.provider,
        }

    def _is_ios_device(self, service_name: str) -> bool:
        """Check if a notification service is an iOS device."""
        service_lower = service_name.lower()
        for ios_dev in self.ios_devices:
            if ios_dev.lower() in service_lower or service_lower in ios_dev.lower():
                return True
        
        ios_indicators = ['iphone', 'ipad', 'ios', 'apple']
        return any(indicator in service_lower for indicator in ios_indicators)

    async def send_notification(self, result: dict[str, Any]) -> None:
        """Send notification with analysis result - handles iOS and Android differently."""
        if not self.notify_services:
            return
        
        camera = result.get("friendly_name", result.get("camera", "Camera"))
        description = result.get("description", "Motion detected")
        identified = result.get("identified_people", [])
        snapshot_url = result.get("snapshot_url")
        camera_key = result.get("camera", "camera")
        
        # Build title
        if identified:
            if len(identified) == 1:
                title = f"👤 {identified[0]['name']} - {camera}"
            else:
                names = ", ".join(p["name"] for p in identified)
                title = f"👥 {names} - {camera}"
        else:
            title = f"🎥 {camera}"
        
        for service in self.notify_services:
            try:
                parts = service.split(".")
                if len(parts) == 2:
                    domain, service_name = parts
                else:
                    domain = "notify"
                    service_name = service
                
                is_ios = self._is_ios_device(service)
                
                data: dict[str, Any] = {
                    "title": title,
                    "message": description,
                }
                
                notification_data: dict[str, Any] = {
                    "tag": f"ha_video_vision_{camera_key}",
                    "group": f"ha_video_vision_{camera_key}",
                    "ttl": 0,
                    "priority": "high",
                }
                
                if snapshot_url:
                    import time
                    snapshot_with_ts = f"{snapshot_url}?t={int(time.time())}"
                    
                    if is_ios:
                        notification_data["attachment"] = {
                            "url": snapshot_with_ts,
                            "content-type": "jpeg",
                            "hide-thumbnail": False,
                        }
                        notification_data["url"] = snapshot_with_ts
                        notification_data["push"] = {
                            "sound": {
                                "name": "default",
                                "critical": 1 if self.critical_alerts else 0,
                                "volume": 1.0 if self.critical_alerts else 0.5,
                            },
                            "interruption-level": "time-sensitive" if self.critical_alerts else "active",
                        }
                    else:
                        notification_data["image"] = snapshot_with_ts
                        notification_data["clickAction"] = snapshot_with_ts
                        notification_data["channel"] = "ha_video_vision"
                        notification_data["importance"] = "high"
                
                data["data"] = notification_data
                
                await self.hass.services.async_call(
                    domain, service_name, data, blocking=False
                )
                _LOGGER.info("Sent notification to %s (iOS: %s)", service, is_ios)
                
            except Exception as e:
                _LOGGER.error("Failed to send notification to %s: %s", service, e)
