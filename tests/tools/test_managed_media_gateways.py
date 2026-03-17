import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tool_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, TOOLS_DIR / filename)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_tools_package():
    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package
    sys.modules["tools.debug_helpers"] = types.SimpleNamespace(
        DebugSession=lambda *args, **kwargs: types.SimpleNamespace(
            active=False,
            session_id="debug-session",
            log_call=lambda *a, **k: None,
            save=lambda: None,
            get_session_info=lambda: {},
        )
    )
    sys.modules["tools.managed_tool_gateway"] = _load_tool_module(
        "tools.managed_tool_gateway",
        "managed_tool_gateway.py",
    )


def _install_fake_fal_client(captured):
    client_module = types.SimpleNamespace(
        FAL_QUEUE_RUN_HOST="queue.fal.run",
        QUEUE_RUN_URL=None,
        QUEUE_URL_FORMAT="https://queue.fal.run/",
    )
    auth_module = types.SimpleNamespace(FAL_QUEUE_RUN_HOST="queue.fal.run")

    def submit(model, arguments=None, headers=None):
        raise AssertionError("managed FAL gateway mode should use fal_client.SyncClient")

    class SyncClient:
        def __init__(self, key=None):
            captured["client_key"] = key

        def submit(self, model, arguments=None, headers=None):
            captured["submit_via"] = "sync_client"
            captured["model"] = model
            captured["arguments"] = arguments
            captured["headers"] = headers
            captured["queue_run_host"] = client_module.FAL_QUEUE_RUN_HOST
            captured["queue_run_url"] = client_module.QUEUE_RUN_URL
            captured["queue_url_format"] = client_module.QUEUE_URL_FORMAT
            captured["auth_queue_run_host"] = auth_module.FAL_QUEUE_RUN_HOST
            return types.SimpleNamespace(get=lambda: {"images": []})

    fal_client_module = types.SimpleNamespace(
        submit=submit,
        SyncClient=SyncClient,
        client=client_module,
        auth=auth_module,
    )
    sys.modules["fal_client"] = fal_client_module
    return fal_client_module


def _install_fake_openai_module(captured, transcription_response=None):
    class FakeSpeechResponse:
        def stream_to_file(self, output_path):
            captured["stream_to_file"] = output_path

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

            def create_speech(**kwargs):
                captured["speech_kwargs"] = kwargs
                return FakeSpeechResponse()

            def create_transcription(**kwargs):
                captured["transcription_kwargs"] = kwargs
                return transcription_response

            self.audio = types.SimpleNamespace(
                speech=types.SimpleNamespace(
                    create=create_speech
                ),
                transcriptions=types.SimpleNamespace(
                    create=create_transcription
                ),
            )

    fake_module = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=Exception,
        APIConnectionError=Exception,
        APITimeoutError=Exception,
    )
    sys.modules["openai"] = fake_module


def test_managed_fal_submit_uses_gateway_origin_and_nous_token(monkeypatch):
    captured = {}
    _install_fake_tools_package()
    _install_fake_fal_client(captured)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("FAL_QUEUE_GATEWAY_URL", "http://127.0.0.1:3009")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    image_generation_tool = _load_tool_module(
        "tools.image_generation_tool",
        "image_generation_tool.py",
    )
    image_generation_tool._submit_fal_request(
        "fal-ai/flux-2-pro",
        {"prompt": "test prompt", "num_images": 1},
    )

    assert captured["model"] == "fal-ai/flux-2-pro"
    assert captured["submit_via"] == "sync_client"
    assert captured["client_key"] == "nous-token"
    assert captured["queue_run_host"] == "127.0.0.1:3009"
    assert captured["queue_run_url"] == "http://127.0.0.1:3009"
    assert captured["queue_url_format"] == "http://127.0.0.1:3009/"
    assert captured["auth_queue_run_host"] == "127.0.0.1:3009"
    assert sys.modules["fal_client"].client.QUEUE_RUN_URL is None
    assert sys.modules["fal_client"].client.QUEUE_URL_FORMAT == "https://queue.fal.run/"
    assert sys.modules["fal_client"].auth.FAL_QUEUE_RUN_HOST == "queue.fal.run"
    assert captured["headers"] is None


def test_openai_tts_uses_managed_audio_gateway_when_direct_key_absent(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts("hello world", str(output_path), {"openai": {}})

    assert captured["api_key"] == "nous-token"
    assert captured["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert captured["speech_kwargs"]["model"] == "gpt-4o-mini-tts"
    assert captured["stream_to_file"] == str(output_path)


def test_transcription_uses_model_specific_response_formats(monkeypatch, tmp_path):
    whisper_capture = {}
    _install_fake_tools_package()
    _install_fake_openai_module(whisper_capture, transcription_response="hello from whisper")
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    transcription_tools = _load_tool_module(
        "tools.transcription_tools",
        "transcription_tools.py",
    )
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF0000WAVEfmt ")

    whisper_result = transcription_tools.transcribe_audio(str(audio_path), model="whisper-1")
    assert whisper_result["success"] is True
    assert whisper_capture["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert whisper_capture["transcription_kwargs"]["response_format"] == "text"

    json_capture = {}
    _install_fake_openai_module(
        json_capture,
        transcription_response=types.SimpleNamespace(text="hello from gpt-4o"),
    )
    transcription_tools = _load_tool_module(
        "tools.transcription_tools",
        "transcription_tools.py",
    )

    json_result = transcription_tools.transcribe_audio(
        str(audio_path),
        model="gpt-4o-mini-transcribe",
    )
    assert json_result["success"] is True
    assert json_result["transcript"] == "hello from gpt-4o"
    assert json_capture["transcription_kwargs"]["response_format"] == "json"
