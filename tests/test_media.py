import pytest

from kel.budget import Budget, BudgetExceededError, BudgetTracker
from kel.media import (
    FalJobHandle,
    FalMediaModel,
    FalSTTProvider,
    FalTTSProvider,
    get_audio_model,
    get_image_model,
    get_video_model,
)
from kel.media.registry import _PROVIDERS, register_media_provider
from kel.media.types import MediaResult
from kel.models.errors import AuthenticationError, ProviderError, RateLimitError


class FakeFalClient:
    def __init__(self, response):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    def run(self, endpoint, arguments):
        self.calls.append((endpoint, arguments))
        return self._response


class FakeAsyncFalClient:
    def __init__(self, response):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    async def run(self, endpoint, arguments):
        self.calls.append((endpoint, arguments))
        return self._response


def test_media_result_extracts_every_url_recursively():
    result = MediaResult(
        {
            "images": [{"url": "https://fal.example/a.png"}, {"url": "https://fal.example/b.png"}],
            "seed": 42,
            "nested": {"video": {"url": "https://fal.example/c.mp4"}},
        }
    )

    assert result.urls == ["https://fal.example/a.png", "https://fal.example/b.png", "https://fal.example/c.mp4"]
    assert result.raw["seed"] == 42


def test_fal_media_model_generate_calls_injected_client_with_arguments():
    client = FakeFalClient({"images": [{"url": "https://fal.example/out.png"}]})
    model = FalMediaModel("fal-ai/flux/schnell", client=client)

    result = model.generate(prompt="a cat in space", image_size="square_hd")

    assert client.calls == [("fal-ai/flux/schnell", {"prompt": "a cat in space", "image_size": "square_hd"})]
    assert result.urls == ["https://fal.example/out.png"]


async def test_fal_media_model_agenerate_calls_injected_async_client():
    client = FakeAsyncFalClient({"video": {"url": "https://fal.example/out.mp4"}})
    model = FalMediaModel("fal-ai/kling-video/v1.6/standard/text-to-video", client=client)

    result = await model.agenerate(prompt="a cat flying", duration=5)

    assert client.calls == [("fal-ai/kling-video/v1.6/standard/text-to-video", {"prompt": "a cat flying", "duration": 5})]
    assert result.urls == ["https://fal.example/out.mp4"]


def test_get_image_model_resolves_fal_spec_and_forwards_client():
    client = FakeFalClient({"images": [{"url": "https://fal.example/img.png"}]})

    model = get_image_model("fal:fal-ai/flux/schnell", client=client)

    assert isinstance(model, FalMediaModel)
    assert model.endpoint == "fal-ai/flux/schnell"
    result = model.generate(prompt="hi")
    assert result.urls == ["https://fal.example/img.png"]


def test_get_video_model_and_get_audio_model_resolve_the_same_way():
    video_model = get_video_model("fal:fal-ai/kling-video/v1.6/standard/text-to-video", client=FakeFalClient({}))
    audio_model = get_audio_model("fal:fal-ai/elevenlabs/tts/turbo-v2.5", client=FakeFalClient({}))

    assert video_model.endpoint == "fal-ai/kling-video/v1.6/standard/text-to-video"
    assert audio_model.endpoint == "fal-ai/elevenlabs/tts/turbo-v2.5"


def test_get_image_model_rejects_a_spec_without_a_provider_prefix():
    with pytest.raises(ValueError, match="provider:model_ref"):
        get_image_model("fal-ai/flux/schnell")


def test_get_image_model_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown media provider"):
        get_image_model("unknown-vendor:some-model")


def test_register_media_provider_adds_a_new_provider_prefix():
    calls = []

    def factory(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return ("fake-model", model_ref)

    register_media_provider("fake-vendor", factory)
    try:
        model = get_image_model("fake-vendor:some-model", api_key="k")
        assert model == ("fake-model", "some-model")
        assert calls == [("some-model", {"api_key": "k", "client": None})]
    finally:
        del _PROVIDERS["fake-vendor"]


def test_fal_tts_provider_synthesizes_by_downloading_the_returned_url():
    client = FakeFalClient({"audio": {"url": "https://fal.example/out.wav"}})
    media_model = FalMediaModel("fal-ai/elevenlabs/tts/turbo-v2.5", client=client)
    downloaded = []

    def fake_download(url):
        downloaded.append(url)
        return b"fake-audio-bytes"

    tts = FalTTSProvider(media_model, download=fake_download)

    audio_bytes = tts.synthesize("hello world")

    assert client.calls == [("fal-ai/elevenlabs/tts/turbo-v2.5", {"text": "hello world"})]
    assert downloaded == ["https://fal.example/out.wav"]
    assert audio_bytes == b"fake-audio-bytes"


def test_fal_tts_provider_raises_a_clear_error_when_no_url_comes_back():
    client = FakeFalClient({"seed": 1})
    media_model = FalMediaModel("fal-ai/elevenlabs/tts/turbo-v2.5", client=client)
    tts = FalTTSProvider(media_model, download=lambda url: b"")

    with pytest.raises(ValueError, match="no audio URL"):
        tts.synthesize("hello")


def test_fal_stt_provider_uploads_then_transcribes():
    client = FakeFalClient({"text": "hello world"})
    media_model = FalMediaModel("fal-ai/whisper", client=client)
    uploaded = []

    def fake_upload(audio_bytes):
        uploaded.append(audio_bytes)
        return "https://fal.example/uploaded.wav"

    stt = FalSTTProvider(media_model, fake_upload)

    result = stt.transcribe(b"raw-audio-bytes")

    assert uploaded == [b"raw-audio-bytes"]
    assert client.calls == [("fal-ai/whisper", {"audio_url": "https://fal.example/uploaded.wav"})]
    assert result.text == "hello world"
    assert result.is_final is True


class _StatusCodeError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeErrorClient:
    def __init__(self, exc):
        self._exc = exc

    def run(self, endpoint, arguments):
        raise self._exc


def test_generate_translates_a_401_into_authentication_error():
    model = FalMediaModel("fal-ai/flux/schnell", client=FakeErrorClient(_StatusCodeError("bad key", 401)))
    with pytest.raises(AuthenticationError):
        model.generate(prompt="hi")


def test_generate_translates_a_429_into_rate_limit_error():
    model = FalMediaModel("fal-ai/flux/schnell", client=FakeErrorClient(_StatusCodeError("slow down", 429)))
    with pytest.raises(RateLimitError):
        model.generate(prompt="hi")


def test_generate_translates_an_unknown_error_into_provider_error():
    model = FalMediaModel("fal-ai/flux/schnell", client=FakeErrorClient(RuntimeError("boom")))
    with pytest.raises(ProviderError):
        model.generate(prompt="hi")


def test_generate_charges_a_fixed_cost_against_a_budget_tracker():
    client = FakeFalClient({"images": [{"url": "https://fal.example/out.png"}]})
    tracker = BudgetTracker(Budget(max_cost_usd=10.0))
    model = FalMediaModel("fal-ai/flux/schnell", client=client, budget=tracker, cost_usd=0.05)

    model.generate(prompt="a cat")
    model.generate(prompt="a dog")

    assert tracker.cost_usd_used == pytest.approx(0.10)


def test_generate_charges_a_cost_computed_from_the_result():
    client = FakeFalClient({"images": [{"url": "https://fal.example/out.png"}], "duration_seconds": 4})
    tracker = BudgetTracker(Budget(max_cost_usd=10.0))
    model = FalMediaModel(
        "fal-ai/kling-video",
        client=client,
        budget=tracker,
        cost_usd=lambda result: result.raw["duration_seconds"] * 0.5,
    )

    model.generate(prompt="a cat flying")

    assert tracker.cost_usd_used == pytest.approx(2.0)


def test_generate_raises_budget_exceeded_once_the_cap_trips():
    client = FakeFalClient({"images": [{"url": "https://fal.example/out.png"}]})
    tracker = BudgetTracker(Budget(max_cost_usd=0.5))
    model = FalMediaModel("fal-ai/flux/schnell", client=client, budget=tracker, cost_usd=1.0)

    with pytest.raises(BudgetExceededError):
        model.generate(prompt="too expensive")


class FakeJobHandle:
    def __init__(self, response):
        self._response = response
        self.cancelled = False

    def status(self):
        return "COMPLETED"

    def get(self):
        return self._response

    def cancel(self):
        self.cancelled = True


class FakeQueueClient:
    def __init__(self, handle):
        self._handle = handle
        self.calls = []

    def submit(self, endpoint, arguments):
        self.calls.append((endpoint, arguments))
        return self._handle


def test_submit_returns_a_job_handle_that_polls_for_the_result():
    handle = FakeJobHandle({"video": {"url": "https://fal.example/out.mp4"}})
    client = FakeQueueClient(handle)
    model = FalMediaModel("fal-ai/kling-video", client=client)

    job = model.submit(prompt="a slow video")

    assert isinstance(job, FalJobHandle)
    assert client.calls == [("fal-ai/kling-video", {"prompt": "a slow video"})]
    assert job.status() == "COMPLETED"
    result = job.result()
    assert result.urls == ["https://fal.example/out.mp4"]


def test_job_handle_cancel_delegates_to_the_raw_handle():
    handle = FakeJobHandle({})
    job = FalJobHandle(handle)

    job.cancel()

    assert handle.cancelled is True


def test_job_handle_result_translates_errors_too():
    class FailingHandle:
        def get(self):
            raise _StatusCodeError("gone", 404)

    job = FalJobHandle(FailingHandle())
    with pytest.raises(ProviderError):
        job.result()
