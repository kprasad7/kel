import pytest

from kel.budget import Budget, BudgetExceededError, BudgetTracker
from kel.media import ReplicateJobHandle, ReplicateMediaModel, get_image_model, get_video_model
from kel.models.errors import AuthenticationError, ProviderError, RateLimitError


class FakeReplicateClient:
    def __init__(self, output):
        self._output = output
        self.calls: list[tuple[str, dict]] = []

    def run(self, model_ref, input):
        self.calls.append((model_ref, input))
        return self._output


class FakeAsyncReplicateClient:
    def __init__(self, output):
        self._output = output
        self.calls: list[tuple[str, dict]] = []

    async def async_run(self, model_ref, input):
        self.calls.append((model_ref, input))
        return self._output


class _StatusCodeError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeErrorClient:
    def __init__(self, exc):
        self._exc = exc

    def run(self, model_ref, input):
        raise self._exc


def test_generate_normalizes_a_single_url_string_output():
    client = FakeReplicateClient("https://replicate.example/out.png")
    model = ReplicateMediaModel("black-forest-labs/flux-schnell", client=client)

    result = model.generate(prompt="a cat")

    assert client.calls == [("black-forest-labs/flux-schnell", {"prompt": "a cat"})]
    assert result.urls == ["https://replicate.example/out.png"]


def test_generate_normalizes_a_list_of_url_outputs():
    client = FakeReplicateClient(["https://replicate.example/a.png", "https://replicate.example/b.png"])
    model = ReplicateMediaModel("black-forest-labs/flux-schnell", client=client)

    result = model.generate(prompt="two cats")

    assert result.urls == ["https://replicate.example/a.png", "https://replicate.example/b.png"]


def test_generate_passes_through_a_structured_dict_output_unchanged():
    client = FakeReplicateClient({"video": {"url": "https://replicate.example/out.mp4"}, "seed": 7})
    model = ReplicateMediaModel("minimax/video-01", client=client)

    result = model.generate(prompt="a cat flying")

    assert result.urls == ["https://replicate.example/out.mp4"]
    assert result.raw["seed"] == 7


async def test_agenerate_calls_injected_async_client():
    client = FakeAsyncReplicateClient("https://replicate.example/out.mp4")
    model = ReplicateMediaModel("minimax/video-01", client=client)

    result = await model.agenerate(prompt="a cat flying")

    assert client.calls == [("minimax/video-01", {"prompt": "a cat flying"})]
    assert result.urls == ["https://replicate.example/out.mp4"]


def test_get_image_model_and_get_video_model_resolve_replicate_spec():
    client = FakeReplicateClient("https://replicate.example/out.png")

    image_model = get_image_model("replicate:black-forest-labs/flux-schnell", client=client)

    assert isinstance(image_model, ReplicateMediaModel)
    assert image_model.model_ref == "black-forest-labs/flux-schnell"

    video_model = get_video_model("replicate:minimax/video-01", client=FakeReplicateClient({}))
    assert video_model.model_ref == "minimax/video-01"


def test_generate_translates_a_401_into_authentication_error():
    model = ReplicateMediaModel("owner/model", client=FakeErrorClient(_StatusCodeError("bad key", 401)))
    with pytest.raises(AuthenticationError):
        model.generate(prompt="hi")


def test_generate_translates_a_429_into_rate_limit_error():
    model = ReplicateMediaModel("owner/model", client=FakeErrorClient(_StatusCodeError("slow down", 429)))
    with pytest.raises(RateLimitError):
        model.generate(prompt="hi")


def test_generate_translates_an_unknown_error_into_provider_error():
    model = ReplicateMediaModel("owner/model", client=FakeErrorClient(RuntimeError("boom")))
    with pytest.raises(ProviderError):
        model.generate(prompt="hi")


def test_generate_charges_a_fixed_cost_against_a_budget_tracker():
    client = FakeReplicateClient("https://replicate.example/out.png")
    tracker = BudgetTracker(Budget(max_cost_usd=10.0))
    model = ReplicateMediaModel("owner/model", client=client, budget=tracker, cost_usd=0.02)

    model.generate(prompt="a cat")
    model.generate(prompt="a dog")

    assert tracker.cost_usd_used == pytest.approx(0.04)


def test_cost_estimator_reserves_budget_before_the_call():
    client = FakeReplicateClient("https://replicate.example/out.mp4")
    tracker = BudgetTracker(Budget(max_cost_usd=1.0))
    model = ReplicateMediaModel(
        "minimax/video-01", client=client, budget=tracker, cost_estimator=lambda arguments: arguments["duration"] * 0.5
    )

    with pytest.raises(BudgetExceededError):
        model.generate(prompt="a very long video", duration=10)

    assert client.calls == []


class FakePrediction:
    def __init__(self, output):
        self.output = output
        self.status = "starting"
        self.cancelled = False
        self._reload_count = 0

    def reload(self):
        self._reload_count += 1
        self.status = "succeeded"

    def wait(self):
        self.status = "succeeded"

    def cancel(self):
        self.cancelled = True


class FakePredictions:
    def __init__(self, prediction):
        self._prediction = prediction
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, input))
        return self._prediction


class FakeQueueClient:
    def __init__(self, prediction):
        self.predictions = FakePredictions(prediction)


def test_submit_returns_a_job_handle_that_polls_for_the_result():
    prediction = FakePrediction("https://replicate.example/out.mp4")
    client = FakeQueueClient(prediction)
    model = ReplicateMediaModel("minimax/video-01", client=client)

    job = model.submit(prompt="a slow video")

    assert isinstance(job, ReplicateJobHandle)
    assert client.predictions.calls == [("minimax/video-01", {"prompt": "a slow video"})]
    assert job.status() == "succeeded"
    result = job.result()
    assert result.urls == ["https://replicate.example/out.mp4"]


def test_job_handle_cancel_delegates_to_the_raw_prediction():
    prediction = FakePrediction("https://replicate.example/out.mp4")
    job = ReplicateJobHandle(prediction)

    job.cancel()

    assert prediction.cancelled is True
