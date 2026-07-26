"""Image and video generation via kel.media — a generic gateway over
third-party generative-media APIs (fal.ai here), following the same
"provider:model" spec pattern as kel.get_model. Also shows the two things
that matter most once real money is on the line: budget governance
(cost_estimator= reserves cost *before* the network call) and vendor
error handling (RateLimitError, same hierarchy chat models use).

Run:
    pip install "pykel[fal]"
    export FAL_KEY=...          # https://fal.ai/dashboard/keys
    python examples/03_media_generation.py
"""

from kel.budget import Budget, BudgetExceededError, BudgetTracker
from kel.media import get_image_model, get_video_model
from kel.models import RateLimitError

# ---------------------------------------------------------------------------
# 1. Image generation. "Scale" (resolution, aspect ratio, how many
#    variants) is just an argument the specific fal endpoint defines —
#    kel doesn't hardcode a fixed schema per model.
# ---------------------------------------------------------------------------
image_model = get_image_model("fal:fal-ai/flux/schnell")


def generate_image() -> None:
    result = image_model.generate(
        prompt="a lighthouse at sunset, photorealistic",
        image_size="landscape_16_9",
        num_images=1,
    )
    print("image URL(s):", result.urls)
    print("full response:", result.raw)


# ---------------------------------------------------------------------------
# 2. Video generation, budget-guarded. Video can take a while and cost
#    real money per call — cost_estimator reserves an estimated cost
#    against the tracker *before* the network call runs, so a call that
#    would exceed the budget never actually happens (as opposed to
#    cost_usd=, which would only tell you about the overspend after the
#    call already completed).
# ---------------------------------------------------------------------------
# One tracker shared across both video calls below (5s * $0.35/s = $1.75
# each) — cap set high enough for this demo's two calls to both succeed;
# lower it to see generate_video_within_budget() actually refuse to spend.
tracker = BudgetTracker(Budget(max_cost_usd=5.0))
video_model = get_video_model(
    "fal:fal-ai/kling-video/v1.6/standard/text-to-video",
    budget=tracker,
    cost_estimator=lambda arguments: arguments.get("duration", 5) * 0.35,  # rough $/second estimate
)


def generate_video_within_budget() -> None:
    try:
        result = video_model.generate(prompt="a drone shot flying over mountains at dawn", duration=5)
        print("video URL:", result.urls[0] if result.urls else None)
    except BudgetExceededError as exc:
        print(f"skipped: {exc}")
    except RateLimitError:
        print("fal rate-limited this request — safe to retry after a short backoff")


# ---------------------------------------------------------------------------
# 3. The same video call, but via submit()/poll instead of blocking —
#    fal's own docs recommend this for anything slow, so a request thread
#    isn't tied up waiting minutes for a render to finish.
# ---------------------------------------------------------------------------
def generate_video_without_blocking() -> None:
    job = video_model.submit(prompt="a slow cinematic pan across a mountain range", duration=5)
    print("submitted, status:", job.status())
    # ... do other work here; come back and poll/collect whenever's convenient ...
    result = job.result()  # blocks only here, once you actually need the output
    print("video URL:", result.urls[0] if result.urls else None)


if __name__ == "__main__":
    generate_image()
    generate_video_within_budget()
    generate_video_without_blocking()
