from helpers import ScriptedModel
from kel.models.types import ModelResponse, ToolUsePart, Usage
from kel.retrieval.reranker import LLMReranker
from kel.retrieval.types import Chunk, ScoredChunk


def _structured_response(scores: list[float], rid: str = "r") -> ModelResponse:
    tool_call = ToolUsePart(id="1", name="return_structured_output", input={"scores": scores})
    return ModelResponse(id=rid, model="fake-1", content=[tool_call], stop_reason="tool_use", usage=Usage())


def _candidate(chunk_id: str, text: str, first_stage_score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=chunk_id, text=text), score=first_stage_score)


def test_rerank_reorders_candidates_by_llm_relevance_scores():
    candidates = [
        _candidate("a", "irrelevant passage", first_stage_score=0.9),
        _candidate("b", "the actual answer to the query", first_stage_score=0.5),
    ]
    # model says candidate "b" (second in the list) is actually more relevant,
    # inverting the first-stage vector-similarity ranking
    model = ScriptedModel("fake-1", [_structured_response([0.1, 0.95])])
    reranker = LLMReranker(model)

    results = reranker.rerank("what is the answer?", candidates, k=2)

    assert [r.chunk.id for r in results] == ["b", "a"]
    assert results[0].score == 0.95


def test_rerank_truncates_to_k():
    candidates = [_candidate(str(i), f"text {i}", 0.5) for i in range(5)]
    model = ScriptedModel("fake-1", [_structured_response([0.1, 0.2, 0.3, 0.4, 0.5])])
    reranker = LLMReranker(model)

    results = reranker.rerank("query", candidates, k=2)

    assert len(results) == 2
    # highest scores (0.5, 0.4) correspond to candidates "4" and "3"
    assert [r.chunk.id for r in results] == ["4", "3"]


def test_rerank_falls_back_to_first_stage_order_on_score_count_mismatch():
    candidates = [_candidate("a", "x", 0.5), _candidate("b", "y", 0.4)]
    # model returns only 1 score for 2 candidates — don't risk mismapping
    model = ScriptedModel("fake-1", [_structured_response([0.9])])
    reranker = LLMReranker(model)

    results = reranker.rerank("query", candidates, k=2)

    assert [r.chunk.id for r in results] == ["a", "b"]


def test_rerank_on_empty_candidates_returns_empty_without_calling_the_model():
    model = ScriptedModel("fake-1", [])
    reranker = LLMReranker(model)

    assert reranker.rerank("query", [], k=5) == []


async def test_arerank_reorders_candidates_by_llm_relevance_scores():
    candidates = [
        _candidate("a", "irrelevant passage", first_stage_score=0.9),
        _candidate("b", "the actual answer to the query", first_stage_score=0.5),
    ]
    model = ScriptedModel("fake-1", [_structured_response([0.1, 0.95])])
    reranker = LLMReranker(model)

    results = await reranker.arerank("what is the answer?", candidates, k=2)

    assert [r.chunk.id for r in results] == ["b", "a"]
