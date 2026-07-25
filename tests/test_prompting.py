from kel.agents.tool import Tool
from kel.prompting import (
    FewShotExample,
    build_react_system_prompt,
    chain_of_thought_prompt,
    extract_final_answer,
    format_few_shot_prompt,
)


def test_format_few_shot_prompt_includes_instructions_examples_and_query():
    examples = [FewShotExample(input="2+2", output="4"), FewShotExample(input="3+3", output="6")]
    prompt = format_few_shot_prompt(examples, query="5+5", instructions="Answer arithmetic questions.")

    assert prompt.startswith("Answer arithmetic questions.")
    assert "Input: 2+2\nOutput: 4" in prompt
    assert "Input: 3+3\nOutput: 6" in prompt
    assert prompt.endswith("Input: 5+5\nOutput:")


def test_format_few_shot_prompt_without_instructions():
    examples = [FewShotExample(input="hi", output="hello")]
    prompt = format_few_shot_prompt(examples, query="hey")
    assert not prompt.startswith("\n")
    assert "Input: hi\nOutput: hello" in prompt


def test_format_few_shot_prompt_custom_template():
    examples = [FewShotExample(input="x", output="y")]
    prompt = format_few_shot_prompt(examples, query="z", example_template="Q: {input} A: {output}")
    assert "Q: x A: y" in prompt


def test_chain_of_thought_prompt_appends_default_suffix():
    prompt = chain_of_thought_prompt("What is 7*8?")
    assert prompt.startswith("What is 7*8?")
    assert "Final Answer:" in prompt


def test_extract_final_answer_pulls_text_after_marker():
    text = "Let's think: 7*8=56.\n\nFinal Answer: 56"
    assert extract_final_answer(text) == "56"


def test_extract_final_answer_falls_back_to_whole_text_when_marker_missing():
    text = "the answer is 56"
    assert extract_final_answer(text) == "the answer is 56"


def test_extract_final_answer_uses_last_occurrence_if_marker_repeated():
    text = "Final Answer: draft\nMore thinking...\nFinal Answer: 56"
    assert extract_final_answer(text) == "56"


def test_build_react_system_prompt_lists_tool_descriptions():
    tools = [
        Tool(name="search", description="search the web", input_schema={}, fn=lambda i: ""),
        Tool(name="calculate", description="do math", input_schema={}, fn=lambda i: ""),
    ]
    prompt = build_react_system_prompt(tools)

    assert "- search: search the web" in prompt
    assert "- calculate: do math" in prompt
    assert "Thought:" in prompt
    assert "Final Answer:" in prompt
