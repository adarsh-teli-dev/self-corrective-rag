
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class TestResult(BaseModel):
    decision: Literal["correct", "ambiguous", "incorrect"]
    reasoning: str = Field(description="Brief explanation")


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

structured_llm = llm.with_structured_output(
    TestResult,
    method="json_schema",
)

result = structured_llm.invoke(
    "The evidence directly answers the question. "
    "Return decision correct and explain why."
)

print("Structured output succeeded!")
print(result)
print("Decision:", result.decision)
print("Reasoning:", result.reasoning)
