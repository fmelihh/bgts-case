from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from bgts_case.secret import secrets

MODEL = "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"
SYSTEM_PROMPT = "Sen yardımcı bir asistansın."

model = ChatOpenAI(
    model=MODEL,
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=secrets.fireworks_api_key,
    temperature=0,
)

graph = create_agent(
    model=model,
    tools=[],
    system_prompt=SYSTEM_PROMPT,
)


def run_agent() -> None:
    """Entry point: invoke the LangGraph agent once with a sample prompt."""
    result = graph.invoke(
        {"messages": [HumanMessage(content="Python'da fibonacci fonksiyonu yaz.")]}
    )
    print(result["messages"][-1].content)
