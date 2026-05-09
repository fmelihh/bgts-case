import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
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

mcp_client = MultiServerMCPClient(
    {
        "tickets": {
            "url": secrets.mcp_server_url,
            "transport": "streamable_http",
        },
    }
)


async def _build_graph():
    tools = await mcp_client.get_tools()
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


graph = asyncio.run(_build_graph())

__all__ = ["graph"]
