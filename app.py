"""Chainlit UI for YouTube RAG — chat with indexed YouTube creators."""

import asyncio

import chainlit as cl

from channel_agent import ChannelAgent
from persona_builder import load_persona
from state_manager import PipelineState


def _build_channel_list(channels: list[str]) -> str:
    lines = ["**Welcome to YouTube RAG!** Here are your indexed channels:\n"]
    for i, ch in enumerate(channels, 1):
        persona = load_persona(ch)
        if persona:
            name = persona.get("display_name", ch)
            topics = ", ".join(persona.get("topics", [])[:3])
            tone = persona.get("tone", "")
            lines.append(f"**{i}.** {name} — {topics} *(tone: {tone})*")
        else:
            lines.append(f"**{i}.** {ch}")
    lines.append("\nType a **number** to start chatting.")
    return "\n".join(lines)


@cl.on_chat_start
async def start():
    channels = PipelineState().get_tracked_channels()
    cl.user_session.set("channels", channels)

    if not channels:
        await cl.Message(
            content=(
                "No channels indexed yet. Run this command to index one:\n"
                "```\npython main.py index <youtube_url>\n```"
            )
        ).send()
        cl.user_session.set("state", "no_channels")
        return

    await cl.Message(content=_build_channel_list(channels)).send()
    cl.user_session.set("state", "selecting")


@cl.on_message
async def handle_message(message: cl.Message):
    state = cl.user_session.get("state")

    if state == "no_channels":
        await cl.Message(
            content="Please index a channel first:\n```\npython main.py index <youtube_url>\n```"
        ).send()
        return

    if state == "selecting":
        await _handle_selection(message)
        return

    if state == "chatting":
        await _handle_chat(message)
        return


async def _handle_selection(message: cl.Message):
    channels = cl.user_session.get("channels", [])
    text = message.content.strip()

    channel = None
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(channels):
            channel = channels[idx]
    else:
        text_lower = text.lower().lstrip("@")
        for ch in channels:
            if text_lower in ch.lower():
                channel = ch
                break

    if channel is None:
        await cl.Message(
            content="Couldn't find that channel. Please type a number from the list."
        ).send()
        return

    agent = ChannelAgent(channel)
    cl.user_session.set("agent", agent)
    cl.user_session.set("state", "chatting")

    persona = agent.persona
    if persona:
        name = persona.get("display_name", channel)
        summary = persona.get("persona_summary", "")
        topics = ", ".join(persona.get("topics", [])[:5])
        confirm = (
            f"**Now chatting with {name}**\n\n"
            f"{summary}\n\n"
            f"*Topics: {topics}*\n\n"
            "Ask me anything!"
        )
    else:
        confirm = f"**Now chatting with {channel}**\n\nAsk me anything!"

    await cl.Message(content=confirm).send()


async def _handle_chat(message: cl.Message):
    agent: ChannelAgent = cl.user_session.get("agent")
    tool_calls_log = []

    def on_tool_call(query: str, result: str):
        doc_count = result.count("[From:")
        tool_calls_log.append({"query": query, "docs_found": doc_count})

    agent.on_tool_call = on_tool_call

    response = await asyncio.to_thread(agent.chat, message.content)

    if tool_calls_log:
        queries = "\n".join(f"• {tc['query']}" for tc in tool_calls_log)
        total_docs = sum(tc["docs_found"] for tc in tool_calls_log)
        label = "query" if len(tool_calls_log) == 1 else "queries"
        step = cl.Step(name=f"Searched videos ({len(tool_calls_log)} {label})", type="tool")
        step.input = queries
        step.output = f"Retrieved {total_docs} relevant chunk{'s' if total_docs != 1 else ''}"
        await step.send()

    await cl.Message(content=response).send()
