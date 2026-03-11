"""Chainlit UI for YouTube RAG — chat with indexed YouTube creators."""

import asyncio

import chainlit as cl

from channel_agent import ChannelAgent
from persona_builder import load_persona
from state_manager import PipelineState


async def _send_channel_gallery(channels: list[str]) -> None:
    await cl.Message(content="## Welcome to YouTube RAG\nSelect a channel to start chatting:").send()
    for ch in channels:
        persona = load_persona(ch)
        if persona:
            name = persona.get("display_name", ch)
            topics = ", ".join(persona.get("topics", [])[:3])
            tone = persona.get("tone", "")
            card_text = f"### {name}\n**Topics:** {topics}  \n**Tone:** {tone}"
        else:
            name = ch
            card_text = f"### {ch}"

        actions = [
            cl.Action(
                name="select_channel",
                payload={"channel": ch},
                label="Chat →",
            )
        ]
        await cl.Message(content=card_text, actions=actions).send()


@cl.on_chat_start
async def start():
    channels = PipelineState().get_tracked_channels()
    cl.user_session.set("channels", channels)
    cl.user_session.set("state", "selecting")

    if not channels:
        await cl.Message(
            content=(
                "No channels indexed yet. Run this command to index one:\n"
                "```\npython main.py index <youtube_url>\n```"
            )
        ).send()
        cl.user_session.set("state", "no_channels")
        return

    await _send_channel_gallery(channels)


@cl.action_callback("select_channel")
async def on_channel_selected(action: cl.Action):
    channel = action.payload["channel"]
    agent = ChannelAgent(channel)
    cl.user_session.set("agent", agent)
    cl.user_session.set("state", "chatting")

    persona = agent.persona
    if persona:
        name = persona.get("display_name", channel)
        summary = persona.get("persona_summary", "")
        topics = persona.get("topics", [])
    else:
        name = channel
        summary = ""
        topics = []

    starters = [
        cl.Action(
            name="starter_q",
            payload={"question": f"Tell me about {t}"},
            label=f"Tell me about {t}",
        )
        for t in topics[:3]
    ]

    confirm = f"**Now chatting with {name}**\n\n{summary}"
    if starters:
        confirm += "\n\n*Click a question below or type your own:*"

    await cl.Message(content=confirm, actions=starters).send()
    await action.remove()


@cl.action_callback("starter_q")
async def on_starter_question(action: cl.Action):
    await action.remove()
    fake_msg = cl.Message(content=action.payload["question"], author="User")
    await _handle_chat(fake_msg)


@cl.on_message
async def handle_message(message: cl.Message):
    state = cl.user_session.get("state")

    if state in ("no_channels", "selecting"):
        await cl.Message(
            content="Please select a channel by clicking **Chat →** on one of the cards above."
        ).send()
        return

    if state == "chatting":
        await _handle_chat(message)


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
