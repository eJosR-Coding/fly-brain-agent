"""Shared fake chat model: reply is a function of the conversation so far. No network."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FnModel(GenericFakeChatModel):
    reply: object = None
    seen: list = []
    disable_streaming: bool = True  # plain one-shot replies, like a provider that does not stream

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.reply(messages))])


def fn_model(reply) -> FnModel:
    return FnModel(messages=iter([]), reply=reply, seen=[])


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])
