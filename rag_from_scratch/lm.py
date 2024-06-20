import openai
import os
from rag_from_scratch.abstractclass import ChatBot
from dotenv import load_dotenv, find_dotenv
from typing import Tuple

load_dotenv(find_dotenv(), override=True)


class OpenAIChatBot(ChatBot):
    def __init__(self, system: str = ""):
        self.system_msg = system
        self.messages = []
        if self.system_msg != "":
            self.messages.append({"role": "system", "content": self.system_msg})
        self.prompt_stats = []

    def __call__(self, message: str):
        self.messages.append({"role": "user", "content": message})
        result, prompt_stats = self.get_response()
        self.messages.append({"role": "assistant", "content": result})
        self.prompt_stats.append(prompt_stats)
        return result, prompt_stats

    def get_response(self) -> Tuple[str, str]:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=self.messages
        )
        return completion.choices[0].message.content, completion.usage

    def inspect_chat_history(self, n: int = 1):
        """Get the chat history in the reverse order of chatting

        Args:
            n (int, optional): how many chat conversations to return. Defaults to 1.
        """
        if self.system_msg != "":
            curr_len_messages = len(self.messages) - 1
        else:
            curr_len_messages = len(self.messages)

        assert (
            curr_len_messages <= 2 * n
        ), f"You are trying to access more number of conversations than you have, you can access {curr_len_messages//2} previous chats"

        chat_histories = self.messages[-2 * n :]
        chat_history_str = self.system_msg + "\n"
        for chat_history in chat_histories:
            chat_history_str += chat_history["content"]
        return chat_history_str
