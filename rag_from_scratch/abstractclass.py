from abc import ABC, abstractmethod


class ChatBot(ABC):
    @abstractmethod
    def __call__(self, message: str):
        pass

    @abstractmethod
    def get_response(self):
        pass

    @abstractmethod
    def inspect_chat_history(self, n: int = 1):
        pass


class Retriever(ABC):
    @abstractmethod
    def query(self, query: str):
        pass

    @abstractmethod
    def build_database(self):
        pass

    @abstractmethod
    def load_database(self):
        pass


class ChatOpenAI(ABC):
    @abstractmethod
    def load_retriever(self):
        pass

    @abstractmethod
    def load_chatbot(self):
        pass

    @abstractmethod
    def chat(self):
        pass
