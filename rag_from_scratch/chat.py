from rag_from_scratch.abstractclass import ChatOpenAI
from rag_from_scratch.retrieve import OpenAIRetriever
from rag_from_scratch.lm import OpenAIChatBot
from rag_from_scratch.data import query_pubmed
from typing import List


class OpenAIChatBotMain(ChatOpenAI):
    def __init__(
        self,
        *,
        query_list: List[str],
        database_path: str = "./chromadb",
        collection_name: str = "chroma_collection",
        load_only: bool = True,
        MAX_RESULTS: int = 10
    ):
        """OpenAI Chatbot constructor for creating a chatbot to answer questions regarding the PubMed dataset

        Args:
            database_path (str, optional): database path. Defaults to "./chromadb".
            collection_name (str, optional): collection name. Defaults to "chroma_collection".
            load_only (bool): whether to load or build the database. Defaults to True.
        """
        self.database_path = database_path
        self.collection_name = collection_name
        if not load_only:
            pubmed_docs = []
            pubmed_metadata = []
            for query in query_list:
                docs, metadata = query_pubmed(query,MAX_RESULTS)
                pubmed_docs.extend(docs)
                pubmed_metadata.extend(metadata)
            self.retriever = OpenAIRetriever(
                pubmed_docs,
                pubmed_metadata,
                database_path=self.database_path,
                collection_name=self.collection_name,
            )
            self.retriever.build_database()
            self.collection = self.retriever.load_database()
        else:
            self.retriever = OpenAIRetriever(
                docs=[],
                metadata=[],
                database_path=self.database_path,
                collection_name=self.collection_name,
            )
            self.collection = self.retriever.load_database()

        self.openai_chat_model = OpenAIChatBot(
            system="You are a helpful assistant that can answer questions from the provided context about PubMed abstracts.\n Be faithful to the context and answer from the context only and if it is not mentioned in the context, then say that you don't know."
        )

    def chat(self,question):
        res = self.retriever.query(
                question=question,collection=self.collection,num_results=2
            )
        context = [doc + "\n" for doc in res['documents'][0]]
        context_str = "\n".join(context)

        format_prompt = """
        ---------------------
        Context: {context_str}
        ---------------------

        Question: {question}

        Answer:
        """

        prompt = format_prompt.format(context_str=context_str,question=question)
        answer,prompt_stats = self.openai_chat_model(prompt)
        return answer,prompt_stats
    