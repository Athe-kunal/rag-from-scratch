import os
from rag_from_scratch.abstractclass import Retriever
import chromadb
from typing import List
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
import logging

load_dotenv(find_dotenv(), override=True)


class OpenAIRetriever(Retriever):
    def __init__(
        self,
        docs: List[str],
        metadata: List[dict],
        *,
        database_path: str = "./chromadb",
        collection_name: str = "chroma_collection",
    ):
        """_summary_

        Args:
            docs (List[str]): documents to embed
            metadata (List[dict]): list of metadata for each document
            database_path (str, optional): database path name to store the embeddings. Defaults to "./chromadb".
            collection_name (str, optional): collection name. Defaults to "chroma_collection".

        Raises:
            AssertionError: if the length of the documents and metadata don't match.
        """
        chroma_client = chromadb.PersistentClient(database_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=f"{collection_name}_retriever.log",
            encoding="utf-8",
            level=logging.DEBUG,
        )
        self.collection = chroma_client.get_or_create_collection(
            collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-3-small",
            ),
        )
        assert len(docs) == len(
            metadata
        ), f"There should be the same number of documents and metadata but docs length is {len(self.docs)} and metadata length is {len(self.metadata)}."
        self.docs = docs
        self.metadata = metadata
        self.database_path = database_path
        self.collection_name = collection_name

    def build_database(self):
        ids = [str(i) for i in range(len(self.docs))]
        self.collection.add(
            ids=ids,
            documents=self.docs,
            metadatas=self.metadata,
        )
        self.logger.info(
            f"Database was created with database path {self.database_path} with collection name {self.collection_name}"
        )

    def load_database(self) -> chromadb.Collection:
        chroma_client = chromadb.PersistentClient(self.database_path)
        collection = chroma_client.get_collection(
            self.collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-3-small",
            ),
        )
        self.logger.info(
            f"Database was loaded with database path {self.database_path} with collection name {self.collection_name}"
        )
        return collection

    def query(
        self, *, question: str, collection: chromadb.Collection, num_results: int = 2
    ) -> dict:
        relevant_documents = collection.query(
            query_texts=question, n_results=num_results
        )
        return relevant_documents
