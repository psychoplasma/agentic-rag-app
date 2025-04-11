from typing_extensions import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


class VectorStore:
    def __init__(self,
                 embeddings: Embeddings,
                 persist_directory: str = "./chroma_langchain_db",
                 collection_name: str = "example_collection"):
        self.vector_store = Chroma(
            collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        )

    def add(self, documents: List[Document]) -> int:
        doc_ids = self.vector_store.add_documents(documents=documents)
        return len(doc_ids)

    def search(self, query: str, k: int) -> List[Document]:
        return self.vector_store.similarity_search(query=query, k=k)
