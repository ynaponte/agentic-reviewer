from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional
from threading import Lock
import os
import uuid


class VectorDatabaseManager:

    _instance = None
    _lock = Lock()  # Proteção para evitar que várias threads tentem criar o mesmo objeto ao mesmo tempo

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorDatabaseManager, cls).__new__(cls)
            return cls._instance

    def __init__(self, embedding_model: str = "nomic-embed-text:latest"):
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        self.vectorstore = None
        self.current_collection = None

    # ------------------------ Public Methods ------------------------

    def initialize_db(self, persist_directory: str = "./chroma_db", collection_name: str = "default") -> 'VectorDatabaseManager':
        """
        Inicializa o banco de dados com configuração flexível
        :param persist_directory: Diretório onde o banco de dados será salvo/existe
        :param collection_name: Nome da coleção do banco de dados a ser acessada
        :return: Instância do VectorDatabaseManager
        """
        os.makedirs(persist_directory, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=persist_directory
        )
        self.current_collection = collection_name
        return self
    
    def store_documents(self, directory: str, uploader: str) -> int:
        """Carrega documentos com chunking e metadados"""
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        doc_loaded_count = 0
        
        for filename in os.listdir(directory):
            if not filename.endswith(".pdf"):
                continue

            try:
                loader = PyMuPDFLoader(os.path.join(directory, filename))
                pages = loader.load()
                
                for page in pages:
                    processed_docs = self._process_document(
                        filename=filename,
                        uploader=uploader,
                        content=page.page_content
                    )
                    
                    self.vectorstore.add_documents(
                        documents=processed_docs,
                        ids=[str(uuid.uuid4()) for _ in processed_docs]
                    )
                    doc_loaded_count += len(processed_docs)
                
            except Exception as e:
                print(f"Erro processando {filename}: {str(e)}")
                continue
        
        return doc_loaded_count

    def query(
        self,
        query: Optional[str] = "",
        uploader: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = 100
    ) -> List[Document]:
        
        """Consulta flexível com filtros"""
        
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
            
        filters = [
            {filter_argument: value} for filter_argument, value in (("uploader", uploader), ("source", source))
            if value is not None
        ]

        chroma_filters = None if filters == [] else (
            {"$and": filters} if len(filters) > 1 else filters[0]
        )

        query_output = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=chroma_filters
        )

        return [self._format_query_output(doc) for doc in query_output]
    
    def switch_current_collection(self, new_collection) -> 'VectorDatabaseManager':
        """Muda a coleção atual para a nova coleção"""

        if new_collection is not self.vectorstore._collection_name:
            self.vectorstore = Chroma(
                collection_name=new_collection,
                embedding_function=self.embedding,
                persist_directory=self.vectorstore._persist_directory,
            )

        return self

    # ------------------------ Private Methods ------------------------

    def _process_document(self, filename: str, uploader: str, content: str) -> List[Document]:
        """Processa um documento em chunks com metadados"""
        base_metadata = {
            "doc_id": str(uuid.uuid4()),
            "source": filename,
            "uploader": uploader,
            "last_modified": datetime.now().isoformat()
        }

        chunks = self.text_splitter.split_text(content)
        return [
            Document(
                page_content=chunk,
                metadata={**base_metadata, "chunk_id": f"{i:04d}"}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    @staticmethod
    def _format_query_output(doc: Document) -> Dict:
        """Formatar a saída da consulta para um dicionário"""
        return {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "uploader": doc.metadata.get("uploader"),
        }
