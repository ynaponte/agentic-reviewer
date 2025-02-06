from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any
from threading import Lock
import os
import hashlib
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
            chunk_size=2000,
            chunk_overlap=20,
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
    
    def store_documents(self, directory: str, uploader: str) -> 'VectorDatabaseManager':
        """Carrega documentos com chunking e metadados"""
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
        
        for filename in os.listdir(directory):
            if not filename.endswith(".pdf"):
                continue

            try:
                source = filename[:-4].lower()  # Retira a extensão .pdf do nome do artigo e deixa em caixa baixa
                doc_id = self._gen_doc_id(source)

                # Checa se o documento já existe no banco de dados
                if self.search_doc_by_meta(doc_id=doc_id):
                    continue

                loader = PyMuPDFLoader(os.path.join(directory, filename))
                pages = loader.load_and_split(text_splitter=self.text_splitter)  # Retorna uma lista de chunks, contendo metadata do documento
                total_of_chunks = len(pages)
                metadata_to_delete = [
                    'author',
                    'creationDate',
                    'creator',
                    'file_path'
                    'format',
                    'keywords',
                    'modDate',
                    'producer',
                    'subject',
                    'title',
                    'trapped'
                ]

                for chunk_idx, page_chunk in enumerate(pages):
                    page_chunk.metadata.update({
                        "doc_id":doc_id,
                        "source": filename,
                        "uploader": uploader,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_of_chunks,
                        "last_modified": datetime.now().isoformat()
                    })
                    for metadata in metadata_to_delete:
                        page_chunk.metadata.pop(metadata, None)

                    page_chunk.id = str(uuid.uuid4())
                
                self.vectorstore.add_documents(
                    documents=pages
                )
                
            except Exception as e:
                print(f"Erro processando {filename}: {str(e)}")
                continue
        
        return self
    
    def query(
        self,
        query: Optional[str] = "",
        uploader: Optional[List[str]] = None,
        source: Optional[List[str]] = None,
        top_k: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        
        """
        Consulta flexível em todos os artigos da base de dados,
        possibilitando utilizar todos os artigos para obteção de informações específicas.
        Também permite a filtragem por uploader e por artigos específicos. Caso especificado
        uploader e/ou source, limita os artigos da consulta a apenas aqueles com tais dados
        em seus metadados.
        """
        
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
        
        # Filtros para pesquisa avançada    
        filters = [
            {filter_argument: {"$in": value}} for filter_argument, value in (
                ("uploader", uploader),
                ("source", source)
            )if value is not None
        ]

        chroma_filters = None if filters == [] else (
            {"$and": filters} if len(filters) > 1 else filters[0]
        )

        query_output = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=chroma_filters
        )

        return [self._format_output(doc) for doc in query_output]
    
    def search_doc_by_meta(
        self, 
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        uploader: Optional[str] = None,
        metadata_only: Optional[bool] = True
    ) -> List[Dict[str, Any]]:
        """
        Método para obter as chunks de um artigo específico, identificado pelo nome(source).
        Se o parâmetro page for None, retorna todos os chunks (documento completo).
        Se page for um número, retorna somente os chunks correspondentes àquela página.
        """

        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
        
        filters = [
            {filter_argument: spec} for filter_argument, spec in (
                ("doc_id", {"$eq": doc_id}),
                ("source", {"$eq": source}),
                ("uploader", {"$eq": uploader})
            )if spec['$eq'] is not None
        ]
        
        where_filters = {"$and": filters} if len(filters) > 1 else filters[0]
        search_result = self.vectorstore.get(
            where=where_filters,
            include=["metadatas"] if metadata_only else ["metadatas", "documents"]
        )

        if search_result['ids'] == []:
            return f"Artigo não encontrado. Dados da busca:\ndoc_id: {doc_id}\nsource: {source}\nuploader: {uploader}."
        
        return (
            self._format_meta_search_output(search_result)
            if metadata_only else self._format_full_doc_search_output(search_result)
        )

    # ------------------------ Private Methods ------------------------

    @staticmethod
    def _gen_doc_id(filename: str) -> str:
        """Gera um id para o documento, a partir do seu nome"""
        return hashlib.sha256(filename.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _format_output(doc: Document) -> Dict:
        """Formatar a saída da consulta para um dicionário"""
        return {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "uploader": doc.metadata.get("uploader"),
            "page": doc.metadata.get("page"),
            "total_pages": doc.metadata.get("total_pages"),
            "doc_id": doc.metadata.get("doc_id"),
            "chunck_idx": doc.metadata.get("chunck_idx"),
            "total_chunks": doc.metadata.get("total_chunks"),
        }

    @staticmethod
    def _format_meta_search_output(only_metadatas) -> List[Dict[str, Any]]:
        """Formatar a saída da consulta para uma lista de metadados únicos"""
        metadatas = only_metadatas['metadatas']
        keys_to_search = ("doc_id", "source", "total_chunks", "total_pages", "uploader")
        unique_ocorrences = {
            tuple(metadata[key] for key in keys_to_search)
            for metadata in metadatas
        }

        unique_metadata = []
        for metadata_value in unique_ocorrences:
            unique_metadata.append(
                {
                    key: value for key, value in zip(keys_to_search, metadata_value)
                }
            )
        
        return unique_metadata
    
    @staticmethod
    def _format_full_doc_search_output(metadata_and_chunks) -> Dict[str, List[Dict[str, Any]]]:
        """
        Reformata o dicionário resulatante da busca para a estrutura:
            articles = {source: [{content:"...", chunk_id:...},{...},...{...}]}
        """
        chunks = metadata_and_chunks['documents']
        metadatas = metadata_and_chunks['metadatas']
        # Se aproveita da natureza sequencial que os chunks aparecem(grandes blocos do mesmo artigo)
        # para organizar os chunks em um dicionário.
        # O dicionário tem como chave a source do documento e como valor uma lista de chunks e sua metadata
        articles = {}
        for idx in range(len(chunks)):
            content_and_metadata = {
                "content": chunks[idx],
                **metadatas[idx]
            }

            articles.setdefault(metadatas[idx]['source'], []).append(content_and_metadata)

        return articles
    