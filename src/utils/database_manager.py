from datetime import datetime
from multiprocessing import Pool, cpu_count
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any
from src.utils.section_classifier import SectionClassifier
from threading import Lock
import os
import hashlib
import uuid
import pymupdf4llm
  

class VectorDatabaseManager:

    _instance = None
    _lock = Lock()  # Proteção para evitar que várias threads tentem criar o mesmo objeto ao mesmo tempo

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorDatabaseManager, cls).__new__(cls)
            return cls._instance

    def __init__(self, embedding_model: str = "nomic-embed-text:latest"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len,
        )
        self.vectorstore = None
        self.current_collection = None
        self._initialized = True

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
    
    def store_documents(self, directory_list: List[str], type: str) -> 'VectorDatabaseManager':
        """Carrega documentos com chunking e metadados"""
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
        
        for directory in directory_list:
            sources = [
                file for file in os.listdir(directory)
                if file.endswith(".pdf") and not self._check_doc_existence(file)
            ]

            if sources == []:
                return self

            documents_to_process = [
                os.path.join(directory, source) for source in sources
            ]
                        
            with Pool(processes=cpu_count()) as pool:
                loaded_articles = pool.map(self._aload_documents, documents_to_process)
            
            section_finder = SectionClassifier()
            docs_to_upload = []
            for i in range(len(sources)):
                # Endereça os arquivos a serem acessados c -> current
                c_source = sources[i]
                c_pages = loaded_articles[i]
                
                doc_id = hashlib.sha256(c_source.encode('utf-8')).hexdigest()
                # Processamento de metadados
                keys_to_del = [
                    'format', 'title', 'author', 'subject', 'keywords', 'creator',
                    'producer','trapped', 'encryption', 'toc_items', 'words'
                ]

                pages_as_docs = []
                for page in c_pages:
                    new_metadata = {
                        key: value for key, value in page['metadata'].items() if key not in keys_to_del 
                    }

                    new_metadata['doc_id'] = doc_id
                    #new_metadata['tables'] = page.get('tables', [])
                    #new_metadata['images'] = page.get('images', [])
                    new_metadata['source'] = c_source[:-4]
                    new_metadata['type'] = type
                    pages_as_docs.append(Document(page_content=page.get('text',''), metadata=new_metadata))

                chunks = self.text_splitter.split_documents(pages_as_docs)
                for chunk in chunks:  # Classifica a qual seção o texto pertence e atribui um ID 
                    chunk = section_finder.classify_document(chunk)
                    chunk.id = str(uuid.uuid4())

                section_finder.reset
                docs_to_upload.extend(chunks)

            self.vectorstore.add_documents(documents=docs_to_upload)

        return self
    
    def query(
        self,
        query: Optional[str] = "",
        type: Optional[List[str]] = None,
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
                ("uploader", type),
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

    def _check_doc_existence(
        self,
        source: str = None,
    ) -> bool:
        """
        Verifica se um documento já existe na base de dados com base nos metadados.
        Retorna True se encontrar correspondência, False caso contrário.
        """
        result = self.vectorstore.get(
            where={"source": {"$eq": source}},
            include=["metadatas"],
            limit=1
        )

        return len(result["ids"]) > 0
    
    @staticmethod
    def _aload_documents(
        file_path: str,
    ) -> List[Dict[str, Any]]:

        try:
            docs = pymupdf4llm.to_markdown(
                file_path,
                page_chunks=True,
                dpi=200,
                show_progress=True
            )
            return docs
        except Exception as e:
            print(f"Error processing file: {e}")
            raise e
    
    @staticmethod
    def _format_output(doc: Document) -> Dict:
        """Formatar a saída da consulta para um dicionário"""
        return {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "uploader": doc.metadata.get("uploader"),
            "page": doc.metadata.get("page"),
            "total_pages": doc.metadata.get("total_pages"),
            "doc_id": doc.metadata.get("doc_id"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "total_chunks": doc.metadata.get("total_chunks"),
            "sections": doc.metadata.get("sections"),
        }

    @staticmethod
    def _format_meta_search_output(only_metadatas) -> List[Dict[str, Any]]:
        """Formatar a saída da consulta para uma lista de metadados únicos"""
        metadatas = only_metadatas['metadatas']
        meta_to_search = ("doc_id", "source", "total_chunks", "total_pages", "uploader")
        unique_ocorrences = {
            tuple(metadata[meta] for meta in meta_to_search)
            for metadata in metadatas
        }

        unique_metadata = []
        for metadata_value in unique_ocorrences:
            unique_metadata.append(
                {
                    metadata: value for metadata, value in zip(meta_to_search, metadata_value)
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
