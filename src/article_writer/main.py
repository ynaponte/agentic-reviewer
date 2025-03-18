from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.results_and_discussion_crew import ReviewCrew
from ..tools import FetchMetadataTool
from typing import List, Optional
from ..utils import VectorDatabaseManager
from .types.doc_report import ContentReport

import asyncio
import json


class ArticleWriterState(BaseModel):
    theme: str = (
        "Utilização de algorítimos genéticos para criação de portas lógicas ópticas "
        "em cristal fotônico"
    )
    drafts_documents: List[str] = ['Resultado1.pdf']
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    chunk_reports: List[ContentReport] = []


class ArticleWriterFlow(Flow[ArticleWriterState]):
    
    articles_db = VectorDatabaseManager()

    @start()
    def start_flow(self):
        # Inicializa a base de dados para uso das ferramentas
        self.articles_db.initialize_db(
            persist_directory="article_vectorstore",
            collection_name="flow_test_collection"
        )

    @listen(start_flow)
    async def pre_processing(self):
        def batching(num_chunks, batch_size):
            '''
            Defina o tamanho do lote de acordo?
                - Llama3.1:8b -> 
            :param: batch_size - tamanho do lote de chunks
            '''
            return [
                list(range(i, min(i + batch_size, num_chunks))) 
                for i in range(0, num_chunks, batch_size)
            ]
        
        def process_single_document(doc_name: str, chunk_indexes: List[int]):
            output = ReviewCrew().crew().kickoff(
                    inputs={
                        "target_document": doc_name,
                        "chunk_indexes": chunk_indexes,
                        "theme": self.state.theme,                        
                    }
                )
            return output

        tasks = []
        for draft_document in self.state.drafts_documents:
            doc_meta = self.articles_db.search_doc_by_meta(source=draft_document, type='draft')
            chunk_batches = batching(doc_meta[0]['total_chunks'], 2)
            for batch in chunk_batches:
                task = process_single_document(draft_document, batch)
                tasks.append(task)

        #reports = await asyncio.gather(*tasks)
        #self.state.chunk_reports = reports
        print(tasks.raw)

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()

def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()

if __name__ == "__main__":
    kickoff()
