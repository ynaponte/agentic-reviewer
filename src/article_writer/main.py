from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.results_and_discussion_crew import ReviewCrew
from ..tools import FetchMetadataTool
from typing import List, Optional
from ..utils import VectorDatabaseManager
from article_writer.types.types import ChunkReport
import asyncio

import json

class ArticleWriterState(BaseModel):
    theme: str = ""
    drafts_documents: List[str] = []
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    reports = List[ChunkReport] = []

class ArticleWriterFlow(Flow[ArticleWriterState]):
    
    articles_db = VectorDatabaseManager()

    @start()
    def start_flow(self):
        # Declara o tema do artigo
        self.state.theme = (
            "Utilização de algorítimos genéticos para criação de portas lógicas ópticas"
            "em cristal fotônico"
        )
        self.articles_db.initialize_db(
            persist_directory="article_vectorstore",
            collection_name="flow_test_collection"
        )
        self.state.drafts_documents = ['Resultado1.pdf', 'Resultado2.pdf']

    @listen(start_flow)
    async def pre_processing(self):
        def batching(num_chuns):
            #TODO: implementar o sistema de criação de batches de chunks
            pass
        
        async def process_single_document(doc_name: str, chunk_indexes: List):
            output = await (
                ReviewCrew()
                .crew()
                .kickoff_async(
                    inputs={
                        "target_document": doc_name,
                        "chunk_indexes": chunk_indexes,
                        "theme": self.state.theme,
                        
                    }
                )
            )

        for draft_document in self.state.drafts_documents:
            tasks = []

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()

def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()

if __name__ == "__main__":
    kickoff()
