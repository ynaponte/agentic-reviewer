from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.results_and_discussion_crew import ResultAndDiscussionCrew
from ..tools import FetchMetadataTool
from typing import List, Optional
from ..utils import VectorDatabaseManager

import json

class ArticleWriterState(BaseModel):
    theme: str = ""
    drafts_documents: List[str] = []
    results: str = ""
    discussion: str = ""
    conclusion: str = ""

class ArticleWriterFlow(Flow[ArticleWriterState]):
    
    articles_db = VectorDatabaseManager()

    @start()
    def start_flow(self):
        # Declara o tema do artigo
        self.state.theme = "Acopladores ópticos"
        self.articles_db.initialize_db(
            persist_directory="article_vectorstore",
            collection_name="flow_test_collection"
        )
        print(FetchMetadataTool()._run(source='Resultado1.pdf', type='draft'))
        self.state.drafts_documents = 'Resultado1.pdf, Resultado2.pdf'

    @listen(start_flow)
    def write_results_and_conclusion(self):
        res_and_conc = ResultAndDiscussionCrew().crew().kickoff(
            inputs={
                "theme": self.state.theme,
                "document_list": self.state.drafts_documents
            }
        )

        print(res_and_conc.raw)

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()

def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()

if __name__ == "__main__":
    kickoff()
