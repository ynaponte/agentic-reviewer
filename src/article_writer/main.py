from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
from .crews.outline_crew.outline_crew import OutlineCrew
from .crews.res_disc_writer.res_dics_writer_crew import ResultAndDiscussionCrew
from ..tools import FetchArticlesTool
from typing import Dict, List 
from ..utils import VectorDatabaseManager
#from .types.doc_report import AnaliseCriticaResultadosDiscussao

import asyncio
import json


class ChunkReview(BaseModel):
    """A class representing a chunk review."""
    critical_analysis: List[str]
    key_points_detailing: List[str]
    methodology_analysis: List[str]
    techenical_elements: List[str]


class ArticleWriterState(BaseModel):
    theme: str = (
        "Utilização de algorítimos genéticos para criação de portas lógicas ópticas "
        "em cristal fotônico"
    )
    drafts_documents: List[str] = ['Resultado1.pdf']
    full_analysis: str = ""
    technical_elements: str = ""
    results_discussion_outline: str = ""
    conclusion_outline: str = ""
    methodology_outline: str = ""
    discussion: str = ""
    conclusion: str = ""
    #chunk_reports: List[AnaliseCriticaResultadosDiscussao] = []


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
    def generate_outlines(self):
        draft_report = self.articles_db.search_doc_by_meta(source='Relatorio.pdf', metadata_only=False)
        outcrew_output = OutlineCrew().crew().kickoff(
            inputs={
                "report": draft_report['Relatorio.pdf']['text_content'],
            }
        )
        for task_output in outcrew_output.tasks_output:
            if task_output.name == "generate_outline_results_discussion":
                self.state.results_discussion_outline = task_output.raw
                print(f"Outline de Resultados e Discussao:\n\n{task_output.raw}")
            elif task_output.name == "generate_conclusion_outline":
                self.state.conclusion_outline = task_output.raw
                print(f"Outline de conclusão:\n\n{task_output.raw}")
            else:
                self.state.methodology_outline = task_output.raw
                print(f"Outline de metodologia:\n\n{task_output.raw}")

    @listen(generate_outlines)
    def res_and_disc_chapter_generation(self):
        chapter = ResultAndDiscussionCrew().crew().kickoff(
            inputs={
                "chapter_outline": self.state.results_discussion_outline,
                "technical_report": self.state.technical_elements
            }
        )
        print(chapter.raw)

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()

def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()

if __name__ == "__main__":
    kickoff()
