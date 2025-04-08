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
    async def doc_process_by_chunks(self):
        def batching(num_chunks, batch_size):
            return [
                list(range(i, min(i + batch_size, num_chunks))) 
                for i in range(0, num_chunks, batch_size)
            ]
        
        async def process_single_document(
            doc_name: str, 
            chunk_indexes: List[int], 
            batch_index: int,
            content: str
        ):
            output = await ChunkReviewCrew().crew().kickoff_async(
                    inputs={
                        "target_document": doc_name,
                        "chunk_indexes": chunk_indexes,
                        "batch_number": batch_index,
                        "content": content         
                    }
                )
            return output

        fragmented_analysis = {}   # Dicionário responsável por armazenar os fragmentos de analises
        for draft_document in self.state.drafts_documents:
            tasks = []   # Inicializa a lista para as tarefas assíncronas
            doc_meta = self.articles_db.search_doc_by_meta(source=draft_document, type='draft')
            chunk_batches = batching(doc_meta[0]['total_chunks'], 3)
            for i, batch in enumerate(chunk_batches, start=1):
                content = FetchArticlesTool()._run(source=draft_document, chunk_id=batch, doc_type='draft')
                tasks.append(asyncio.create_task(process_single_document(draft_document, batch, i, content)))

            crews_outputs = await asyncio.gather(*tasks)
            # Separação e armazenamento dos fragmentos de analises
            for crew_output in crews_outputs:
                fragmented_analysis.setdefault("methodology_results_consistency", []).append(
                    crew_output.tasks_output[0].raw
                )
                fragmented_analysis.setdefault("results_analysis", []).append(
                    crew_output.tasks_output[1].raw
                )
                self.state.technical_elements += crew_output.tasks_output[2].raw
                
        full_analysis = ""
        for frags in fragmented_analysis.values():
            output = ReportWriterCrew().crew().kickoff(
            inputs={
                "input_text": "".join(
                        frag for frag in frags
                    ),
                }
            )
            full_analysis += output.raw

        full_analysis = ReportWriterCrew().crew().kickoff(
            inputs={
                "input_text": full_analysis
            }
        )
        self.state.full_analysis = full_analysis.raw

    @listen(doc_process_by_chunks)
    def outline_generation(self):
        asd = self.state.full_analysis
        outcrew_output = OutlineCrew().crew().kickoff(
            inputs={
                "analysis": self.state.full_analysis,
                "theme": self.state.theme
                }
        )
        for task_output in outcrew_output.tasks_output:
            if task_output.name == "generate_results_discussion_outline":
                self.state.results_discussion_outline = task_output.raw
                print(f"Outline de Resultados e Discussao:\n\n{task_output.raw}")
            elif task_output.name == "generate_conclusion_outline":
                self.state.conclusion_outline = task_output.raw
                print(f"Outline de conclusão:\n\n{task_output.raw}")
            else:
                self.state.methodology_outline = task_output.raw
                print(f"Outline de metodologia:\n\n{task_output.raw}")

    @listen(outline_generation)
    def res_and_disc_chapter_generation(self):
        chapter = ResultAndDiscussionCrew().crew().kickoff(
            inputs={
                "chapter_outline": self.state.results_discussion_outline,
                "theme": self.state.theme
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
