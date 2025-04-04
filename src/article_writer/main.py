from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
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
    docs_chunks_reports: Dict[str, Dict[str, List]] = {}
    docs_reports: List[str] = []
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
            '''
            Defina o tamanho do lote de acordo?
                - Llama3.1:8b -> 
            :param: batch_size - tamanho do lote de chunks
            '''
            return [
                list(range(i, min(i + batch_size, num_chunks))) 
                for i in range(0, num_chunks, batch_size)
            ]
        
        def process_single_document(
            doc_name: str, 
            chunk_indexes: List[int], 
            batch_index: int,
            content: str
        ):
            output = ChunkReviewCrew().crew().kickoff(
                    inputs={
                        "target_document": doc_name,
                        "chunk_indexes": chunk_indexes,
                        "theme": self.state.theme,
                        "batch_number": batch_index,
                        "content": content         
                    }
                )
            return output

        tasks = []
        for draft_document in self.state.drafts_documents:
            doc_meta = self.articles_db.search_doc_by_meta(source=draft_document, type='draft')
            chunk_batches = batching(doc_meta[0]['total_chunks'], 2)
            for i, batch in enumerate(chunk_batches, start=1):
                content = FetchArticlesTool()._run(source=draft_document, chunk_id=batch, doc_type='draft')
                #task = asyncio.create_task(process_single_document(draft_document, batch, i))
                #tasks.append(task)
                tasks.append(process_single_document(draft_document, batch, i, content))

            #crews_outputs = await asyncio.gather(*tasks)
            crews_outputs = tasks
            chunk_report = {}
            # Armazena os resultados de cada chunk
            for crew_output in crews_outputs:
                # Indice em crew_output.tasks_output corresponte a task:
                #   0 -> 'critical_analysis'
                #   1 -> 'key_points_extraction'
                #   2 -> 'methodology_analysis'
                #   3 -> 'results_analysis'
                #   4 -> 'elements_extraction'
                #   5 -> 'report_consolidation'
                chunk_report.setdefault("critical_analysis", []).append(
                    crew_output.tasks_output[0].raw
                )
                chunk_report.setdefault("key_points_extraction", []).append(
                    crew_output.tasks_output[1].raw
                )
                chunk_report.setdefault("methodology_analysis", []).append(
                    crew_output.tasks_output[2].raw
                )
                chunk_report.setdefault("results_analysis", []).append(
                    crew_output.tasks_output[3].raw
                )

            self.state.docs_chunks_reports[draft_document] = chunk_report

    @listen(doc_process_by_chunks)
    def report_consolidation(self):
        for rprt_frag in self.state.docs_chunks_reports:
            report = ReportWriterCrew.crew().kickoff(
                inputs={
                    "critical_analysis_fragments": rprt_frag["critical_analysis"],
                    "key_points_fragments": rprt_frag["key_points_extraction"],
                    "methodology_fragments": rprt_frag["methodology_analysis"],
                    "results_fragments": rprt_frag["results_analysis"],
                }
            )

            self.state.docs_reports.append(report.raw)

    @listen(report_consolidation)
    def res_and_disc_sec_writting(self):
        pass

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()

def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()

if __name__ == "__main__":
    kickoff()
