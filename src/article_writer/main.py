from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
from .crews.outline_crew.outline_crew import OutlineCrew
from .crews.technical_chapter_writer.technical_chapter_writer_crew import TechnicalChapterWriterCrew
from ..tools import QueryArticlesTool
from typing import Dict, List
from ..utils import VectorDatabaseManager
# from .types.doc_report import AnaliseCriticaResultadosDiscussao
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
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
    # chunk_reports: List[AnaliseCriticaResultadosDiscussao] = []


class ArticleWriterFlow(Flow[ArticleWriterState]):

    articles_db = VectorDatabaseManager()

    @start()
    def start_flow(self):
        # Inicializa a base de dados para uso das ferramentas
        self.articles_db.initialize_db(
            persist_directory="article_vectorstore",
            collection_name="flow_test_collection"
        )

        self.state.results_discussion_outline = """
Outline de Resultados e Discussao:

{
  "section_name": "Resultados e Discussão",
  "discution_topics": [
    {
      "topic": "APRESENTAR o propósito do capítulo: relatar os resultados obtidos com a modelagem da porta lógica OR de cinco entradas utilizando algoritmos genéticos e discutir suas implicações."
    },
    {
      "topic": "ANTECIPAR os aspectos que serão explorados: eficiência do algoritmo, desempenho óptico da porta OR, robustez estrutural e limitações identificadas."  
    }
  ],
  "subsections": [
    {
      "section_title": "Resultados",
      "discution_topics": [
        {
          "topic": "DESCREVER a eficiência do algoritmo genético na convergência das soluções ao longo das gerações.",
          "visual_elements_to_contextualize": [
            {
              "name": "Figura 13",
              "description": "Evolução do fitness para a primeira porta."
            },
            {
              "name": "Figura 15",
              "description": "Evolução do fitness para a segunda porta."
            }
          ],
          "numerical_results": [
            "Saturação observada por volta da 30ª e 50ª gerações."
          ]
        },
        {
          "topic": "ANALISAR os resultados das tabelas verdade para validar a funcionalidade lógica da porta OR com cinco entradas.",
          "visual_elements_to_contextualize": [
            {
              "name": "Tabela 1",
              "description": "Contrastes da primeira porta."
            },
            {
              "name": "Tabela 2",
              "description": "Contrastes da segunda porta."
            }
          ],
          "numerical_results": [
            "CR(Out) > 0,30 em 100% dos casos simulados."
          ]
        }
      ]
    },
    {
      "section_title": "Discussão",
      "discution_topics": [
        {
          "topic": "RELACIONAR os resultados obtidos com a literatura anterior, destacando a viabilidade da solução proposta e sugerindo necessidade de comparação quantitativa."
        },
        {
          "topic": "ANALISAR as implicações teóricas do uso de algoritmos genéticos na modelagem de circuitos ópticos multientrada."
        },
        {
          "topic": "IDENTIFICAR as principais limitações da pesquisa, como:",
          "numerical_results": [
            "Ausência de validação experimental;",
            "Violação parcial de critérios geométricos;",
            "Ausência de análise estatística entre múltiplas execuções."
          ]
        },
        {
          "topic": "DESTACAR as contribuições do estudo, incluindo:",
          "visual_elements_to_contextualize": [
            {
              "name": "Figuras 13, 15, 16",
              "description": "Diagramas que demonstram a eficiência e o desempenho da porta OR."
            },
            {
              "name": "Código-fonte 12",
              "description": "O código utilizado para a modelagem da porta lógica OR de cinco entradas."
            }
          ],
          "numerical_results": [
            "Preenchimento de lacuna na literatura sobre portas OR quíntuplas em fibra óptica;",
            "Redução da complexidade estrutural em circuitos lógicos ópticos;",
            "Proposta replicável com base em código-fonte claro e condições bem definidas."
          ]
        }
      ]
    }
  ]
        
        """

    @listen(start_flow)
    def generate_outlines(self):
        
        draft_report = self.articles_db.search_doc_by_meta(
            source='Relatorio.pdf', metadata_only=False)
        outcrew_output = OutlineCrew().crew().kickoff(
            inputs={
                "report": draft_report['Relatorio.pdf']['text_content'],
            }
        )
        for task_output in outcrew_output.tasks_output:
            if task_output.name == "generate_outline_results_discussion":
                self.state.results_discussion_outline = task_output.json_dict
                print(
                    f"Outline de Resultados e Discussao:\n\n{task_output.raw}")
            elif task_output.name == "generate_outline_conclusion":
                self.state.conclusion_outline = task_output.json_dict
                print(f"Outline de conclusão:\n\n{task_output.raw}")
            else:
                self.state.methodology_outline = task_output.json_dict
                print(f"Outline de metodologia:\n\n{task_output.raw}")

    @listen(generate_outlines)
    def res_and_disc_chapter_generation(self):
        return
        chapter = TechnicalChapterWriterCrew().crew().kickoff(
            inputs={
                "chapter_title": "Resultados e Discussão",
                "chapter_outline": self.state.results_discussion_outline,
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
