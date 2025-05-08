from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
from .crews.outline_crew.outline_crew import OutlineCrew
from .crews.res_and_disc_research_crew.rdresearch_crew import RDResearchCrew
from .crews.final_editing_crew.final_editing_crew import FinalEditingCrew
from pydantic import BaseModel  
from ..tools import QueryArticlesTool
from typing import Dict, List
from ..utils import VectorDatabaseManager
# from .types.doc_report import AnaliseCriticaResultadosDiscussao
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
import asyncio
import itertools
import json

class ChunkReview(BaseModel):
    """A class representing a chunk review."""
    critical_analysis: List[str]
    key_points_detailing: List[str]
    methodology_analysis: List[str]
    techenical_elements: List[str]


class ArticleWriterState(BaseModel):
    drafts_documents: List[str] = ['Resultado1.pdf']
    full_analysis: str = ""
    technical_elements: str = ""
    results_discussion_outline: dict = {}
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
        results_discussion_outline = """
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
      "discussion_topics": [
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
          "numerical_results_to_include": [
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
          "numerical_results_to_include": [
            "CR(Out) > 0,30 em 100\u0025 dos casos simulados."
          ]
        }
      ]
    },
    {
      "section_title": "Discussão",
      "discussion_topics": [
        {
          "topic": "RELACIONAR os resultados obtidos com a literatura anterior, destacando a viabilidade da solução proposta e sugerindo necessidade de comparação quantitativa."
        },
        {
          "topic": "ANALISAR as implicações teóricas do uso de algoritmos genéticos na modelagem de circuitos ópticos multientrada."
        },
        {
          "topic": "IDENTIFICAR as principais limitações da pesquisa, como: Ausência de validação experimental; Violação parcial de critérios geométricos; Ausência de análise estatística entre múltiplas execuções."
        },
        {
          "topic": "DESTACAR as contribuições do estudo, incluindo: Preenchimento de lacuna na literatura sobre portas OR quíntuplas em fibra óptica; Redução da complexidade estrutural em circuitos lógicos ópticos; Proposta replicável com base em código-fonte claro e condições bem definidas.",
          "visual_elements_to_contextualize": [
            {
              "name": "Figuras 13, 15, 16",
              "description": "Diagramas que demonstram a eficiência e o desempenho da porta OR."
            },
            {
              "name": "Código-fonte 12",
              "description": "O código utilizado para a modelagem da porta lógica OR de cinco entradas."
            }
          ]
        }
      ] 
    }
  ]
}        
        """

    @listen(start_flow)
    def generate_res_and_disc_outlines(self):
        draft_report = self.articles_db.search_doc_by_meta(
            source='Relatorio.pdf', metadata_only=False)
        outcrew_output = OutlineCrew().crew().kickoff(
            inputs={
                "report": draft_report['Relatorio.pdf']['text_content'],
            }
        )
        self.state.results_discussion_outline = outcrew_output

    @listen(generate_res_and_disc_outlines)
    async def res_and_disc_chapter_generation(self):
      # Função para chamada assincrona da crew de escrita dos tópicos
      async def acall_write_topic_crew(
        section_title, 
        topic, 
        visual_elements_to_contextualize, 
        numerical_results_to_include,
        rhetorical_purpose,
        narrative_guidance,
        subsection_flow
      ):
        reasearch_crew_output = await RDResearchCrew().crew().kickoff_async(
          inputs={
            "section_title": section_title,
            "discussion_topic": topic,
            "visual_elements_to_contextualize": visual_elements_to_contextualize,
            "numerical_results_to_include": numerical_results_to_include,
            "rhetorical_purpose": rhetorical_purpose,
            "narrative_guidance": narrative_guidance,
            "subsection_flow": subsection_flow
          }
        )
        return reasearch_crew_output
      
      async def acall_final_editing_crew(
        section_title: str,
        concatenated_text: str,
        visual_elements_list: str
      ):
        final_edit_output = FinalEditingCrew().crew().kickoff_async(
          inputs={
            "section_title": section_title,
            "concatenated_text": concatenated_text,
            "visual_elements_list": visual_elements_list
          }
        )
        return final_edit_output

      #outline = json.loads(self.state.results_discussion_outline)
      subs_outlines = self.state.results_discussion_outline
      async_tasks_to_exec = []
      visual_elements_to_retrive = {}
      section_and_topics = {}
      for subsection in subs_outlines:
        for disc_topic in subsection['discussion_topics']:
          visual_elements_to_contextualize = "\n".join([
            (
              f"- name: {element["identifier"]} {element["name"]}; "
              f"description: {element["description"]}; "
              f"role_in_topic: {element["role_in_topic"]}"
            )
            for element in disc_topic.get('visual_elements', [])
          ])
          if subsection['subsection_name'] not in visual_elements_to_retrive:
            visual_elements_to_retrive[subsection['subsection_name']] = ""
      
          visual_elements_to_retrive[subsection['subsection_name']] += visual_elements_to_contextualize + "\n"
          numerical_results_to_include = "\n".join([
            (
               f'- value: {result['verbatim_value']}; '
               f'context: {result["context_description"]}; '
               f'associated visual: {result["associated_visual"]}'
            ) 
            for result in disc_topic.get('numerical_results', [])
          ])
          async_tasks_to_exec.append(
            asyncio.create_task(
              acall_write_topic_crew(
                  subsection['subsection_name'], 
                  disc_topic['topic'], 
                  visual_elements_to_contextualize, 
                  numerical_results_to_include,
                  disc_topic['rhetorical_purpose'],
                  disc_topic['narrative_guidance'],
                  subsection['subsection_flow']
              )
            )
          )
        topic_w_outputs = await asyncio.gather(*async_tasks_to_exec)
        section_and_topics[subsection['subsection_name']] = [
          topic_w_output.json_dict for topic_w_output in topic_w_outputs
        ]
      async_tasks_to_exec = []
      for section_title, discussion_topics in section_and_topics.items():
        concatenated_text = '\n'.join([topic['text'] for topic in discussion_topics])
        visual_elements_list = visual_elements_to_retrive[section_title]
        async_tasks_to_exec.append(
          asyncio.create_task(
            acall_final_editing_crew(
              section_title=section_title,
              concatenated_text=concatenated_text,
              visual_elements_list=visual_elements_list
            )
          )
        )
      final_edit_outputs = await asyncio.gather(*async_tasks_to_exec)
      final_edit_outputs = [final_edit_output.json_dict for final_edit_output in final_edit_outputs]
      
      print(final_edit_outputs)


def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()


def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()


if __name__ == "__main__":
    kickoff()
