from pydantic import BaseModel
from crewai.flow import Flow, listen, start, and_
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
from .crews.res_and_disc_crew.r_d_outline_crew.r_d_outline_crew import RDOutlineCrew
from .crews.res_and_disc_crew.r_d_topic_rag_crew.r_d_topic_rag_crew import RDTopicRagCrew
from .crews.conclusion_crew.c_outline_crew.c_outline_crew import COutlineCrew
from .crews.conclusion_crew.c_topic_rag_crew.c_topic_rag_crew import ConclusionTopicRagCrew
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
    draft_report: str = ""
    results_discussion_section: dict = {}
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
    self.state.draft_report = self.articles_db.search_doc_by_meta(
      source='Relatorio.pdf', metadata_only=False
    )
  
  #@listen(start_flow)
  def res_and_disc_outline_generation(self): 
    subsections_outline = RDOutlineCrew().crew().kickoff(
        inputs={
          "report": self.state.draft_report['Relatorio.pdf']['text_content'],
        }
    )
    return {
      "section_name" : "Resultados e Discussão",
      "subsections": subsections_outline
    }
  
  @listen(start_flow)
  def dev_res_and_disc_outline_generation(self):
    outline_file_path = './outlines/res_and_disc_outline.json'
    try:
      with open(outline_file_path, 'r', encoding='utf-8') as json_file:
        res_and_disc_outline = json.load(json_file)
    except FileNotFoundError:
      print(f"Warning: The outline file '{outline_file_path}'was not found. "
            "Generating a new outline...")
      res_and_disc_outline = self.res_and_disc_outline_generation()
    return res_and_disc_outline
  
  #@listen(res_and_disc_outline_generation)
  async def res_and_disc_chapter_generation(self, results_discussion_outline):
    # Função para permitir chamada assincrona da crew de escrita dos tópicos
    async def acall_write_topic_crew(
      section_title, 
      topic, 
      topic_description,
      visual_elements_to_contextualize, 
      numerical_results_to_include,
      rhetorical_purpose,
      narrative_guidance,
      subsection_flow
    ):
      topic_content = await RDTopicRagCrew().crew().kickoff_async(
        inputs={
          "section_title": section_title,
          "discussion_topic": topic,
          "topic_description": topic_description,
          "visual_elements_to_contextualize": visual_elements_to_contextualize,
          "numerical_results_to_include": numerical_results_to_include,
          "rhetorical_purpose": rhetorical_purpose,
          "narrative_guidance": narrative_guidance,
          "subsection_flow": subsection_flow
        }
      )
      return topic_content

    subs_outlines = results_discussion_outline.get('subsections', [])
    visual_elements_to_retrive = {}
    subsections_content = {}
    for subsection in subs_outlines:
      async_tasks_to_exec = []
      for disc_topic in subsection['discussion_topics']:
        # TODO: Passar o processamento do input para a crew, no @before_kickoff.
        visual_elements_to_contextualize = "\n".join([
          (
            f"- name: {element['name']}; role_in_topic: {element['role_in_topic']}"
          )
          for element in disc_topic.get('visual_elements', [])
        ])
        if subsection['subsection_name'] not in visual_elements_to_retrive:
          visual_elements_to_retrive[subsection['subsection_name']] = ""
    
        visual_elements_to_retrive[subsection['subsection_name']] += visual_elements_to_contextualize + "\n"
        numerical_results_to_include = "\n".join([
          (
              f'- verbatim value: {result['verbatim_value']}; '
              f'role_in_topic: {result["role_in_topic"]}; '
              f'associated visual: {result["associated_visual"]}'
          ) 
          for result in disc_topic.get('numerical_results', [])
        ])
        async_tasks_to_exec.append(
          asyncio.create_task(
            acall_write_topic_crew(
                subsection['subsection_name'], 
                disc_topic['topic'],
                disc_topic['topic_description'],
                visual_elements_to_contextualize, 
                numerical_results_to_include,
                disc_topic['rhetorical_purpose'],
                disc_topic['narrative_guidance'],
                subsection['subsection_flow']
            )
          )
        )
      subsection_topics_content = await asyncio.gather(*async_tasks_to_exec)
      subsections_content[subsection['subsection_name']] = [
        topic_content.json_dict for topic_content in subsection_topics_content
      ]
    results_and_discussion_section = {
      "section_name": "Resultado e Discussão", 
      "topics_content": subsections_content
    }
    self.state.results_discussion = results_discussion_outline
    return results_and_discussion_section
  
  @listen(res_and_disc_chapter_generation)
  def conclusion_outline_generation(self, results_and_discussion_section):
    conclusion_outline = COutlineCrew().crew().kickoff(
        inputs={
          "report": self.state.draft_report['Relatorio.pdf']['text_content'],
          "generated_sections_content": results_and_discussion_section
        }
    )
    return conclusion_outline

  # Utilizado para agilizar desenvolvimento
  @listen(dev_res_and_disc_outline_generation)
  def dev_conclusion_outline_generation(self, results_discussion_outline):
    outline_file_path = './outlines/conclusion_outline.json'
    try:
      with open(outline_file_path, 'r', encoding='utf-8') as json_file:
        conclusion_outline = json.load(json_file)
    except FileNotFoundError:
      print(f"Warning: The outline file '{outline_file_path}'was not found. "
            "Generating a new outline...")
      conclusion_outline = self.conclusion_outline_generation(results_discussion_outline)
    return conclusion_outline

  @listen(dev_conclusion_outline_generation)
  # NOTA: talvez seja bom esperar o também o termino da execução de `res_and_disc_chapter_generation`, para evitar muitas tarefas assícronas rodando.
  async def conclusion_chapter_generation(
    self,
    conclusion_outline
  ):
    async def acall_write_topic_crew(
      topic,
      rhetorical_purpose,
      topic_description,
      narrative_guidance
    ):
      conclusion_topics_content = (
        await ConclusionTopicRagCrew()
        .crew()
        .kickoff_async(
          inputs={               
            "conclusion_topic": topic,
            "rhetorical_purpose": rhetorical_purpose,
            "topic_description": topic_description,
            "narrative_guidance": narrative_guidance
          }
        )
      )
      return conclusion_topics_content
    
    discussion_topics = conclusion_outline['discussion_topics']
    async_tasks_to_exec = []
    for discussion_topic in discussion_topics:
      async_tasks_to_exec.append(
        asyncio.create_task(
          acall_write_topic_crew(
            discussion_topic['topic'],
            discussion_topic['rhetorical_purpose'],
            discussion_topic['topic_description'],
            discussion_topic['narrative_guidance']
          )
        )
      )
    conclusion_topics = await asyncio.gather(*async_tasks_to_exec)
    return conclusion_topics
  
def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()


def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()


if __name__ == "__main__":
    kickoff()
