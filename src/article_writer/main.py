from pydantic import BaseModel
from crewai.flow import Flow, listen, start, and_
from .crews.res_and_disc_crew.r_d_outline_crew.r_d_outline_crew import RDOutlineCrew
from .crews.res_and_disc_crew.r_d_topic_rag_crew.r_d_topic_rag_crew import RDTopicRagCrew
from .crews.conclusion_crew.c_outline_crew.c_outline_crew import COutlineCrew
from .crews.conclusion_crew.c_topic_rag_crew.c_topic_rag_crew import ConclusionTopicRagCrew
from .crews.methodology_crew.methodology_outline_crew.methodology_outline_crew import MethodologyOutlineCrew
from .crews.methodology_crew.methodology_topic_rag_crew.methodology_topic_rag_crew import MethodologyTopicRagCrew
from .crews.theoretical_fdmt_crew.theoretical_fdmt_outline_crew.theoretical_fdmt_outline_crew import TheoreticalFdmtOutlineCrew
from .crews.theoretical_fdmt_crew.theoretical_fdmt_rag_crew.theoretical_fdmt_topic_rag_crew import TheoreticalFdmtTopicRagCrew
from typing import Dict, List
from ..utils import VectorDatabaseManager
# from .types.doc_report import AnaliseCriticaResultadosDiscussao
import asyncio
import json


class ArticleWriterState(BaseModel):
    draft_report: str = ""
    sections_and_content: List[Dict[str, str]] = []
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
  
  @listen(start_flow)
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
  
  #@listen(start_flow)
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
  
  @listen(res_and_disc_outline_generation)
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
    # Inicializa o dicionário para armazenar o conteúdo gerado para a seção e suas subseções
    results_and_discussion_section = {
      "section_name": "Resultado e Discussão", 
      "topics": []
    }
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
                disc_topic['topic_title'],
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
      for topic_content in subsection_topics_content:
        # Prepara os resultados para serem armazenados em formato JSON, seguindo uma estrutura padrão,
        # onde cada tópico, além da informação sobre seu nome e seu conteúdo textual, carrega também a 
        # qual seção ele pertence. Caso seja da principal, se informa como 'null'.
        topic_content = topic_content.json_dict 
        topic_content['subsection'] = subsection['subsection_name']
        results_and_discussion_section['topics'].append(topic_content)
    
    self.state.sections_and_content.append(results_discussion_outline)
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
  #@listen(dev_res_and_disc_outline_generation)
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

  @listen(conclusion_outline_generation)
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
    conclusion_section = {
      "section_name": "Conclusão",
      "topics": []
    }
    for discussion_topic in discussion_topics:
      async_tasks_to_exec.append(
        asyncio.create_task(
          acall_write_topic_crew(
            discussion_topic['topic_title'],
            discussion_topic['rhetorical_purpose'],
            discussion_topic['topic_description'],
            discussion_topic['narrative_guidance']
          )
        )
      )
    conclusion_topics_content = await asyncio.gather(*async_tasks_to_exec)
    for topic_content in conclusion_topics_content:
      topic_content = topic_content.json_dict
      topic_content['subsection'] = 'main'
      conclusion_section['topics'].append(topic_content)

    self.state.sections_and_content.append(conclusion_outline)
    return conclusion_section
  
  @listen(conclusion_chapter_generation)
  def methodology_outline_generation(self):
    methodology_outline = MethodologyOutlineCrew().crew().kickoff(
        inputs={
          "report": self.state.draft_report['Relatorio.pdf']['text_content'],
          "generated_sections_content": self.state.sections_and_content
        }
    )
    return methodology_outline
  
  @listen(methodology_outline_generation)
  async def methodology_chapter_generation(self, methodology_outline):
    async def acall_write_topic_crew(
      subsection_name,
      subsection_description,
      topic_title,
      rhetorical_purpose,
      topic_description,
      narrative_guidance
    ):
      methodology_topics_content = (
        await MethodologyTopicRagCrew()
        .crew()
        .kickoff_async(
          inputs={               
            "methodology_subsection_name": subsection_name,
            "methodology_subsection_description": subsection_description,
            "methodology_topic_title": topic_title,
            "methodology_topic_description": topic_description,
            "methodology_rhetorical_purpose": rhetorical_purpose,
            "methodology_narrative_guidance": narrative_guidance
          }
        )
      )
      return methodology_topics_content
    
    methodology_section = {
      "section_name": "Metodologia",
      "topics": []
    }
    for subsecion in methodology_outline['subsections']:
      async_tasks_to_exec = []  # Inicia/reseta a lista de tarefas
      subsecion_name = subsecion['subsection_name']
      subsecion_description = subsecion['subsection_description']
      for discussion_topic in subsecion['discussion_topics']:
        async_tasks_to_exec.append(
          asyncio.create_task(
            acall_write_topic_crew(
              subsection_name=subsecion_name,
              subsection_description=subsecion_description,
              topic_title=discussion_topic['topic_title'],
              topic_description=discussion_topic['topic_description'],
              rhetorical_purpose=discussion_topic['rhetorical_purpose'],
              narrative_guidance=discussion_topic['narrative_guidance']
            )
          )
        )
      methodology_topics_content = await asyncio.gather(*async_tasks_to_exec)
      for topic_content in methodology_topics_content:
        topic_content = topic_content.json_dict
        topic_content['subsection'] = subsecion_name  # Cria uma nova chave no dicionário para armazenar a que seção o tópico pertence
        methodology_section['topics'].append(topic_content)

    self.state.sections_and_content.append(methodology_outline)
    return methodology_section    
  
  @listen(and_(methodology_outline_generation, methodology_chapter_generation))
  def theoretical_fdmt_outline_generation(self, methodology_outline, _):
    theoretical_fdmt_outline = TheoreticalFdmtOutlineCrew().crew().kickoff(
      inputs={
        "report": self.state.draft_report['Relatorio.pdf']['text_content'],
        "methodology_outline": methodology_outline
      }
    )
    return theoretical_fdmt_outline

  @listen(theoretical_fdmt_outline_generation)
  async def theoretical_fdmt_chapter_generation(self, theoretical_fdmt_outline):
    async def acall_write_topic_crew(
      subsection_title,
      subsection_description,
      topic_title,
      rhetorical_purpose,
      topic_description,
      narrative_guidance
    ):
      theoretical_fdmt_topics_content = (
        await TheoreticalFdmtTopicRagCrew()
        .crew()
        .kickoff_async(
          inputs={
            "theoretical_foundation_subsection_name": subsection_title,
            "theoretical_foundation_subsection_description": subsection_description,            
            "theoretical_foundation_topic_title": topic_title,
            "theoretical_foundation_rhetorical_purpose": rhetorical_purpose,
            "theoretical_foundation_topic_description": topic_description,
            "theoretical_foundation_narrative_guidance": narrative_guidance
          }
        )
      )
      return theoretical_fdmt_topics_content
    
    theoretical_fdmt_section = {
      "section_name": "Fundamentação Teórica",
      "topics": []
    }

    for subsecion in theoretical_fdmt_outline['subsections']:
      async_tasks_to_exec = []  # Inicia/reseta a lista de tarefas
      subsecion_name = subsecion['subsection_name']
      subsecion_description = subsecion['subsection_description']
      for discussion_topic in subsecion['discussion_topics']:
        async_tasks_to_exec.append(
          asyncio.create_task(
            acall_write_topic_crew(
              subsection_name=subsecion_name,
              subsection_description=subsecion_description,
              topic_title=discussion_topic['topic_title'],
              topic_description=discussion_topic['topic_description'],
              rhetorical_purpose=discussion_topic['rhetorical_purpose'],
              narrative_guidance=discussion_topic['narrative_guidance']
            )
          )
        )
      methodology_topics_content = await asyncio.gather(*async_tasks_to_exec)
      for topic_content in methodology_topics_content:
        topic_content = topic_content.json_dict
        topic_content['subsection'] = subsecion_name  # Cria uma nova chave no dicionário para armazenar a que seção o tópico pertence
        theoretical_fdmt_section['topics'].append(topic_content)

    self.state.sections_and_content.append(theoretical_fdmt_outline)
    return theoretical_fdmt_section
    
  @listen(theoretical_fdmt_chapter_generation)
  def save_results(self):
    print("Resultado:\n")
    print(self.state.sections_and_content)
    print("\nSalvando Resultados...")
    with open("resultados.json", "w", encoding="utf-8") as json_file:
      json.dump(self.state.sections_and_content, json_file, ensure_ascii=False, indent=4)

def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()


def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()


if __name__ == "__main__":
    kickoff()
