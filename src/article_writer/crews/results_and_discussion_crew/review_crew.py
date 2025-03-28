from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from langchain_ollama import ChatOllama
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool
#from src.article_writer.types.doc_report import AnaliseCriticaResultadossDiscussao
from src.article_writer.types.results_report import ElementsExtraction



@CrewBase
class ReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8000,
        max_tokens=128000,
        temperature=0.5
    )

    @agent
    def critical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['critical_analyst'],
            llm=self.std_llm,
            tools=[FetchArticlesTool()]
        )

    @agent
    def technical_data_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_data_extractor'],
            llm=self.std_llm,
            tools=[FetchArticlesTool()]
        )
    
    @agent
    def report_redactor(self) -> Agent:
        return Agent(
            config=self.agents_config['report_redactor'],
            llm=self.std_llm
        )

    @task
    def critical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['critical_analysis'],
            tools=[FetchArticlesTool()],
            #output_pydantic=AnaliseCriticaResultadosDiscussao
        )
    
    @task
    def methodology_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['methodology_analysis'],
            tools=[FetchArticlesTool()],
        )
    
    @task
    def results_analysis(self) ->Task:
        return Task(
            config=self.tasks_config['results_analysis'],
            tools=[FetchArticlesTool()],
        )     

    @task
    def elements_extraction(self) -> Task:
        return Task(
            config=self.tasks_config['elements_extraction'],
            tools=[FetchArticlesTool()],
            #output_pydantic=ElementsExtraction
        )
    
    @task
    def report_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['report_consolidation']
        )
    
    @crew
    def crew(self) -> Crew:
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
        return crew