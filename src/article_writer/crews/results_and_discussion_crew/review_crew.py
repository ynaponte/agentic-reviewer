from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from langchain_ollama import ChatOllama
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool
from src.article_writer.types.doc_report import AnaliseCriticaResultadosDiscussao
from src.article_writer.types.results_report import ElementsExtraction



@CrewBase
class ReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434",
        max_completion_tokens=6000,
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
            tools=[FetchMetadataTool()]
        )

    @task
    def critical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['critical_analysis'],
            tools=[FetchArticlesTool()],
            #output_pydantic=AnaliseCriticaResultadosDiscussao
        )

    @task
    def elements_extraction(self) -> Task:
        return Task(
            config=self.tasks_config['elements_extraction'],
            tools=[FetchArticlesTool()],
            output_pydantic=ElementsExtraction
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