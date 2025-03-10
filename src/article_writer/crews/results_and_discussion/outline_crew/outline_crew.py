from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool


@CrewBase
class OutlineCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434",
        max_completion_tokens=8192,
        max_tokens=32768,
        temperature=0.1
    )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            llm=self.llm,
            tools=[FetchMetadataTool(), FetchArticlesTool()]
        )
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            llm=self.llm,
            tools=[QueryArticlesTool()]
        )

    @task
    def initial_assessment(self) -> Task:
        return Task(
            config=self.tasks_config['initial_assessment']
        )
    
    @task
    def query_generation(self) -> Task:
        return Task(
            config=self.tasks_config['query_generation']
        )
    
    @task
    def query_execution(self) -> Task:
        return Task(
            config=self.tasks_config['query_execution']
        )