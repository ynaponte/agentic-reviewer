from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool


@CrewBase
class ReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    reviewer_llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434",
        max_completion_tokens=8192,
        max_tokens=131072,
        temperature=0.2
    )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['reviewer'],
            llm=self.reviewer_llm,
            tools=[FetchArticlesTool()]
        )
    
    @task
    def document_reading(self) -> Task:
        print(self.inputs)
        return Task(
            config=self.tasks_config['document_reading'],
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