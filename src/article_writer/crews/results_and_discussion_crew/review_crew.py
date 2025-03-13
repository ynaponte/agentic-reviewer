from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from langchain_ollama import ChatOllama
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool
from src.article_writer.types.doc_report import ChunkReport


@CrewBase
class ReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    reviewer_llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434",
        n=3,
        max_completion_tokens=2000,
        max_tokens=8192,
        temperature=0.4
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
        return Task(
            config=self.tasks_config['document_reading'],
            tools=[FetchArticlesTool()],
            output_pydantic=ChunkReport
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