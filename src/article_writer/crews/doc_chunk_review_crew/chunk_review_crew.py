from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.tools import FetchArticlesTool


@CrewBase
class ChunkReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/llama3.1:latest",
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
    def report_redactor(self) -> Agent:
        return Agent(
            config=self.agents_config['report_redactor'],
            llm=self.std_llm
        )

    @task
    def critical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['critical_analysis'],
            tools=[FetchArticlesTool()]
        )
    
    @task
    def key_points_extraction(self) -> Task:
        return Task(
            config=self.tasks_config['key_points_extraction']
        )
    
    @task
    def methodology_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['methodology_analysis']
        )
    
    @task
    def results_analysis(self) ->Task:
        return Task(
            config=self.tasks_config['results_analysis']
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