from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool


@CrewBase
class ResultsCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434"
    )

    @agent
    def research_draft_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['research_draft_analyst'],
            llm=self.llm,
            tools=[FetchMetadataTool(), FetchArticlesTool()]
        )
    
    @agent
    def results_and_discussion_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['results_and_discussion_writer'],
            llm=self.llm,
        )
    
    @task
    def draft_report_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['draft_report_analysis_task'],
            tools=[FetchMetadataTool(), FetchArticlesTool()]
        )
    
    @task
    def results_section_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['results_section_writing_task'],
        )

    @task
    def discussion_section_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['discussion_section_writing_task'],
            tools=[QueryArticlesTool()]
        )
    
    @task
    def final_section_writing(self) -> Task:
        return Task(
            config=self.tasks_config['final_section_writing'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )