from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM


@CrewBase
class ReportCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = LLM(
        model="ollama/llama3.1:latest",
        base_url="http://localhost:11434"
    )

    @agent
    def results_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['results_writer'],
            llm=self.llm
        )
    
    @task
    def results_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['results_writing_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )