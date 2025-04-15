from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.article_writer.types.results_report import ElementsList


@CrewBase
class OutlineCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/qwen2.5:14b-instruct-q8_0",
        base_url="http://localhost:11434",
        max_completion_tokens=128000,
        max_tokens=128000,
        temperature=0.5
    )

    @agent
    def core_chapters_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['core_chapters_outliner'],
            llm=self.std_llm,
        )

    @task
    def generate_outline_results_discussion(self) -> Task:
        return Task(
            config=self.tasks_config['generate_outline_results_discussion']
        )
    
    @task
    def generate_outline_conclusion(self) -> Task:
        return Task(
            config=self.tasks_config['generate_outline_conclusion']
        )
    
    @task
    def generate_outline_methodology(self) -> Task:
        return Task(
            config=self.tasks_config['generate_outline_methodology']
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