from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM


@CrewBase
class ChunkReviewCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/qwen2.5:14b",
        base_url="http://localhost:11434",
        max_completion_tokens=128000,
        max_tokens=128000,
        temperature=0.5
    )

    @agent
    def critical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['critical_analyst'],
            llm=self.std_llm,
        )
    @agent
    def methodology_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['methodology_evaluator'],
            llm=self.std_llm,
        )
    
    @task
    def methodology_results_consistency(self) -> Task:
        return Task(
            config=self.tasks_config['methodology_results_consistency']
        )
    
    @task
    def results_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['results_analysis'],
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