from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM


@CrewBase
class ReportWriterCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    consolidation_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8000,
        max_tokens=128000,
        temperature=0.5
    )
    writer_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8000,
        max_tokens=128000,
        temperature=0.6
    )
    
    @agent
    def report_consolidator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_consolidator'],
            llm=self.consolidation_llm
        )
    
    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            llm=self.writer_llm
        )
    
    @task
    def final_critical_consolidation (self) -> Task:
        return Task(
            config=self.tasks_config['final_critical_consolidation']
        )
    
    @task
    def final_key_points_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['final_key_points_consolidation']
        )
    
    @task
    def final_methodology_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['final_methodology_consolidation']
        )
    
    @task
    def final_results_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['final_results_consolidation']
        )
    
    @task
    def final_elements_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['final_elements_consolidation']
        )
    
    @task
    def final_report_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['final_report_consolidation']
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
    