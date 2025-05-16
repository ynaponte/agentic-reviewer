from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, after_kickoff
from crewai.llm import LLM
from .pydantic_output.pydantic_output import ConclusionSectionOutline


@CrewBase
class COutlineCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        max_completion_tokens=128000,
        max_tokens=128000,
        temperature=0.5
    )

    @after_kickoff
    def final_formatting(self, result):
        conclusion_outline = next((
            task_output.json_dict for task_output in result.tasks_output 
            if task_output.name == 'structure_conclusion_outline'
        ), {})
        return conclusion_outline

    @agent
    def report_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['report_analyst'],
            llm=self.std_llm,
        )

    @agent
    def conclusion_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['conclusion_outliner'],
            llm=self.std_llm,
        )
    
    @task
    def analyze_report_for_conclusion(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_report_for_conclusion'],
        )
    
    @task
    def structure_conclusion_outline(self) -> Task:
        return Task(
            config=self.tasks_config['structure_conclusion_outline'],
            output_json=ConclusionSectionOutline
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