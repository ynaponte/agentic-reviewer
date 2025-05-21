from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, after_kickoff
from crewai.llm import LLM
from .pydantic_output.pydantic_output import RDSubSectionOutline


@CrewBase
class RDOutlineCrew:

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
        res_and_disc_outlines = [
            task_output.json_dict for task_output in result.tasks_output 
            if task_output.name in ['structure_results_outline', 'structure_discussion_outline']
        ]
        return res_and_disc_outlines
        

    @agent
    def report_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['report_analyst'],
            llm=self.std_llm,
        )

    @agent
    def results_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['results_outliner'],
            llm=self.std_llm,
        )
    
    @agent
    def discussion_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['discussion_outliner'],
            llm=self.std_llm,
        )
    
    @task
    def analyze_report_for_results(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_report_for_results'],
        )
    
    @task
    def structure_results_outline(self) -> Task:
        return Task(
            config=self.tasks_config['structure_results_outline'],
            output_json=RDSubSectionOutline
        )

    @task
    def analyze_report_for_discussion(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_report_for_discussion'],
        )
    
    @task
    def structure_discussion_outline(self) -> Task:
        return Task(
            config=self.tasks_config['structure_discussion_outline'],
            output_json=RDSubSectionOutline
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