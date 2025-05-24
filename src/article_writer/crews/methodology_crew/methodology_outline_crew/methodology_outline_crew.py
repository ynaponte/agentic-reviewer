from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, after_kickoff, before_kickoff
from crewai.llm import LLM
from .pydantic_output.pydantic_output import MethodologySectionOutline
import json

@CrewBase
class MethodologyOutlineCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    std_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        max_completion_tokens=128000,
        max_tokens=128000,
        temperature=0.3
    )
    
    @before_kickoff
    def input_formatting(self, inputs):
        inputs['generated_sections_content'] = json.dumps(inputs['generated_sections_content'], indent=2)
        return inputs

    @after_kickoff
    def final_formatting(self, result):
        methodology_outline = next((
            task_output.json_dict for task_output in result.tasks_output 
            if task_output.name == 'expand_methodology_subsections'
        ), {})
        return methodology_outline

    @agent
    def report_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['report_analyst'],
            llm=self.std_llm,
        )

    @agent
    def methodology_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['methodology_outliner'],
            llm=self.std_llm,
        )
    # ------------ Tarefas ------------
    @task
    def analyze_report_for_methodology_components(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_report_for_methodology_components'],
        )
    
    @task
    def analyze_sections_for_methodological_context(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_sections_for_methodological_context'],
        )
    
    @task
    def define_methodology_subsections(self) -> Task:
        return Task(
            config=self.tasks_config['define_methodology_subsections'],
            output_json=MethodologySectionOutline
        )

    @task
    def expand_methodology_subsections(self) -> Task:
        return Task(
            config=self.tasks_config['expand_methodology_subsections'],
            output_json=MethodologySectionOutline
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