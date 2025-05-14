from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from pydantic import BaseModel, Field
import json


class MergedTextOutput(BaseModel):
  subsection_title: str = Field(
    description="Exact title of the subsection of which the raw text belongs"
  )
  subsection_text: str = Field(
    description="Fluid, continuous and restructured text of the subsection, in academic Brazilian Portuguese."
  )


@CrewBase
class FinalEditingCrew:

  agents_config = "config/agents.yaml"
  tasks_config = "config/tasks.yaml"

  merger_llm = LLM(
    model="ollama/qwen2.5:7b-instruct",
    base_url="http://localhost:11434",
    timeout=1800.0,
    temperature=0.4,
    max_tokens=128000
  )
  subsection_specific_directives = {
    'Results': (
      '- Present experimental findings clearly and concisely, prioritizing precise references to numerical data and visual elements '
      '(e.g., "Tabela 2", "Figura 3").\n'
      '- Structure the text to emphasize a factual, data-driven narrative, organizing findings to highlight results and their supporting '
      'evidence first.\n'
      '- Ensure all numerical results and visual references from the outline are integrated accurately, preserving their role in validating '
      'the findings.\n'
    ),
    'Discussion': (
      '- Integrate comparisons with prior studies, limitations, contributions, and future directions into a cohesive interpretive narrative, '
      'preserving specific details like study titles and results.\n'
      '- Connect analytical discussions to broader scientific advancements (e.g., implications for optical logic or neural network optimization)'
      ', ensuring contextual relevance.\n'
      '- Ensure visual and numerical evidence supports interpretive arguments, with clear and purposeful references to tables or figures.\n'
    )
  }

  @before_kickoff
  def spec_agent_from_inputs(self, inputs):
    ss_spec = self.subsection_specific_directives.get(inputs['subsection_title'])
    inputs['subsection_specific_directives'] = ss_spec
    inputs['topics_text_json'] = json.dumps(inputs['topics_text_json'], indent=2)
    inputs['subsection_outline'] = json.dumps(inputs['subsection_outline'], indent=2)
    return inputs

  @agent
  def subsection_merger(self) -> Agent:
    return Agent(
      config=self.agents_config['subsection_merger'],
      llm=self.merger_llm
    )
    
  @task
  def merge_subsection(self) -> Task:
    return Task(
      config=self.tasks_config['merge_subsection'],
      output_json=MergedTextOutput
    )
  
  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=self.agents,
      tasks=self.tasks,
      process=Process.sequential,
      verbose=True
    )