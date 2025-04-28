from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field


class MergedTextOutput(BaseModel):
  subsection_title: str = Field(
    description="Exact title of the subsection of which the raw text belongs"
  )
  subsection_text: str = Field(
    description="Fluid, continuous and restructured text of the subsection, in academic Brazilian Portuguese."
  )


@CrewBase
class ContentMergerCrew:
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'
  merger_llm = LLM(
    model="ollama/qwen2.5:7b:instruct",
    base_url="http://localhost:11434",
    timeout=1800.0,
    temperature=0.4,
    max_tokens=128000
  )

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