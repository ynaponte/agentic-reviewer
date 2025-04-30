from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.article_writer.types.results_report import ElementsList
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Union


class VisualElements(BaseModel):
    name: str = Field(description="Name of the visual element")
    description: str = Field(description="Description of the visual element")    

class DiscussionTopic(BaseModel):
    topic: str = Field(description="Description of the topic to be discussed")
    visual_elements_to_contextualize: Optional[List[VisualElements]] = Field(
        default=[],
        description="Visual elements that are important to the topic and should be contextualized in the section's texts"
    )
    numerical_results: Optional[List[str]] = Field(
        default=[],
        description="Numerical results that are important to the topic"
    )

class SubSection(BaseModel):
    section_title: str = Field(
        description="Title of the subsection"
    )
    discussion_topics: List[DiscussionTopic] = Field(
        description="List of topics that the text content of the subsection should discuss"
    )


class Outline(BaseModel):
    section_name: str = Field(description="Name of section")
    discussion_topics: List[DiscussionTopic] = Field(description="Topics to discuss in the section's introduction text")
    subsections: List[SubSection] = Field(
        description="Subsections that the main section contains"
    )


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
    def results_discussion_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['results_discussion_outliner'],
            llm=self.std_llm,
        )

    @agent
    def conclusion_outliner(self) -> Agent:
        return Agent(
            config=self.agents_config['core_chapters_outliner'],
            llm=self.std_llm,
        )

    @task
    def generate_outline_results_discussion(self) -> Task:
        return Task(
            config=self.tasks_config['generate_outline_results_discussion'],
            output_json=Outline
        )
    
    @task
    def generate_conclusion_outline(self) -> Task:
        return Task(
            config=self.tasks_config['generate_conclusion_outline'],
            output_json=Outline
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