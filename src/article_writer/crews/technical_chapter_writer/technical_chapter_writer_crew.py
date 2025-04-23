from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Union


class Insights(BaseModel):
    insights: List[str] = Field(description="List of detailed insights about the researched item")


@CrewBase
class TechnicalChapterWriterCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    manager_llm = LLM(
        model="ollama/deepseek-r1:7b",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.5,
        reasoning_effort='medium'
    )

    researcher_and_editor_llm = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.4
    )

    func_caller = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.2
    )

    writer_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.7
    )

    @agent
    def chapter_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['chapter_manager'],
            allow_delegation=True,
            verbose=True,
            llm=self.manager_llm
        )

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['topic_researcher'],
            llm=self.researcher_and_editor_llm,
            tools=[QueryArticlesTool()],
            verbose=True
        )

    @agent
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer'],
            llm=self.writer_llm,
            verbose=True
        )

    @agent
    def technical_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_editor'],
            llm=self.researcher_and_editor_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            tools=[QueryArticlesTool()],
            output_json=Insights
        )
    
    @task
    def write_section(self) -> Task:
        return Task(
            config=self.tasks_config['write_section'],
        )
    
    @task
    def final_editing(self) -> Task:
        return Task(
            config=self.tasks_config['final_editing'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            #manager_llm=self.manager_llm,
            agents=[
                self.topic_researcher(),
            ],
            tasks=[
                self.research_task(),
            ],
            process=Process.sequential,
            verbose=True,
            planning=False
        )
