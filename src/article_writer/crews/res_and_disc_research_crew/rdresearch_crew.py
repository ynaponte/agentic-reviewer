from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool
from pydantic import BaseModel, Field
from typing import List


class ResearchOutput(BaseModel):
    item_name: str = Field(description="Name of the researched item")
    research_results: str = Field(
        default_factory=str,
        description="The research results about the item"
    )


class Insights(BaseModel):
    topic: str = Field(description="Name of the topic")
    insights: str = Field(description="Text with the detailed insights about the topic")


class VisualElementsResearchOutput(BaseModel):
    about_the_visual_elements: List[ResearchOutput]= Field(
        default_factory=list,
        description="List of researched visual elements and their respective research results"
    )


class NumericalResultsResearchOutput(BaseModel):
    about_the_numerical_results: List[ResearchOutput] = Field(
        default_factory=list,
        description="List of researched numerical results and their respective research results"
    )


class TopicResearchOutput(BaseModel):
    about_the_topic: ResearchOutput = Field(
        default_factory=list,
        description="Research results about the topic"
    )


@CrewBase
class RDResearchCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    manager_llm = LLM(
        model="ollama/deepseek-r1:7b",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=128000,
        temperature=0.4
    )

    researcher_llm = LLM(
        model="ollama/qwen2.5:14b-instruct-q8_0",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=128000,
        temperature=0.4
    )

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['topic_researcher'],
            llm=self.researcher_llm,
            tools=[QueryArticlesTool()],
            verbose=True,
            memory=True
        )
    
    @agent
    def insight_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['insight_writer'],
            llm=self.researcher_llm,
            verbose=True,
        )
    
    @task
    def topic_research(self) -> Task:
        return Task(
            config=self.tasks_config['topic_research'],
            output_json=TopicResearchOutput
        )

    @task
    def visual_elements_research(self) -> Task:
        return Task(
            config=self.tasks_config['visual_elements_research'],
            output_json=VisualElementsResearchOutput
        )

    @task
    def numerical_results_research(self) -> Task:
        return Task(
            config=self.tasks_config['numerical_results_research'],
            output_json=NumericalResultsResearchOutput
        )

    @task
    def gather_insights(self) -> Task:
        return Task(
            config=self.tasks_config['gather_insights'],
            output_json=Insights
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            #manager_llm=self.manager_llm,
            agents=[
                self.topic_researcher(),
                self.insight_writer()
            ],
            tasks=[
                self.topic_research(),
                self.visual_elements_research(),
                self.numerical_results_research(),
                self.gather_insights(),
            ],
            process=Process.sequential,
            verbose=True,
            planning=False
        )
