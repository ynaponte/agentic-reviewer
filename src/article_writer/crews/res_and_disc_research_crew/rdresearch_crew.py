from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.tasks.conditional_task import ConditionalTask
from src.tools import QueryArticlesTool
from pydantic import BaseModel, Field
from typing import List


class ResearchOutput(BaseModel):
    item_name: str = Field(description="Name of the researched item")
    research_results: str = Field(description="The research results about the item")


class TopicTextContent(BaseModel):
    topic: str = Field(description="Exact name of the topic that writing was requested upon")
    text: str = Field(
        description=(
            "Full multi-paragraph scientific text about the topic, with 1000+ words, "
            "written in brazilian portuguese"
        )
    )


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
    researcher_llm = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.4
    )
    writer_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=128000,
        temperature=0.6
    )
    _should_execute_ve_research = True
    _should_execute_nr_research = True

    @before_kickoff
    def check_inpus(self, inputs: dict):
        # Checa se existem elementos visuais e resultados numéricos a serem pesquisados.
        # Caso não existão, previne a execução das respectivas tarefas
        self._should_execute_ve_research = inputs.get('visual_elements_to_contextualize', '') != ''
        self._should_execute_nr_research = inputs.get('numerical_results_to_include', '') != ''
        return inputs

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
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer'],
            llm=self.writer_llm,
            verbose=True,
        )
    
    @task
    def topic_research(self) -> Task:
        return Task(
            config=self.tasks_config['topic_research'],
            async_execution=False
        )

    @task
    def visual_elements_research(self) -> Task:
        return ConditionalTask(
            config=self.tasks_config['visual_elements_research'],
            async_execution=False,
            condition=lambda x: self._should_execute_ve_research  # Tem que ser 'callable'
        )

    @task
    def numerical_results_research(self) -> Task:
        return ConditionalTask(
            config=self.tasks_config['numerical_results_research'],
            async_execution=False,
            condition=lambda x: self._should_execute_nr_research  # Tem que ser 'callable'
        )

    @task
    def write_topic_text(self) -> Task:
        return Task(
            config=self.tasks_config['write_topic_text'],
            output_json=TopicTextContent
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=False
        )
