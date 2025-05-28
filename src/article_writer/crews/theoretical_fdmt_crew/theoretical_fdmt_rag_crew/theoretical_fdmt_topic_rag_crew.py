from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from pydantic import BaseModel, Field
from src.tools import QueryArticlesTool


class TopicTextContent(BaseModel):
    topic: str = Field(description="Exact name of the topic that writing was requested upon")
    text: str = Field(
        description=(
            "Full multi-paragraph scientific text about the topic, with 200-500 words, "
            "written in brazilian portuguese"
        )
    )


@CrewBase
class TheoreticalFdmtTopicRagCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    researcher_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.3
    )
    writer_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.6
    )

    @agent
    def theoretical_foundation_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['theoretical_foundation_researcher'],
            llm=self.researcher_llm,
            tools=[QueryArticlesTool()],
            verbose=True,
            memory=True
        )

    @agent
    def theoretical_foundation_writer(self) -> Agent:
      return Agent(
        config=self.agents_config['theoretical_foundation_writer'],
        llm=self.writer_llm,
        verbose=True,
      )
    
    @task
    def theoretical_topic_external_literature_research(self) -> Task:
      return Task(
        config=self.tasks_config['theoretical_topic_external_literature_research'],
        async_execution=False
      )
    
    @task
    def theoretical_topic_internal_context_research(self) -> Task:
      return Task(
        config=self.tasks_config['theoretical_topic_internal_context_research'],
        async_execution=False
      )
    
    @task
    def write_theoretical_foundation_topic_text(self) -> Task:
      return Task(
        config=self.tasks_config['write_theoretical_foundation_topic_text'],
        output_json=TopicTextContent
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