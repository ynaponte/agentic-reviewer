from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from pydantic import BaseModel, Field
from src.tools import QueryArticlesTool


class TopicTextContent(BaseModel):
    topic: str = Field(description="Exact name of the topic that writing was requested upon")
    text: str = Field(
        description=(
            "Full multi-paragraph scientific text about the topic, with 1000+ words, "
            "written in brazilian portuguese"
        )
    )


@CrewBase
class ConclusionTopicRagCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    researcher_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.4
    )
    writer_llm = LLM(
        model="ollama/qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.6
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
    def conclusion_writer(self) -> Agent:
      return Agent(
        config=self.agents_config['conclusion_writer'],
        llm=self.writer_llm,
        verbose=True,
      )
    
    @task
    def conclusion_topic_reference_research(self) -> Task:
      return Task(
        config=self.tasks_config['conclusion_topic_reference_research'],
        async_execution=False
      )
    
    @task
    def conclusion_topic_report_draft_research(self) -> Task:
      return Task(
        config=self.tasks_config['conclusion_topic_report_draft_research'],
        async_execution=False
      )
    
    @task
    def write_conclusion_topic_text(self) -> Task:
      return Task(
        config=self.tasks_config['write_conclusion_topic_text'],
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