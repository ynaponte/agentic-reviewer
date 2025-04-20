from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool


@CrewBase
class TechnicalChapterWriterCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    planner = LLM(
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

    writer_llm = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.7
    )

    @agent
    def chapter_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['chapter_manager'],
            llm=self.researcher_and_editor_llm,  # Trocar depois para o deepseek ou QWQ
            allow_delegation=True,
        )

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['topic_researcher'],
            llm=self.researcher_and_editor_llm,
            tools=[QueryArticlesTool()]
        )

    @agent
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer'],
            llm=self.writer_llm,
        )

    @agent
    def technical_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_editor'],
            llm=self.researcher_and_editor_llm,
        )

    @task
    def research_chapter_outline(self) -> Task:
        return Task(
            config=self.tasks_config['research_chapter_outline'],
            tools=[QueryArticlesTool()]
        )

    @task
    def write_and_edit_chapter(self) -> Task:
        return Task(
            config=self.tasks_config['write_and_edit_chapter']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            manager_agent=self.chapter_manager(),
            manager_llm=self.researcher_and_editor_llm,
            agents=[
                self.topic_researcher(),
                self.technical_writer(),
                self.technical_editor()
            ],
            tasks=[self.research_chapter_outline(
            ), self.write_and_edit_chapter()],
            process=Process.hierarchical,
            verbose=True,
        )
