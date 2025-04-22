from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool


@CrewBase
class TechnicalChapterWriterCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    manager_llm = LLM(
        model="ollama/deepseek-r1:32b",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.5,
        reasoning_effort='medium'
    )

    researcher_and_editor_llm = LLM(
        model="ollama/qwen2.5:32b",
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
    def write_technical_chapter(self) -> Task:
        return Task(
            config=self.tasks_config['write_technical_chapter'],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            tools=[QueryArticlesTool()]
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
            manager_agent=self.chapter_manager(),
            agents=[
                self.topic_researcher(),
                self.technical_writer(),
                self.technical_editor()
            ],
            tasks=[
                self.write_technical_chapter(),
                self.research_task(),
                self.write_section(),
                self.final_editing()
            ],
            process=Process.hierarchical,
            verbose=True,
            planning=True,
            planning_llm=self.manager_llm
        )
