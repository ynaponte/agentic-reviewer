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

    func_caller = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
        max_tokens=32000,
        temperature=0.2
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
            verbose=True
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
    def research_and_write_sections(self) -> Task:
        return Task(
            config=self.tasks_config['research_and_write_sections'],
        )
    
    @task
    def assemble_and_finalize_chapter(self) -> Task:
        return Task(
            config=self.tasks_config['assemble_and_finalize_chapter'],
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
            function_calling_llm=self.func_caller,
            tasks=[self.research_and_write_sections(), self.assemble_and_finalize_chapter()],
            process=Process.hierarchical,
            verbose=True,
            planning=False,
            #planning_llm=self.planner
        )
