from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from src.tools import QueryArticlesTool
from typing import Dict, Any
import yaml


class TechnicalChapterWriterCrew():

    agents_config_path = 'src/article_writer/crews/technical_chapter_writer/config/agents.yaml'
    tasks_config_path = 'src/article_writer/crews/technical_chapter_writer/config/tasks.yaml'

    researcher_and_editor_llm = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
		max_tokens=128000,
		temperature=0.4
    )

    writer_llm = LLM(
        model="ollama/qwen2.5:3b-instruct-q6_K",
        base_url="http://localhost:11434",
        timeout=1800.0,
		max_tokens=128000,
		temperature=0.7
    )

    def __init__(self):
        with open(self.agents_config_path, 'r') as a_cfg_file:
            self.agents_config = yaml.safe_load(a_cfg_file)

        with open(self.tasks_config_path, 'r') as t_cfg_file:
            self.tasks_config = yaml.safe_load(t_cfg_file)

        self.crew = self._setup_crew()

    def _setup_crew(self) -> Crew:
        return Crew (
            manager_agent=self.chapter_manager(),
            agents=[self.technical_editor(), self.topic_researcher(), self.technical_writer()],
            tasks=self.write_technical_chapter(),
            process=Process.hierarchical,
            verbose=True,
            memory=True
        )
    
    def kickoff(self, inputs: Dict[str, Any]):
        return self.crew.kickoff(inputs=inputs)

    def chapter_manager(self) -> Agent:
        return Agent(
            config=self.agents_config_path['chapter_manager'],
            llm=self.researcher_and_editor_llm,  # Trocar depois para o deepseek ou QWQ
            allow_delegation=True,
        )

    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config_path['topic_researcher'],
            llm=self.researcher_and_editor_llm,
            tools=[QueryArticlesTool()]
        )
    
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config_path['technical_writer'],
            llm=self.writer_llm,
        )
    
    def technical_editor(self) -> Agent:
        return Agent(
            config=self.agents_config_path['technical_editor'],
            llm=self.researcher_and_editor_llm,
        )
    
    def write_technical_chapter(self) -> Task:
        return Task(
            config=self.tasks_config['write_technical_chapter'],
        )
    