from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool


@CrewBase
class TechnicalChapterWriterCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    writer_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        timeout=1800.0,
		max_tokens=128000,
		temperature=0.7
    )

    tool_call_llm = LLM(
        model='ollama/qwen2.5:14b',
        base_url='http://localhost:11434',
        max_tokens=128000,
        temperature=0.2
    )
	
    @agent
    def technical_chapter_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_chapter_writer'],
            llm=self.writer_llm,
            memory=True,
            tools=[QueryArticlesTool()]
        )
    
    @task
    def write_technical_chapter(self) -> Task:
        return Task(
            config=self.tasks_config['write_technical_chapter'],
            tools=[QueryArticlesTool()]
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
    