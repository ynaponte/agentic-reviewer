from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from src.tools import FetchMetadataTool, FetchArticlesTool, QueryArticlesTool


@CrewBase
class ResultAndDiscussionCrew:
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    analyst_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8192,
        max_tokens=131072,
        temperature=0.2
    )

    researcher_llm = LLM(
        model="ollama/deepseek-r1:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8192,
        max_tokens=131072,
        temperature=0.7
    )

    writer_llm = LLM(
        model="ollama/qwen2.5:32b",
        base_url="http://localhost:11434",
        max_completion_tokens=8192,
        max_tokens=131072,
        temperature=0.7
    )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            llm=self.analyst_llm,
            tools=[FetchArticlesTool()]
        )
    
    #@agent
    #def researcher(self) -> Agent:
    #    return Agent(
    #        config=self.agents_config['researcher'],
    #        llm=self.researcher_llm,
    #        tools=[QueryArticlesTool()]
    #    )
#
    #@agent
    #def writer(self) -> Agent:
    #    return Agent(
    #        config=self.agents_config['writer'],
    #        llm=self.writer_llm,
    #    )

    @task
    def initial_assessment(self) -> Task:
        return Task(
            config=self.tasks_config['initial_assessment']
        )
    
    #@task
    #def query_execution(self) -> Task:
    #    return Task(
    #        config=self.tasks_config['query_execution'],
    #        tools=[QueryArticlesTool()]
    #    )
    #
    #@task
    #def generate_outline(self) -> Task:
    #    return Task(
    #        config=self.tasks_config['generate_outline'],
    #    )
    #
    #@task
    #def compose_section(self) -> Task:
    #    return Task(
    #        config=self.tasks_config['compose_section'],
    #    )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )