from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool


@CrewBase
class ResultAndDiscussionCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llm = LLM(
        model="ollama/qwen2.5:14b",
        base_url="http://localhost:11434",
		max_tokens=128000,
		temperature=0.3
    )

    tool_calling_llm = LLM(
        model='ollama/qwen2.5:14b',
        base_url='http://localhost:11434',
        max_tokens=128000,
        temperature=0.2
    )
	
    @agent
    def results_discussion_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['results_discussion_writer'],
            llm=self.llm,
            function_calling_llm=self.tool_calling_llm,
            tools=[QueryArticlesTool()]
        )
    
    @task
    def contextualization(self) -> Task:
        return Task(
            config=self.tasks_config['contextualization'],
            tools=[QueryArticlesTool()]
        )
    
    @task
    def results_discussion_writting(self) -> Task:
        return Task(
            config=self.tasks_config['results_discussion_writting']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
    