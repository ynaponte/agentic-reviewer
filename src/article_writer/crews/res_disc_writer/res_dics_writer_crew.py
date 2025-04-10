from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import QueryArticlesTool


@CrewBase
class ResultAndDiscussionCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    writer_llm = LLM(
        model="ollama/qwen2.5:14b-instruct-q8_0",
        base_url="http://localhost:11434",
		max_tokens=128000,
		temperature=0.7
    )

    researcher_llm = LLM(
        model='ollama/qwen2.5:14b',
        base_url='http://localhost:11434',
        max_tokens=128000,
        temperature=0.2
    )

    tool_call_llm = LLM(
        model='ollama/qwen2.5:14b',
        base_url='http://localhost:11434',
        max_tokens=128000,
        temperature=0.2
    )
	
    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            llm=self.researcher_llm,
            tools=[QueryArticlesTool()]

    @agent
    def results_discussion_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['results_discussion_writer'],
            llm=self.writer_llm,
            function_calling_llm=self.tool_call_llm,
            tools=[QueryArticlesTool()]
        )

    @task
    def select_complementary_elements(self) -> Task:
        return Task(
            config=self.tasks_config['select_complementary_elements'],
        )
    
    @task
    def results_discussion_writting(self) -> Task:
        return Task(
            config=self.tasks_config['results_discussion_writting'],
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
    