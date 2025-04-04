from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class OutlineCrew():

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	llm = LLM(
        model="ollama/qwen2.5:14b",
        base_url="http://localhost:11434",
		max_tokens=128000,
		temperature=0.3
    )

	@agent
	def chapter_outliner(self) -> Agent:
		return Agent(
			config=self.agents_config['chapter_outliner'],
			llm=self.llm,
			verbose=True
		)
	
	@task
	def generate_results_discussion_outline(self) -> Task:
		return Task(
			config=self.tasks_config['generate_results_discussion_outline'],
		)
	
	@task
	def generate_conclusion_outline(self) -> Task:
		return Task(
			config=self.tasks_config['generate_conclusion_outline'],
		)
	
	@task
	def generate_methodology_outline(self) -> Task:
		return Task(
			config=self.tasks_config['generate_methodology_outline'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ConclusionCrew crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
