from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from src.tools import FetchArticlesTool, FetchMetadataTool


class ReportCrew(CrewBase):
    """Equipe responsável por gerar relatórios de artigos ciêntíficos"""

    agents_config = "src/config/agents.yaml"
    tasks_config = "src/config/tasks.yaml"

    @agent
    def coordinator(self) -> Agent:
        """Agente responsável por coordenar a equipe de criação do relatório
        e responder perguntas gerais
        """
        return Agent(
            config=self.agents_config["coordinator"],
            tools=[FetchArticlesTool(), FetchMetadataTool()],
            verbose=True
        )
    
    @agent
    def metodology_and_insights(self) -> Agent:
        """Agente responsável por criar a metodologia e insights dos artigos"""
        return Agent(
            config=self.agents_config["methodology_and_insights"],
            verbose=True
        )
