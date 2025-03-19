from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

@CrewBase
class AyurvedicCrew:
    """Ayurvedic crew"""
    llm = "vertex_ai/gemini-2.0-flash-001"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def fact_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['fact_finder'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True
        )
    
    @agent
    def advisory_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['advisory_analyst'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )
    
    @task
    def fact_finder_task(self) -> Task:
        return Task(
            config=self.tasks_config['fact_verification_task'],
        )
    
    @task
    def advisory_task(self) -> Task:
        return Task(
            config=self.tasks_config['advisory_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Ayurvedic crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )