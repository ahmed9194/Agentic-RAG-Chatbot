from crewai import Crew, Agent

retriever = Agent(name="Retriever", role="fetch relevant hotel chunks")
summarizer = Agent(name="Summarizer", role="remove redundancy and summarize chunks")
composer = Agent(name="Composer", role="compose final response using LLM")

crew = Crew(agents=[retriever, summarizer, composer], goal="Respond to travel queries")
