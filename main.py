from graph import ResearchAgentGraph

initial_state = {
    "query": "Research on Apple Macbook",
    "research_plan": [],
    "notes": [],
    "iteration": 0,
    "max_iterations": 3
}

research_agent_graph = ResearchAgentGraph()
response = research_agent_graph.app.invoke(initial_state)
print(response['answer'])