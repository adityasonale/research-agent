from langgraph.graph import END, StateGraph
from states import ResearchState
from nodes import *

class ResearchAgentGraph:

    def __init__(self):
        workflow = StateGraph(ResearchState)

        # Add Node
        workflow.add_node("supervisor", supervisor_node)

        # Entry Point
        workflow.set_entry_point("supervisor")

        # Direct to End
        workflow.add_edge("supervisor", END)

        self.app = workflow.compile()