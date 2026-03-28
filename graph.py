from langgraph.graph import END, StateGraph
from states import ResearchState
from nodes import *

class ResearchAgentGraph:

    def __init__(self):
        workflow = StateGraph(ResearchState)

        # Add Node
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("search", search_node)
        workflow.add_node("synthesizer", synthesizer_node)

        # Entry Point
        workflow.set_entry_point("supervisor")

        # Conditional routing of supervisor node
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["decision"],
            {
                Decision.SEARCH.value: "search",
                Decision.SYNTHESIZE.value: "synthesizer"
            }
        )

        # After search, always return to supervisor
        workflow.add_edge("search", "supervisor")

        # Direct to End
        workflow.add_edge("supervisor", END)

        self.app = workflow.compile()