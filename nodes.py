# List of Nodes
# supervisor_node = planning + routes
# web_search_node = execute search
# knowledge_node = distill findings
# should_continue = conditional edge
# synthesis_node

# supervisor node
# Create research plan
# generate first search query

import re
import warnings
from enum import Enum
import json

from helpers import extract_json
from states import ResearchState
from llm_engine.llm_service import LLMServices


class Decision(Enum):
    SEARCH = "search"
    SYNTHESIZE = "synthesize"


service = LLMServices()

DEFAULT_MAX_ITERATIONS = 3

def supervisor_node(state: ResearchState):
    """
    Three-branch state machine:
      1. Max iterations reached  → synthesize
      2. No plan yet (first run) → build plan, go to search
      3. Plan exists             → evaluate progress, search or synthesize
    """
    query = state["query"]
    research_plan = state["research_plan"]
    notes = state["notes"]
    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # -------- MAX ITERATIONS REACHED --------
    if iteration >= max_iterations:
        return {"decision": Decision.SYNTHESIZE.value}

    # -------- FIRST ITERATION --------
    if not research_plan:
        request = {
            "request_name": "research-plan",
            "query": query
        }
        try:
            raw = service.get_llm_response(request)
        except Exception as e:
            raise RuntimeError(f"LLM call failed during research-plan: {e}") from e

        response = extract_json(raw)

        research_plan = response.get("plan")

        if not research_plan:
            raise ValueError(f"Missing 'plan' or 'search_plan' in LLM response: {response}")

        return {
            "research_plan": research_plan,
            "iteration": iteration + 1,
            "decision": Decision.SEARCH.value,
        }

    # -------- RESEARCH ITERATION --------
    request = {
        "request_name": "research-iteration",
        "query": query,
        "plan": research_plan,
        "notes": notes,
        "iteration": iteration,
    }
    try:
        raw = service.get_llm_response(request)
    except Exception as e:
        raise RuntimeError(f"LLM call failed during research-iteration: {e}") from e
    
    response = extract_json(raw)

    decision = response.get("decision")
    if decision is None:
        warnings.warn(
            f"LLM response missing 'decision' key, defaulting to SYNTHESIZE. "
            f"Response: {response}",
            RuntimeWarning,
        )
        decision = Decision.SYNTHESIZE.value

    if decision == Decision.SYNTHESIZE.value:
        return {
            "decision":  Decision.SYNTHESIZE.value,
            "iteration": iteration + 1,
        }

    search_query = response.get("search_query")
    if not search_query:
        raise ValueError(f"Missing 'search_query' in LLM response: {response}")

    return {
        "search_query": search_query,
        "iteration":    iteration + 1,
        "decision":     Decision.SEARCH.value,
    }