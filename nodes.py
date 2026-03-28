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

from helpers import extract_json, search_tool
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
            "search_query": research_plan[0],
            "iteration": iteration + 1,
            "decision": Decision.SEARCH.value,
        }

    # -------- RESEARCH ITERATION --------
    request = {
        "request_name": "research-iteration",
        "query": query,
        "research_plan": research_plan,
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

    search_query = research_plan[iteration - 1] if iteration - 1 < len(research_plan) else None
    if not search_query:
        return {"decision": Decision.SYNTHESIZE.value, "iteration": iteration + 1}

    return {
        "search_query": search_query,
        "iteration":    iteration + 1,
        "decision":     Decision.SEARCH.value,
    }

def search_node(state: ResearchState):

    search_query = state.get("search_query")
    
    if not search_query:
        raise ValueError("Missing search_query in state")
    
    # Search
    try:
        results = search_tool(search_query)
    except Exception as e:
        raise RuntimeError(f"Search tool failed for query '{search_query}': {e}") from e
    
    # Handle empty results
    if not results:
        return {"notes": state.get("notes", [])}
    
    # Format results for LLM
    content = "\n".join([r["snippet"] for r in results if r.get("snippet")])

    if not content.strip():
        return {"notes": state.get("notes", [])}

    request = {
        "request_name": "extract-info",
        "query": state["query"],
        "search_query": search_query,
        "content": content
    }

    try:
        extracted = service.get_llm_response(request)
    except Exception as e:
        raise RuntimeError(f"LLM call failed during extract-info: {e}") from e
    
    # Store query + extracted together for tracebility
    note = {
        "search_query": search_query,
        "extracted": extracted
    }

    notes = state.get("notes", [])
    notes.append(note)
    return {"notes": notes}

def synthesizer_node(state: ResearchState):
    query = state["query"]
    notes = state.get("notes", [])
    research_plan = state.get("research_plan", [])

    if not notes:
        return ValueError("No notes to synthesize")
    
    # Format notes for LLM
    formatted_notes = "\n\n".join([
        f"Q: {note['search_query']}\nA: {note['extracted']}"
        for note in notes
    ])

    request = {
        "request_name": "synthesize",
        "query": query,
        "research_plan": research_plan,
        "notes": formatted_notes,
    }

    try:
        raw = service.get_llm_response(request)
    except Exception as e:
        raise RuntimeError(f"LLM call failed during synthesize: {e}") from e
    
    response = extract_json(raw)

    answer = response.get("answer")
    if not answer:
        raise ValueError(f"Missing 'answer' in LLM response: {response}")
    
    return {
        "answer": answer,
        "decision": "done"
    }