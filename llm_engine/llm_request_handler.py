from llm_engine.chains import research_plan_chain, research_iteration_chain
from llm_engine.prompts import AgentPrompts
from typing import Dict, Any
from settings import LLM_SERVICE


def handle_query_requests(request: Dict[str, Any], model) -> Dict[str, Any]:
    request_name = request.get('request_name')
    response = ""
    if request_name == 'research-plan':
        query = request.get("query", "")

        if LLM_SERVICE == "local":
            chain = research_plan_chain(model)
            response = chain.invoke({"query": query})
        
        elif LLM_SERVICE == "groq":
            prompts = AgentPrompts()
            system, user = prompts.research_plan_prompt_groq(query)
            data = model.fetch_response(system, user)
            response = data['choices'][0]['message']['content'] or ''

    elif request_name == "research-iteration":
        query = request.get("query", "")
        research_plan = request.get("research_plan", "")
        notes = request.get("notes", "")
        iteration = request.get("iteration", "")
    
        if LLM_SERVICE == "local":
            chain = research_iteration_chain(model)
            response = chain.invoke({"query":query, "research_plan": research_plan, "notes": notes, "iteration": iteration})

        elif LLM_SERVICE == "groq":
            prompts = AgentPrompts()
            system, user= prompts.research_iteration_prompt_groq(query, research_plan, notes, iteration)
            data = model.fetch_response(system, user)
            response = data['choices'][0]['message']['content'] or ''

    elif request_name == 'extract-info':
        query = request.get("query", "")
        search_query = request.get("search_query", "")
        content = request.get("content", "")

        if LLM_SERVICE == "local":
            pass

        elif LLM_SERVICE == "groq":
            prompts = AgentPrompts()
            system, user = prompts.extract_information_prompt_groq(query, search_query, content)
            data = model.fetch_response(system, user)
            response = data['choices'][0]['message']['content'] or ''

    elif request_name == 'synthesize':
        query = request.get("query", "")
        research_plan = request.get("research_plan", "")
        notes = request.get("notes", "")

        if LLM_SERVICE == "local":
            pass

        elif LLM_SERVICE == "groq":
            prompts = AgentPrompts()
            system, user = prompts.synthesize_prompt_groq(query, research_plan, notes)
            data = model.fetch_response(system, user)
            response = data['choices'][0]['message']['content'] or ''

    return response