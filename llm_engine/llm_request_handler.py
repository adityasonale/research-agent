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
        if LLM_SERVICE == "local":
            query = request.get("query", "")
            plan = request.get("plan", "")
            notes = request.get("notes", "")
            iteration = request.get("iteration", "")

            chain = research_iteration_chain(model)
            response = chain.invoke({"query":query, "plan": plan, "notes": notes, "iteration": iteration})

    return response