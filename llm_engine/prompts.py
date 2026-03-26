from langchain.prompts import PromptTemplate

RESEARCH_PLAN_PROMPT = """
You are supervising a research agent.

User question:
{query}

Break this into 3-5 clear research questions.

IMPORTANT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include <think> tags
- Do NOT include any text outside JSON

Output format:
{
  "plan": ["q1", "q2", "q3"]
}
"""

RESEARCH_ITERATION_PROMPT = """
You are supervising a research agent.

User Question:
{query}

Research Plan:
{plan}

Notes Collected So Far:
{notes}

Current Iteration:
{iteration}

Decide if more research is needed.

If more research is needed:
generate a new search query.

If the research is sufficient:
return decision = "synthesize".

Return JSON:

{{
 "decision": "search" or "synthesize",
 "search_query": "optional"
}}
"""


SYSTEM_PROMPT = """
You are a research planning assistant.

Your job is to convert a user question into 3-5 clear research questions.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include extra text
- Do NOT include <think> or reasoning
- The response must be directly parseable JSON

Output format:
{
  "plan": ["q1", "q2", "q3"]
}
"""

USER_PROMPT = """
User question:
{0}
"""

class AgentPrompts:
    def research_plan_prompt(self):
        research_prompt = PromptTemplate(template=RESEARCH_PLAN_PROMPT, input_variables=["query"])
        return research_prompt
    
    def research_plan_prompt_groq(self, query):
        # research_prompt = PromptTemplate(template=RESEARCH_PLAN_PROMPT, input_variables=["query"])
        # return research_prompt.format(query=query)
        user_prompt = USER_PROMPT.format(query)
        return SYSTEM_PROMPT, user_prompt
    
    def research_iteration_prompt(self):
        research_iteration_prompt = PromptTemplate(template=RESEARCH_ITERATION_PROMPT, input_variables=["query", "plan", "notes", "iteration"])
        return research_iteration_prompt
