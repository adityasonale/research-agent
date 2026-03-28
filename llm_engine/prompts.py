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

# =================== PLANNING PROMPT =========================

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

# ======================= ITERATION ======================

SYSTEM_PROMPT_ITERATION = """
You are a research progress evaluator.

Your job is to evaluate the research collected so far and decide whether more research is needed or the findings are sufficient to synthesize a final answer.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include extra text
- Do NOT include <think> or reasoning
- The response must be directly parseable JSON

Decision rules:
- Return "search" if the notes are missing, incomplete, or do not cover the research plan sufficiently
- Return "synthesize" if the notes adequately cover the research plan and the user question can be answered

Output format:
{
  "decision": "search" or "synthesize",
  "search_query": "only include if decision is search, otherwise omit"
}
"""

USER_PROMPT_ITERATION = """
User Question:
{query}

Research Plan:
{research_plan}

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

#================= INFORMATION EXTRACTION ======================

SYSTEM_PROMPT_EXTRACT_INFO = """
You are a research extraction assistant.

Your job is to extract relevant information from search results to answer a research question.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include extra text
- The response must be directly parseable JSON

Output format:
{
  "extracted": "concise extracted information relevant to the research question"
}
"""

USER_PROMPT = """
User question:
{0}
"""

USER_PROMPT_EXTRACT_INFORMATION = """
Original question:
{0}

Research question:
{1}

Search results:
{2}

Extract only the information relevant to the research question above.
"""

# =============== SYNTHESIZE ===============
SYSTEM_PROMPT_SYNTHESIZE = """
You are a research synthesis assistant.

Your job is to synthesize research notes into a clear, comprehensive answer to the user's question.

STRICT RULES:
- Output ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include extra text
- Do NOT include <think> or reasoning
- The response must be directly parseable JSON

Output format:
{
  "answer": "comprehensive answer to the user's question"
}
"""

USER_PROMPT_SYNTHESIZE = """
User Question:
{0}

Research Plan:
{1}

Research Notes:
{2}

Synthesize the research notes into a clear and comprehensive answer to the user question.
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
    
    def research_iteration_prompt_groq(self, query, research_plan, notes, iteration):
        user_prompt = USER_PROMPT.format(query, research_plan, notes, iteration)
        return SYSTEM_PROMPT_ITERATION, user_prompt
    
    def extract_information_prompt_groq(self, query, search_query, content):
        user_prompt = USER_PROMPT_EXTRACT_INFORMATION.format(query, search_query, content)
        return SYSTEM_PROMPT_EXTRACT_INFO, user_prompt
    
    def synthesize_prompt_groq(self, query, research_plan, notes):
        user_prompt = USER_PROMPT_SYNTHESIZE.format(query, research_plan, notes)
        return SYSTEM_PROMPT_SYNTHESIZE, user_prompt
    
    def research_iteration_prompt(self):
        research_iteration_prompt = PromptTemplate(template=RESEARCH_ITERATION_PROMPT, input_variables=["query", "research_plan", "notes", "iteration"])
        return research_iteration_prompt
