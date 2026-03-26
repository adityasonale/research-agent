from langchain.schema.runnable import RunnableSequence

from llm_engine.prompts import AgentPrompts

def research_plan_chain(llm):

    # use a prompt template from prompts.py
    research_prompt_template = AgentPrompts().research_plan_prompt()

    chain = RunnableSequence(research_prompt_template | llm)
    return chain

def research_iteration_chain(llm):
    research_iteration_prompt = AgentPrompts().research_iteration_prompt()

    chain = RunnableSequence(research_iteration_prompt | llm)
    return chain