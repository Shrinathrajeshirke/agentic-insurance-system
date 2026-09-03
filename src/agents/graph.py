import sys
import json
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from src.schema.state import AgentState
from src.schema.user_profile import UserProfile
from src.agents.llm import get_llm
from src.agents.tools import search_policy_contracts, web_search
from src.logger import logger
from src.exception import CustomException

llm = get_llm()

# --- 1. PROFILER NODE ---
def profiler_node(state: AgentState) -> dict:
    """
    Extracts and updates user underwriting details from conversation history.
    """
    try:
        logger.info("Executing Profiler Node")
        user_profile = state.get("user_profile") or UserProfile()
        messages = state["messages"]

        profiling_prompt = f"""
        You are an insurance underwriting assistant.
        Analyze the conversation and extract structured user details into JSON:
        - age (integer or null)
        - gender (string or null)
        - annual_income (float or null)
        - is_smoker (boolean or null)
        - medical_conditions (list of strings)
        - dependents_count (integer or null)
        - desired_sum_assured (float or null)
        - policy_term_years (integer or null)

        Current Profile: {user_profile.model_dump_json()}

        Respond with ONLY a raw JSON object matching the fields above, nothing else.
        """

        response = llm.invoke([SystemMessage(content=profiling_prompt)] + messages[-3:])
        cleaned_json = response.content.strip()

        # Handle markdown blocks if returned
        if "```json" in cleaned_json:
            cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_json:
            cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

        try:
            profile_data = json.loads(cleaned_json)
            updated_profile = UserProfile(**profile_data)
        except Exception:
            updated_profile = user_profile

        return {"user_profile": updated_profile}
    except Exception as e:
        raise CustomException(e, sys)

## ---2. RETRIEVER NODE ---
def retriver_node(state: AgentState) -> dict:
    """
    Determines whether to search internal drant contracts or fallback to web search.
    """
    try:
        logger.info("Executing Retriever Node")
        last_user_message = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1]

        # Step A: Query Qdrant
        qdrant_results = search_policy_contracts.invoke({"query": last_user_message})

        # Confidence threshold evaluation
        high_confidence = any(r.get("score", 0) > 0.45 for r in qdrant_results) if qdrant_results else False

        if high_confidence:
            logger.info("Internal contract clauses found with high confidence.")
            return {
                "retrieved_policies": qdrant_results,
                "next_action": "advisor"
            }

        # Step B: Fallback to web search
        logger.info("Internal confidence low. Triggering fallback web search.")
        web_results = web_search.invoke({"query": last_user_message})

        fallback_data = [{
            "insurer": "External Web Result",
            "policy_name": "Web Source",
            "clause_type": "web_search",
            "text": web_results,
            "score": 0.0
        }]

        return {
            "retrieved_policies": fallback_data,
            "next_action": "advisor"
        }

    except Exception as e:
        raise CustomException(e, sys)

# --- 3. ADVISOR NODE ---
def advisor_node(state: AgentState) -> dict:
    """
    Genreates tailored advice using retrieved context and extracted profile.
    """
    try:
        logger.info("Executing Advisor Node")
        profile = state.get("user_profile", UserProfile())
        policies = state.get("retrieved_policies", [])
        messages = state["messages"]

        context_blocks = "\n\n".join(
            [f"Source: {p.get('insurer')} - {p.get('policy_name')} ({p.get('clause_type')})\nClause Text: {p.get('text')}" for p in policies]
        )

        system_prompt = f"""
        You are a certified term life insurance advisor.
        
        Applicant Profile:
        {profile.model_dump_json(indent=2)}

        Retrieved Policy Documents:
        {context_blocks}

        Guidelines:
        1. Base your recommendation directly on the retrieved clauses. Cite the insurer and policy name explicitly.
        2. Answer the user's question directly.
        3. Note any gaps in the user's profile if they impact eligibility.
        4. Maintain an objective, professional tone.
        """
        
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        updated_messages = messages + [response]
        return {"messages": updated_messages, "next_action": "end"}

    except Exception as e:
        raise CustomException(e, sys)


# --- 4. GRAPH COMPILATION ---
workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("retriver", retriver_node)
workflow.add_node("advisor", advisor_node)

workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "retriver")
workflow.add_edge("retriver", "advisor")
workflow.add_edge("advisor", END)

agent_app = workflow.compile()