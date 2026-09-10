import sys
import json
import sqlite3
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
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
        messages = state.get("messages", [])

        profiling_prompt = f"""
        You are an insurance underwriting assistant.
        Analyze the conversation and extract structured user details into JSON:
        - age (integer or null)
        - gender (string or null)
        - annual_income (string, float, or null; e.g., "6 - 10 Lakhs")
        - is_smoker (boolean or null)
        - medical_conditions (list of strings)
        - dependents_count (integer or null)
        - desired_sum_assured (string, float, or null; e.g., "1.5 Crore")
        - policy_term_years (integer or null)

        Current Profile: {user_profile.model_dump_json()}

        Respond with ONLY a raw JSON object matching the fields above, nothing else.
        """

        response = llm.invoke([SystemMessage(content=profiling_prompt)] + messages[-3:])
        cleaned_json = response.content.strip()

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

# --- 2. RETRIEVER NODE ---
def retriever_node(state: AgentState) -> dict:
    try:
        logger.info("Executing Retriever Node")
        messages = state.get("messages", [])
        human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        last_user_message = human_messages[-1] if human_messages else ""
        is_initial_assessment = "initial profile assessment" in last_user_message.lower()

        # Step A: Query Qdrant vector database
        qdrant_results = search_policy_contracts.invoke({"query": last_user_message}) if last_user_message else []
        
        # Step B: For initial recommendations, fetch live Indian market options to supplement internal contracts
        if is_initial_assessment:
            logger.info("Initial assessment triggered. Supplementing Qdrant with Indian market plan benchmarks.")
            market_query = "Best term life insurance plans India HDFC Click 2 Protect Max Life Smart Secure Tata AIA Sampoorna Raksha ICICI iProtect eligibility CSR"
            web_results = web_search.invoke({"query": market_query})
            
            combined_policies = list(qdrant_results) if qdrant_results else []
            combined_policies.append({
                "insurer": "Indian Market Benchmark",
                "policy_name": "Major Insurers (HDFC Life, Max Life, Tata AIA, ICICI Pru)",
                "clause_type": "market_overview",
                "text": web_results,
                "score": 1.0
            })
            return {"retrieved_policies": combined_policies, "next_action": "advisor"}

        # Standard routing for follow-up turns
        high_confidence = any(r.get("score", 0) > 0.45 for r in qdrant_results) if qdrant_results else False
        if high_confidence:
            logger.info("Internal contract clauses found with high confidence.")
            return {"retrieved_policies": qdrant_results, "next_action": "advisor"}

        logger.info("Internal confidence low. Fallback to web search.")
        web_results = web_search.invoke({"query": last_user_message}) if last_user_message else ""
        fallback_data = [{
            "insurer": "External Web Result",
            "policy_name": "Web Source",
            "clause_type": "web_search",
            "text": web_results,
            "score": 0.0
        }]
        return {"retrieved_policies": fallback_data, "next_action": "advisor"}

    except Exception as e:
        raise CustomException(e, sys)

# --- 3. ADVISOR NODE ---
def advisor_node(state: AgentState) -> dict:
    try:
        logger.info("Executing Advisor Node")
        profile = state.get("user_profile", UserProfile())
        policies = state.get("retrieved_policies", [])
        messages = state.get("messages", [])

        context_blocks = "\n\n".join(
            [f"Source: {p.get('insurer')} - {p.get('policy_name')} ({p.get('clause_type')})\nClause Text: {p.get('text')}" for p in policies]
        )

        human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        last_user_query = human_messages[-1] if human_messages else ""
        is_initial_assessment = "initial profile assessment" in last_user_query.lower()

        if is_initial_assessment:
            system_prompt = f"""
            You are a certified term life insurance advisor specializing in the Indian insurance market governed by IRDAI.

            Applicant Profile:
            {profile.model_dump_json(indent=2)}

            Retrieved Policy Contracts & Market Data:
            {context_blocks}

            Guidelines:
            1. Recommend exactly 3 specific, recognized Indian term plans (e.g., HDFC Life Click 2 Protect Super, Max Life Smart Secure Plus, Tata AIA Sampoorna Raksha Supreme, or ICICI Prudential iProtect Smart). NEVER use generic placeholders like "Plan A" or "Standard Term Plan".
            2. For each recommendation, provide:
               - Exact Plan Name & Insurer.
               - Why it fits their income ({profile.annual_income}) and desired cover ({profile.desired_sum_assured}).
               - Key features (e.g., Critical Illness cover, Terminal Illness, Zero-cost exit option).
               - Critical exclusion / underwriting rule (e.g., 12-month suicide exclusion clause, medical loading).
            3. Conclude with 2 to 3 practical follow-up questions tailored to their situation (e.g., regular pay vs. limited pay options, return of premium, or rider selection).
            """
        else:
            system_prompt = f"""
            You are a certified term life insurance advisor specializing in the Indian market.

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
conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("advisor", advisor_node)

workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "retriever")
workflow.add_edge("retriever", "advisor")
workflow.add_edge("advisor", END)

agent_app = workflow.compile(checkpointer=checkpointer)