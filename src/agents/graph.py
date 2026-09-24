import sys
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from src.schema.state import AgentState
from src.schema.user_profile import UserProfile
from src.schema.guardrails import evaluate_underwriting_guardrails
from src.agents.llm import get_llm
from src.agents.tools import search_policy_contracts, web_search
from src.tools.calculator import calculate_benchmark_matrix
from src.logger import logger
from src.exception import CustomException

llm = get_llm()

# ==========================================
# 1. PROFILER NODE (ASYNC)
# ==========================================
async def profiler_node(state: AgentState) -> dict:
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

        response = await llm.ainvoke([SystemMessage(content=profiling_prompt)] + messages[-3:])
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


# ==========================================
# 2. DETERMINISTIC UNDERWRITING GUARDRAIL NODE (ASYNC)
# ==========================================
async def guardrail_node(state: AgentState) -> dict:
    try:
        logger.info("Executing Deterministic Underwriting Guardrail Node")
        profile = state.get("user_profile") or UserProfile()
        is_valid, alerts = evaluate_underwriting_guardrails(profile.model_dump())

        for alert in alerts:
            logger.info(f"Guardrail Alert: {alert}")

        return {"underwriting_alerts": alerts}
    except Exception as e:
        raise CustomException(e, sys)


# ==========================================
# 3. RETRIEVER & ACTUARIAL CALCULATOR NODE (ASYNC)
# ==========================================
async def retriever_node(state: AgentState) -> dict:
    try:
        logger.info("Executing Retriever Node")
        messages = state.get("messages", [])
        profile = state.get("user_profile") or UserProfile()

        human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        last_user_message = human_messages[-1] if human_messages else ""
        is_initial_assessment = "initial profile assessment" in last_user_message.lower()

        # 1. Deterministic Actuarial Calculations based on user demographics
        age = profile.age or 28
        gender = profile.gender or "Male"
        is_smoker = profile.is_smoker if profile.is_smoker is not None else False
        cover_str = profile.desired_sum_assured or "1 Crore"

        pricing_matrix = calculate_benchmark_matrix(age, gender, is_smoker, cover_str)
        pricing_text = (
            f"ACTUARIAL RATE ENGINE OUTPUT (Demographics: Age {age}, {gender}, Smoker: {is_smoker}, Cover: {cover_str}):\n"
            f"Tax Rule: Individual term insurance carries 0% GST.\n"
        )
        for item in pricing_matrix:
            pricing_text += (
                f"- {item['plan']}: Pure Term = {item['monthly']}/month ({item['annual']}/year) | "
                f"TROP (Return of Premium) = {item['trop_monthly']}/month ({item['trop_annual']}/year)\n"
            )

        pricing_block = {
            "insurer": "Actuarial Calculation Engine",
            "policy_name": "Premium Rate Engine",
            "page": 1,
            "section": "Underwriting Rate Chart",
            "clause_type": "pricing",
            "text": pricing_text,
            "score": 1.0
        }

        # 2. Vector DB Query (Qdrant)
        qdrant_results = search_policy_contracts.invoke({"query": last_user_message}) if last_user_message else []

        # 3. Assemble Policy Context
        if is_initial_assessment:
            logger.info("Initial assessment: Retrieving contract clauses and benchmarking data.")
            market_query = "Best term life insurance plans India HDFC Click 2 Protect Max Life Smart Secure Tata AIA Sampoorna Raksha ICICI iProtect eligibility CSR"
            web_results = web_search.invoke({"query": market_query})

            combined_policies = list(qdrant_results) if qdrant_results else []
            combined_policies.append(pricing_block)
            combined_policies.append({
                "insurer": "IRDAI Market Benchmark",
                "policy_name": "Major Insurers (HDFC Life, Max Life, Tata AIA, ICICI Pru)",
                "page": 1,
                "section": "Market Overview & Ratios",
                "clause_type": "market_overview",
                "text": web_results,
                "score": 1.0
            })
            return {"retrieved_policies": combined_policies, "next_action": "advisor"}

        # Follow-up turns
        combined_policies = list(qdrant_results) if qdrant_results else []
        combined_policies.append(pricing_block)

        # Check if vector DB had high confidence matches
        high_confidence = any(r.get("score", 0) > 0.45 for r in qdrant_results) if qdrant_results else False
        if not high_confidence:
            logger.info("Internal contract confidence low. Calling web search fallback.")
            web_results = web_search.invoke({"query": last_user_message}) if last_user_message else ""
            combined_policies.append({
                "insurer": "External Web Search",
                "policy_name": "Web Source",
                "page": 1,
                "section": "Web Insights",
                "clause_type": "web_search",
                "text": web_results,
                "score": 0.0
            })

        return {"retrieved_policies": combined_policies, "next_action": "advisor"}

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================
# 4. ADVISOR NODE (ASYNC)
# ==========================================
async def advisor_node(state: AgentState) -> dict:
    try:
        logger.info("Executing Advisor Node")
        profile = state.get("user_profile", UserProfile())
        policies = state.get("retrieved_policies", [])
        alerts = state.get("underwriting_alerts", [])
        messages = state.get("messages", [])

        # Format grounded references with page and section metadata
        context_lines = []
        for idx, p in enumerate(policies, 1):
            insurer = p.get("insurer", "Insurer")
            plan = p.get("policy_name", "Plan Contract")
            page = p.get("page", "N/A")
            sec = p.get("section", "Contract Clause")
            text = p.get("text", "")
            context_lines.append(
                f"[Ref {idx}] Insurer: {insurer} | Policy: {plan} | Page: {page} | Section: {sec}\nExcerpt: {text}"
            )

        context_blocks = "\n\n".join(context_lines)

        alerts_text = "\n".join([f"- {a}" for a in alerts]) if alerts else "None. Applicant meets all standard underwriting boundaries."

        human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        last_user_query = human_messages[-1] if human_messages else ""
        is_initial_assessment = "initial profile assessment" in last_user_query.lower()

        if is_initial_assessment:
            system_prompt = f"""
            You are a certified term life insurance advisor specializing in the Indian market governed by IRDAI.

            Applicant Profile:
            {profile.model_dump_json(indent=2)}

            Deterministic Underwriting Guardrail Flags:
            {alerts_text}

            Retrieved Contracts, Actuarial Pricing & Market Benchmarks:
            {context_blocks}

            Advisory Instructions:
            1. Underwriting Notice: If any UNDERWRITING WARNING, CRITICAL alert, or ADJUSTMENT appears in the flags above, you MUST address it prominently in an introductory 'Underwriting Advisory Notice' before listing recommendations. Explain why the limit applies and what documentation (ITR-V, Form 16) is necessary.
            2. Top 3 Plan Recommendations: Recommend exactly 3 recognized Indian term plans (e.g., HDFC Life Click 2 Protect Super, Max Life Smart Secure Plus, Tata AIA Sampoorna Raksha Supreme, ICICI Prudential iProtect Smart). For each plan include:
               - Exact Plan Name & Insurer.
               - Exact Indicative Premium: Quote the exact calculated monthly and annual figures from the Actuarial Rate Engine (e.g., '₹980/month or ₹11,760/year'). Mention 0% GST applies to individual policies.
               - Eligibility Fit: Why it fits income ({profile.annual_income}) and sum assured ({profile.desired_sum_assured}).
               - Key Differentiators: Critical Illness cover, Terminal Illness, Zero-cost exit (Special Exit Value), and Claim Settlement Ratio (CSR).
               - Verifiable Exclusion / Rule: Cite explicit policy provisions (e.g., 12-month suicide exclusion clause, grace period).
            3. Grounded Citations: When stating specific exclusions or rider mechanics, cite the bracketed reference (e.g., '[Source: HDFC Life, Page 12, Exclusions]').
            4. Concrete Discovery Questions: Conclude with 2 to 3 practical follow-up choices (e.g., Regular pay vs. Limited pay till age 60, Pure Term vs. Return of Premium, Critical Illness rider options).
            """
        else:
            system_prompt = f"""
            You are a certified term life insurance advisor specializing in the Indian market governed by IRDAI.

            Applicant Profile:
            {profile.model_dump_json(indent=2)}

            Deterministic Underwriting Guardrail Flags:
            {alerts_text}

            Retrieved Contracts, Actuarial Pricing & Market Benchmarks:
            {context_blocks}

            Guidelines:
            1. Direct Contract Grounding: Answer the user's inquiry directly using the retrieved excerpts. When citing clauses, state the insurer, document page, and section name.
            2. Actuarial Exactness: When asked about premiums, pricing, or Return of Premium (TROP), quote the exact numbers provided in the 'Actuarial Calculation Engine' block. Note that TROP costs ~2.2x to 2.3x more than pure term because base premiums are returned at maturity. Note that individual term plans carry 0% GST.
            3. Do not use generic refusal templates like 'As an AI language model, I do not have access...'. Use the computed actuarial figures and market benchmarks provided.
            4. Keep explanations structured, objective, and compliant with IRDAI disclosure standards.
            """

        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
        return {"messages": [response], "next_action": "end"}

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================
# 5. GRAPH DEFINITION & WORKFLOW COMPILATION
# ==========================================
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("profiler", profiler_node)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("advisor", advisor_node)

# Edges
workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "guardrail")
workflow.add_edge("guardrail", "retriever")
workflow.add_edge("retriever", "advisor")
workflow.add_edge("advisor", END)