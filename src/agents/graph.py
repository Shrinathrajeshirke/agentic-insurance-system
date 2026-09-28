import sys
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.schema.user_profile import UserProfile
from src.schema.guardrails import evaluate_underwriting_guardrails
from src.agents.tools import search_policy_contracts, web_search
from src.tools.calculator import calculate_policy_rates
from src.agents.tools import search_policy_contracts
from src.agents.llm import get_llm
from src.agents.search_utils import STATIC_FALLBACK_MARKER, needs_web_fallback
from src.logger import logger
from src.exception import CustomException


# ============================================================================
# 1. State Definition
# ============================================================================

class AdvisoryState(TypedDict):
    """
    Main state tracking the conversation, user profile attributes,
    underwriting guardrail evaluations, actuarial calculations, and policy contexts.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: Optional[Dict[str, Any]]
    guardrail_passed: bool
    underwriting_flags: List[str]
    rate_quotes: List[Dict[str, Any]]
    retrieved_contexts: List[Dict[str, Any]]
    next_step: Optional[str]


# ============================================================================
# 2. Graph Node Functions
# ============================================================================

def guardrails_node(state: AdvisoryState) -> Dict[str, Any]:
    """
    Evaluates applicant profile against non-negotiable IRDAI life underwriting rules,
    entry age boundaries (18-65), and HLV multiplier income ceilings.
    """
    try:
        profile = state.get("user_profile")
        if not profile:
            logger.info("No structured profile detected in state; skipping guardrail gate.")
            return {
                "guardrail_passed": True,
                "underwriting_flags": [],
            }

        logger.info(f"Evaluating underwriting guardrails for profile: {profile}")
        is_valid, flags = evaluate_underwriting_guardrails(profile)

        return {
            "guardrail_passed": is_valid,
            "underwriting_flags": flags,
        }
    except Exception as e:
        logger.error(f"Error in guardrails_node: {e}")
        raise CustomException(e, sys)


def calculator_node(state: AdvisoryState) -> Dict[str, Any]:
    """
    Calculates deterministic actuarial premiums across all benchmarked insurers
    including gender discounts, smoker loadings, and zero-percent GST.
    """
    try:
        profile = state.get("user_profile")
        if not profile:
            return {"rate_quotes": []}

        age = profile.get("age", 30)
        is_smoker = profile.get("is_smoker", False)
        gender = profile.get("gender", "Male")
        
        # Parse sum assured string or float into Crore numeric units
        desired_cover = profile.get("desired_sum_assured", "1 Crore")
        if isinstance(desired_cover, (int, float)):
            sum_assured_cr = desired_cover / 10000000.0
        elif isinstance(desired_cover, str):
            cover_map = {
                "50 Lakhs": 0.5,
                "75 Lakhs": 0.75,
                "1 Crore": 1.0,
                "1.5 Crore": 1.5,
                "2 Crore": 2.0,
                "2.5 Crore": 2.5,
                "3 Crore": 3.0,
                "5 Crore+": 5.0,
            }
            sum_assured_cr = cover_map.get(desired_cover, 1.0)
        else:
            sum_assured_cr = 1.0

        quotes = calculate_policy_rates(
            age=age,
            smoker=is_smoker,
            sum_assured_crores=sum_assured_cr,
            gender=gender,
            include_ci_rider=False,
            include_adb_rider=False,
            return_of_premium=False
        )

        logger.info(f"Generated {len(quotes)} actuarial rate quotes.")
        return {"rate_quotes": quotes}
    except Exception as e:
        logger.error(f"Error in calculator_node: {e}")
        raise CustomException(e, sys)


def retrieval_node(state: AdvisoryState) -> Dict[str, Any]:
    """
    Queries Qdrant & BM25 hybrid index for verified contract clauses.
    If no relevant internal clauses match, or if an explicit insurer/brand
    in the query is missing from local corpus, falls back to live web search.
    """
    try:
        messages = state.get("messages", [])
        if not messages:
            return {"retrieved_contexts": []}

        latest_query = messages[-1].content
        logger.info(f"Retrieving context for query: '{latest_query}'")

        # 1. Primary: Query indexed contract database (Dense + BM25 + FlashRank)
        contexts = search_policy_contracts.invoke({"query": latest_query})

        # 2. Fall back to live web search when local retrieval cannot answer:
        #    a) nothing was retrieved, OR
        #    b) the user named an insurer/plan that no retrieved chunk mentions.
        if needs_web_fallback(latest_query, contexts):
            logger.info(f"Local contract match weak or absent. Triggering live search fallback for: '{latest_query}'")
            web_result_text = web_search.invoke({"query": latest_query})

            if web_result_text:
                is_static = web_result_text.startswith(STATIC_FALLBACK_MARKER)
                if is_static:
                    logger.warning("Live search unavailable; advisor will receive static benchmark notice only.")
                    contexts = [{
                        "insurer": "Static Benchmark Reference (live search unavailable)",
                        "policy_name": "Static Benchmark",
                        "page": 1,
                        "section": "Static Reference Data",
                        "clause_type": "Static Benchmark",
                        "source_type": "web_static",
                        "text": web_result_text.replace(STATIC_FALLBACK_MARKER, "", 1).strip(),
                        "score": 0.0,
                    }]
                else:
                    contexts = [{
                        "insurer": "External Web Search (Live IRDAI / Market Data)",
                        "policy_name": "Web Benchmark Search",
                        "page": 1,
                        "section": "Live Public Disclosure",
                        "clause_type": "Market Benchmark",
                        "source_type": "web_live",
                        "text": web_result_text,
                        "score": 1.0,
                    }]

        return {"retrieved_contexts": contexts or []}
    except Exception as e:
        logger.warning(f"Error querying context in retrieval_node: {e}. Continuing with empty context.")
        return {"retrieved_contexts": []}


async def advisor_node(state: AdvisoryState) -> Dict[str, Any]:
    """
    Synthesizes underwriting warnings, calculated quotes, and retrieved contract
    excerpts into an authoritative, grounded advisory response.
    """
    try:
        llm = get_llm()
        messages = state.get("messages", [])
        flags = state.get("underwriting_flags", [])
        rate_quotes = state.get("rate_quotes", [])
        contexts = state.get("retrieved_contexts", [])

        # Build context prompt components
        flags_text = "\n".join([f"- {f}" for f in flags]) if flags else "None. All underwriting criteria passed."
        
        quotes_summary = ""
        if rate_quotes:
            quote_lines = [
                f"- {q['plan_name']}: Annual ₹{q['annual_premium']:,} | Monthly ₹{q['monthly_premium']:,}"
                for q in rate_quotes
            ]
            quotes_summary = "\n".join(quote_lines)
        else:
            quotes_summary = "No quotes calculated."

        context_blocks = []
        for c in contexts:
            if c.get("source_type") in ("web_live", "web_static"):
                header = f"[{c.get('insurer', 'Web')} | {c.get('section', 'Web')}]"
            else:
                header = f"[{c.get('insurer', 'Policy')} | Section: {c.get('section', 'General')} | Page {c.get('page', 1)}]"
            context_blocks.append(f"{header}: {c.get('text', '')}")
        contexts_text = "\n\n".join(context_blocks) if context_blocks else "No direct contract excerpts retrieved."

        system_prompt = f"""You are an elite Indian Life Insurance Advisory Agent specializing in IRDAI term insurance underwriting.

### MANDATORY ADVISORY RULES:
1. Underwriting Accuracy: Always disclose any Underwriting Warnings or Critical Violations upfront.
2. Grounded Knowledge: Ground policy terms, exclusions, free-look periods, and grace periods strictly in the provided Context Excerpts. Cite the insurer, section, or source.
3. Live Web Findings: If the excerpts originate from 'External Web Search', synthesize the information to directly answer the user's inquiry regarding external insurers or plans, and mention the source names.
4. Pricing Transparency: Reference the calculated rate table when quoting premiums. Note that individual life insurance incurs 0% GST.
5. Named Plans: If the user asks about a specific plan or insurer that does not appear in the excerpts, never present a different plan's details as the answer to their question.
6. Static Data: If the excerpts are labelled 'Static Benchmark Reference', live search failed. Tell the user plainly that live details for the requested plan could not be retrieved right now. You may share the static figures only as clearly-labelled approximate market context for the other reference plans, and must never describe them as live findings.

---
### ACTIVE UNDERWRITING STATUS:
{flags_text}

---
### CALCULATED ACTUARIAL QUOTES:
{quotes_summary}

---
### VERIFIED POLICY CONTRACT EXCERPTS & LIVE FINDINGS:
{contexts_text}
---
"""

        conversation_history = [SystemMessage(content=system_prompt)] + messages
        response = await llm.ainvoke(conversation_history)

        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in advisor_node: {e}")
        raise CustomException(e, sys)


# ============================================================================
# 3. Routing Conditional Edges
# ============================================================================

def route_after_guardrails(state: AdvisoryState) -> str:
    """
    Routes directly to advisor node if a critical guardrail rejected the applicant,
    otherwise proceeds to the actuarial calculation engine.
    """
    if not state.get("guardrail_passed", True):
        logger.warning("Applicant failed critical entry age guardrails. Bypassing calculator.")
        return "advisor_node"
    return "calculator_node"


# ============================================================================
# 4. Graph Construction & Compilation Factory
# ============================================================================

def build_advisory_graph():
    """Builds the uncompiled StateGraph definition."""
    workflow = StateGraph(AdvisoryState)

    # Add Nodes
    workflow.add_node("guardrails_node", guardrails_node)
    workflow.add_node("calculator_node", calculator_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("advisor_node", advisor_node)

    # Set Entry Point
    workflow.set_entry_point("guardrails_node")

    # Add Edges
    workflow.add_conditional_edges(
        "guardrails_node",
        route_after_guardrails,
        {
            "calculator_node": "calculator_node",
            "advisor_node": "advisor_node"
        }
    )
    workflow.add_edge("calculator_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "advisor_node")
    workflow.add_edge("advisor_node", END)

    return workflow


# Pre-compile workflow builder
workflow_builder = build_advisory_graph()


def get_graph(checkpointer=None):
    """
    Compiles the advisory graph with an optional or configured checkpointer.
    If checkpointer is None, compiles in stateless/ephemeral mode.
    """
    return workflow_builder.compile(checkpointer=checkpointer)