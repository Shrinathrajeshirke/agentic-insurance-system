import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agents.graph import get_graph
from src.config import settings
from src.logger import logger


# Windows async loop policy fix for psycopg/subprocesses if on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class EvaluationMetrics(BaseModel):
    faithfulness_score: float = Field(
        ..., description="Score between 0.0 and 1.0 indicating if all statements are grounded in retrieved context."
    )
    faithfulness_reasoning: str = Field(
        ..., description="Brief justification for the faithfulness score."
    )
    context_relevance_score: float = Field(
        ..., description="Score between 0.0 and 1.0 indicating if retrieved context contains the necessary answer."
    )
    context_relevance_reasoning: str = Field(
        ..., description="Brief justification for context relevance."
    )


JUDGE_PROMPT = """You are an independent LLM Judge evaluating an Agentic RAG Insurance Advisory System.
Evaluate the response based on the question and retrieved context chunks.

User Question: {question}
Retrieved Context:
{context}

Generated Answer: {answer}

Criteria:
1. Faithfulness (0.0 to 1.0): Does the answer make factual claims NOT supported by the context or underwriting rules?
2. Context Relevance (0.0 to 1.0): Did the retrieved context contain the information needed to resolve the user's question?

Respond strictly according to the schema.
"""


def run_llm_judge(question: str, context: str, answer: str) -> EvaluationMetrics:
    """Invokes OpenAI LLM-as-a-judge with structured output."""
    judge_llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0.0
    )
    structured_judge = judge_llm.with_structured_output(EvaluationMetrics)
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    return structured_judge.invoke(prompt)


async def evaluate_test_case(graph, test_case: Dict[str, Any]) -> Dict[str, Any]:
    question = test_case["question"]
    profile = test_case["profile"]
    expected_clause = test_case.get("expected_clause_topic", "").lower()
    expected_keywords = test_case.get("expected_answer_keywords", [])

    # Set up real state expected by AdvisoryState in src/agents/graph.py
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_profile": profile,
        "guardrail_passed": True,
        "underwriting_flags": [],
        "rate_quotes": [],
        "retrieved_contexts": [],
        "next_step": None
    }

    # 1. Latency Measurement & Real Graph Invocation
    t_start = time.perf_counter()
    final_state = await graph.ainvoke(initial_state)
    elapsed_sec = time.perf_counter() - t_start

    # Extract results from real state
    guardrail_passed = final_state.get("guardrail_passed", True)
    underwriting_flags = final_state.get("underwriting_flags", [])
    retrieved_chunks = final_state.get("retrieved_contexts", [])
    assistant_reply = final_state["messages"][-1].content

    # 2. Objective Contract Clause Recall (Using expected_clause_topic)
    clause_hit = False
    if expected_clause:
        for chunk in retrieved_chunks:
            section_name = chunk.get("section", "").lower()
            chunk_text = chunk.get("text", "").lower()
            if expected_clause in section_name or expected_clause in chunk_text:
                clause_hit = True
                break
    else:
        clause_hit = True

    # Build context representation for the judge
    context_text = "\n---\n".join(
        [f"[{c.get('insurer', 'Policy')} p.{c.get('page', 1)} | {c.get('section', 'Clause')}]: {c.get('text', '')}" for c in retrieved_chunks]
    ) if retrieved_chunks else "No specific contract chunks retrieved."

    # 3. LLM Judge Assessment (Faithfulness & Relevance)
    eval_result = run_llm_judge(question, context_text, assistant_reply)

    # 4. Check keyword/topic alignment
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in assistant_reply.lower())
    keyword_recall = round(keyword_hits / len(expected_keywords), 2) if expected_keywords else 1.0

    # 5. Expected Guardrail Check
    # TC-06 is an applicant aged 67 (> 65), so guardrail is expected to be False.
    expected_valid = False if profile.get("age", 30) > 65 or profile.get("age", 30) < 18 else True
    guardrail_correct = (guardrail_passed == expected_valid)

    return {
        "id": test_case["id"],
        "question": question,
        "latency_sec": round(elapsed_sec, 2),
        "guardrail_passed": guardrail_passed,
        "guardrail_correct": guardrail_correct,
        "flags_raised": len(underwriting_flags),
        "clause_hit": clause_hit,
        "faithfulness": eval_result.faithfulness_score,
        "faithfulness_reasoning": eval_result.faithfulness_reasoning,
        "context_relevance": eval_result.context_relevance_score,
        "context_relevance_reasoning": eval_result.context_relevance_reasoning,
        "keyword_recall": keyword_recall
    }


async def main():
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "golden_dataset.json")
    results_path = os.path.join(base_dir, "eval_results.json")

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    # Compile the real agent graph without checkpointer for evaluation
    graph = get_graph()

    print(f"\n=======================================================")
    print(f"  RUNNING REAL-GRAPH RAG BENCHMARK ({len(golden_dataset)} Golden Scenarios)")
    print(f"=======================================================\n")

    results = []
    for tc in golden_dataset:
        print(f"Evaluating {tc['id']}: {tc['question'][:60]}...")
        res = await evaluate_test_case(graph, tc)
        print(f"   -> Latency: {res['latency_sec']}s | Clause Hit: {res['clause_hit']} | Faithfulness: {res['faithfulness']:.2f}")
        results.append(res)

    # Compute Summary Aggregates
    n = len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / n
    guardrail_accuracy = (sum(1 for r in results if r["guardrail_correct"]) / n) * 100.0
    clause_recall = (sum(1 for r in results if r["clause_hit"]) / n) * 100.0
    avg_faithfulness = (sum(r["faithfulness"] for r in results) / n) * 100.0
    avg_relevance = (sum(r["context_relevance"] for r in results) / n) * 100.0
    avg_keyword_recall = (sum(r["keyword_recall"] for r in results) / n) * 100.0

    summary_payload = {
        "sample_size": n,
        "guardrail_accuracy_pct": round(guardrail_accuracy, 2),
        "contract_clause_recall_pct": round(clause_recall, 2),
        "mean_latency_sec": round(avg_latency, 2),
        "mean_faithfulness_pct": round(avg_faithfulness, 2),
        "mean_context_relevance_pct": round(avg_relevance, 2),
        "mean_keyword_recall_pct": round(avg_keyword_recall, 2),
        "detailed_results": results
    }

    # Persist auditable results to file
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("                      EVALUATION HARNESS SUMMARY                       ")
    print("=" * 80)
    print(f"{'Test ID':<8} | {'Latency':<8} | {'Guardrail':<10} | {'Clause Hit':<11} | {'Faithful':<9} | {'Relevance':<10}")
    print("-" * 80)
    for r in results:
        gr_str = "PASS" if r["guardrail_correct"] else "FAIL"
        cl_str = "HIT" if r["clause_hit"] else "MISS"
        print(f"{r['id']:<8} | {r['latency_sec']:<6.2f}s | {gr_str:<10} | {cl_str:<11} | {r['faithfulness']:<9.2f} | {r['context_relevance']:<10.2f}")
    print("=" * 80)
    print(f"SAMPLE SIZE (N)                : {n}")
    print(f"UNDERWRITING GUARDRAIL ACCURACY: {guardrail_accuracy:.1f}%")
    print(f"CONTRACT CLAUSE RECALL         : {clause_recall:.1f}%")
    print(f"MEAN ADVISORY LATENCY          : {avg_latency:.2f}s")
    print(f"MEAN FAITHFULNESS SCORE        : {avg_faithfulness:.1f}%")
    print(f"MEAN CONTEXT RELEVANCE SCORE   : {avg_relevance:.1f}%")
    print(f"DOMAIN KEYWORD RECALL          : {avg_keyword_recall:.1f}%")
    print(f"AUDIT TRAIL SAVED TO           : {os.path.abspath(results_path)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())