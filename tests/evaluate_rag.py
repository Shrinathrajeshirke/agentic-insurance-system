import json
import os
import sys
import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.schema.guardrails import evaluate_underwriting_guardrails
from src.agents.tools import search_policy_contracts
from src.config import settings
from src.logger import logger


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


async def evaluate_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    question = test_case["question"]
    profile = test_case["profile"]

    # 1. Run Underwriting Guardrail
    is_valid, flags = evaluate_underwriting_guardrails(profile)

    # 2. Retrieve Evidence Chunks from Qdrant
    retrieved_chunks = search_policy_contracts.invoke({"query": question})
    context_text = "\n---\n".join(
        [f"[{c['insurer']} p.{c['page']}]: {c['text']}" for c in retrieved_chunks]
    ) if retrieved_chunks else "No specific contract chunks retrieved."

    # 3. Advisory generation simulation
    flag_context = "\n".join(flags) if flags else "Underwriting Profile: Approved"
    advisor_llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0.1
    )

    response = await advisor_llm.ainvoke(
        f"Underwriting alerts:\n{flag_context}\n\n"
        f"Context excerpts:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Provide an accurate, cited advisory answer based strictly on the above context and underwriting rules."
    )
    generated_answer = response.content

    # 4. LLM Judge Assessment
    eval_result = run_llm_judge(question, context_text, generated_answer)

    # 5. Check keyword/topic alignment
    keyword_hits = sum(1 for kw in test_case["expected_answer_keywords"] if kw.lower() in generated_answer.lower())
    keyword_recall = round(keyword_hits / len(test_case["expected_answer_keywords"]), 2)

    return {
        "id": test_case["id"],
        "question": question,
        "guardrail_passed": is_valid,
        "flags_raised": len(flags),
        "faithfulness": eval_result.faithfulness_score,
        "context_relevance": eval_result.context_relevance_score,
        "keyword_recall": keyword_recall
    }


async def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    print(f"\n=======================================================")
    print(f"  RUNNING RAG EVALUATION BENCHMARK ({len(golden_dataset)} Golden Scenarios)")
    print(f"=======================================================\n")

    results = []
    for tc in golden_dataset:
        print(f"Evaluating {tc['id']}: {tc['question'][:60]}...")
        res = await evaluate_test_case(tc)
        results.append(res)

    # Compute Aggregate Metrics
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["context_relevance"] for r in results) / len(results)
    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print("                 EVALUATION HARNESS SUMMARY            ")
    print("=" * 65)
    print(f"{'Test ID':<10} | {'Faithfulness':<14} | {'Relevance':<12} | {'Keyword Recall':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['id']:<10} | {r['faithfulness']:<14.2f} | {r['context_relevance']:<12.2f} | {r['keyword_recall']:<15.2f}")
    print("=" * 65)
    print(f"MEAN FAITHFULNESS SCORE:       {avg_faithfulness:.2%}")
    print(f"MEAN CONTEXT RELEVANCE SCORE:  {avg_relevance:.2%}")
    print(f"KEYWORD / CLAUSE RECALL:       {avg_recall:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())