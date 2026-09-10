import os
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate
from src.agents.graph import agent_app
from src.schema.user_profile import UserProfile
from src.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

judge_llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

DATASET_NAME = "insurance_policy_ground_truth"

# 1. Dataset Verification
if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Benchmark QA pairs for term life insurance clause accuracy"
    )
    client.create_examples(
        inputs=[
            {"question": "What is the suicide exclusion rule under HDFC Click 2 Protect Super?"},
            {"question": "Does Max Life Smart Secure Plus provide a critical illness benefit, and is there a waiting period?"}
        ],
        outputs=[
            {"expected": "Void within 12 months, 80% premium refunded."},
            {"expected": "Covers 40 illnesses with a 90-day waiting period."}
        ],
        dataset_id=dataset.id
    )

# 2. Target Pipeline Function (with unique thread IDs)
def target_pipeline(inputs: dict):
    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}
    
    state = {
        "messages": [HumanMessage(content=inputs["question"])],
        "user_profile": UserProfile(),
        "retrieved_policies": [],
        "next_action": ""
    }
    result = agent_app.invoke(state, config=config)
    return {
        "answer": result["messages"][-1].content,
        "retrieved_context": [p.get("text", "") for p in result.get("retrieved_policies", [])]
    }

# 3. Groundedness Evaluator
def evaluate_groundedness(run, example) -> dict:
    answer = run.outputs.get("answer", "")
    contexts = "\n".join(run.outputs.get("retrieved_context", []))

    eval_prompt = f"""
    Retrieved Context:
    {contexts}

    Generated Answer:
    {answer}

    Task: Determine if the generated answer is completely grounded in the retrieved context without hallucinating unmentioned terms or clauses.
    Respond strictly in JSON format:
    {{"score": 1, "explanation": "grounded"}} OR {{"score": 0, "explanation": "hallucinated"}}
    """

    res = judge_llm.invoke(eval_prompt).content.strip()
    try:
        clean = res.split("```json")[1].split("```")[0].strip() if "```json" in res else res
        parsed = json.loads(clean)
        return {"key": "faithfulness", "score": parsed.get("score", 0), "comment": parsed.get("explanation")}
    except Exception:
        return {"key": "faithfulness", "score": 1 if "1" in res else 0}

if __name__ == "__main__":
    print("Running LangSmith Evaluation benchmark sequentially...")
    results = evaluate(
        target_pipeline,
        data=DATASET_NAME,
        evaluators=[evaluate_groundedness],
        experiment_prefix="rag-groundedness-test",
        max_concurrency=1
    )
    print("Evaluation successfully finished!")