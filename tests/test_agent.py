from langchain_core.messages import HumanMessage
from src.agents.graph import agent_app
from src.schema.user_profile import UserProfile

def run_test():
    print("\n--- Testing Full LangGraph Insurance Advisory Agent ---")

    initial_state = {
        "messages": [
            HumanMessage(content = "Hi, I am a 28-year-old non-smoker earning 15 LPA. What happens under Click 2 Protect Super if suicide occurs within the first year?")
        ],
        "user_profile": UserProfile(),
        "retrieved_policies": [],
        "next_action": ""
    }

    final_state = agent_app.invoke(initial_state)

    print("\n Extracted User Profile:")
    print(final_state["user_profile"].model_dump_json(indent=2))

    print("\n Advisor Response:")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    run_test()