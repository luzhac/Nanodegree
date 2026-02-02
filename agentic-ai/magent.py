from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError

# ============================================================
# 1. Structured Output (System-level contract)
# ============================================================

class AgentOutput(BaseModel):
    agent: str
    content: str
    confidence: float


# ============================================================
# 2. External Tool (simulated)
# ============================================================

def search_tool(query: str) -> str:
    """
    Simulated external tool.
    In real systems this could be:
    - Web search
    - API call
    - Database query
    """
    return f"[SEARCH RESULT] Python is a programming language used for automation and AI."


# ============================================================
# 3. Shared State (Multi-Agent core difference)
# ============================================================

shared_state: Dict[str, Any] = {
    "steps": [],
    "approved": False,
    "final_result": None,
    "memory": []
}


# ============================================================
# 4. Router Agent
# ============================================================

class RouterAgent:
    """
    Decide which path the system should take.
    """

    def run(self, user_input: str) -> str:
        shared_state["steps"].append("router")
        if "what is" in user_input.lower():
            return "simple"
        return "needs_tool"


# ============================================================
# 5. Planner Agent
# ============================================================

class PlannerAgent:
    """
    Decide how to solve the task.
    """

    def run(self, task_type: str, user_input: str) -> str:
        shared_state["steps"].append("planner")
        plan = f"Answer the question: '{user_input}'"
        return plan


# ============================================================
# 6. Executor Agent (Tool user)
# ============================================================

class ExecutorAgent:
    """
    Execute the plan.
    """

    def run(self, plan: str, user_input: str) -> AgentOutput:
        shared_state["steps"].append("executor")

        if "python" in user_input.lower():
            result = search_tool(user_input)
        else:
            result = "Direct answer without tool."

        output = {
            "agent": "Executor",
            "content": result,
            "confidence": 0.8
        }

        return AgentOutput(**output)


# ============================================================
# 7. Reviewer Agent (Evaluation)
# ============================================================

class ReviewerAgent:
    """
    Evaluate whether the result is acceptable.
    """

    def run(self, output: AgentOutput) -> bool:
        shared_state["steps"].append("reviewer")

        if output.confidence >= 0.7:
            shared_state["approved"] = True
            shared_state["final_result"] = output.content
            return True

        return False


# ============================================================
# 8. Orchestrator (System controller)
# ============================================================

class Orchestrator:
    """
    This is NOT an agent.
    This is the system controller.
    """

    def __init__(self):
        self.router = RouterAgent()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.reviewer = ReviewerAgent()

    def run(self, user_input: str) -> str:
        # Save to shared memory
        shared_state["memory"].append({
            "role": "user",
            "content": user_input
        })

        # 1. Route
        task_type = self.router.run(user_input)

        # 2. Plan
        plan = self.planner.run(task_type, user_input)

        # 3. Execute
        try:
            output = self.executor.run(plan, user_input)
        except ValidationError:
            return "Executor produced invalid output."

        # Save execution result to memory
        shared_state["memory"].append({
            "role": "agent",
            "content": output.content
        })

        # 4. Review
        approved = self.reviewer.run(output)

        # 5. Final decision
        if approved:
            return shared_state["final_result"]

        return "Result not approved. Retry required."


# ============================================================
# 9. Run the system
# ============================================================

if __name__ == "__main__":
    system = Orchestrator()

    question = "What is Python?"
    answer = system.run(question)

    print("\n=== FINAL ANSWER ===")
    print(answer)

    print("\n=== SHARED STATE ===")
    for k, v in shared_state.items():
        print(f"{k}: {v}")
