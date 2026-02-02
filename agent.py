from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError


# =========================
# 1. Structured Output
# =========================

class AgentDecision(BaseModel):
    action: str          # "search" or "answer"
    answer: str | None
    confidence: float


# =========================
# 2. Tool Definition
# =========================

def search_tool(query: str) -> str:
    """
    External tool (simulated)
    """
    return f"Search result for '{query}': Python is a programming language."


# =========================
# 3. Agent Definition
# =========================

class BuildingAgent:
    def __init__(self):
        # ----- Agent State -----
        self.state: Dict[str, Any] = {
            "step": 0
        }

        # ----- Short-term Memory -----
        self.memory: List[Dict[str, str]] = []

    def decide(self, user_input: str) -> str:
        """
        Very simplified 'reasoning'
        (in real use, this is where the LLM decides)
        """
        if "python" in user_input.lower():
            return "search"
        return "answer"

    def run(self, user_input: str) -> AgentDecision:
        self.state["step"] += 1

        # Save user input to memory
        self.memory.append({
            "role": "user",
            "content": user_input
        })

        action = self.decide(user_input)

        if action == "search":
            tool_result = search_tool(user_input)

            # Save tool result to memory
            self.memory.append({
                "role": "tool",
                "content": tool_result
            })

            output = {
                "action": "search",
                "answer": tool_result,
                "confidence": 0.85
            }
        else:
            output = {
                "action": "answer",
                "answer": "I can answer directly without searching.",
                "confidence": 0.60
            }

        # ----- Structured Output Validation -----
        try:
            return AgentDecision(**output)
        except ValidationError as e:
            raise RuntimeError("Invalid agent output") from e


# =========================
# 4. Run Example
# =========================

if __name__ == "__main__":
    agent = BuildingAgent()

    result = agent.run("What is Python?")
    print(result.model_dump())

    print("\nAgent State:", agent.state)
    print("\nAgent Memory:")
    for m in agent.memory:
        print(m)
