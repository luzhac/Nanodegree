from pydantic import BaseModel
from pydantic_ai import Agent


# ============================================================
# 1. Shared structured models
# ============================================================

class QuoteResult(BaseModel):
    price: float
    lead_time_days: int
    reason: str


class ApprovalResult(BaseModel):
    approved: bool
    approver: str
    reason: str


# ============================================================
# 2. Agent 1: Pricing Agent
# ============================================================

pricing_agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=QuoteResult,  # Changed from result_type to output_type
    system_prompt=(
        "You are a pricing agent for a paper company.\n"
        "Generate a quote based on the request."
    ),
)


# ============================================================
# 3. Agent 2: Approval Agent
# ============================================================

approval_agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=ApprovalResult,  # Changed from result_type to output_type
    system_prompt=(
        "You are a finance approval agent.\n"
        "Approve quotes under $300 automatically.\n"
        "Quotes $300 or more require manager approval."
    ),
)


# ============================================================
# 4. Orchestration (Agent → Agent)
# ============================================================

if __name__ == "__main__":
    user_request = "Quote 100 boxes of A4 paper with bulk discount"

    # ---- Agent 1 runs ----
    pricing_response = pricing_agent.run_sync(user_request)
    quote: QuoteResult = pricing_response.output

    print("\n=== QUOTE ===")
    print(quote)

    # ---- Agent 2 runs using Agent 1 output ----
    approval_prompt = (
        f"Quote details:\n"
        f"Price: {quote.price}\n"
        f"Lead time: {quote.lead_time_days} days\n"
        f"Reason: {quote.reason}"
    )

    approval_response = approval_agent.run_sync(approval_prompt)
    approval: ApprovalResult = approval_response.output

    print("\n=== APPROVAL ===")
    print(approval)