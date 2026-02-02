from pydantic import BaseModel
from pydantic_ai import Agent
import json


class QuoteResult(BaseModel):
    price: float
    lead_time_days: int
    reason: str


quote_agent = Agent(
    model="gpt-4o-mini",
    system_prompt=(
        "You are a pricing agent for a paper company.\n"
        "Return ONLY valid JSON with fields:\n"
        "price (float), lead_time_days (int), reason (string).\n"
        "Do not include any extra text."
    ),
)

if __name__ == "__main__":
    user_request = "Quote 100 boxes of A4 paper with bulk discount"

    response = quote_agent.run_sync(user_request)

    # Convert JSON string → dict
    quote_data = json.loads(response.output)

    # Validate with Pydantic
    quote = QuoteResult.model_validate(quote_data)

    print("=== QUOTE RESULT ===")
    print("Price:", quote.price)
    print("Lead time (days):", quote.lead_time_days)
    print("Reason:", quote.reason)
