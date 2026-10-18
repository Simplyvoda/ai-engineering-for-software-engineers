from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import json

client = OpenAI()


class Feature(BaseModel):
    name: str
    importance: Literal["low", "medium", "high", "critical"]
    description: str


class SearchFeatures(BaseModel):
    features: list[Feature]


# Using OpenAI's structured output with Pydantic
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": """Extract search features from this text as JSON.

Text: "Price is crucial, but guest reviews matter too. Amenities help but are secondary."

Return JSON in this exact format with all fields for each feature:
{
  "features": [
    {"name": "price", "importance": "critical", "description": "affects visibility"},
    {"name": "guest reviews", "importance": "high", "description": "builds trust"},
    {"name": "amenities", "importance": "low", "description": "secondary factor"}
  ]
}
"""
    }],
    response_format={"type": "json_object"},
)

data = json.loads(response.choices[0].message.content)
result = SearchFeatures(**data)   # Pydantic validates types automatically

for feature in result.features:
    print(f"{feature.name}: {feature.importance}")