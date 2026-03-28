import json
import os
import re
from tavily import TavilyClient

def extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from a raw LLM response string.
    Handles markdown code fences (```json ... ```) and plain JSON blobs.
    """
    # Strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'```$', '', text.strip(), flags=re.MULTILINE)

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in LLM response: {e}") from e

def search_tool(query: str, k: int = 5):
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=k
    )

    results = []
    for r in response["results"]:
        results.append({
            "title": r["title"],
            "url": r["url"],
            "snippet": f"{r['title']}: {r['content'][:500]}",
        })

    return results
