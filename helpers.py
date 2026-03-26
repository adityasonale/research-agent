import re
import json

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
