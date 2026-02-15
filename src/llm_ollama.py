# src/llm_ollama.py
import json, re, requests

SYS = (
    "You are an enterprise RAG assistant. Use ONLY the PASSAGES provided.\n"
    "- Use the EXACT citation tags shown in the prompt (e.g., [IMF2025Report:18]).\n"
    "- Include at least one citation tag [DOC:PAGE] in EVERY sentence.\n"
    "- If evidence is insufficient, reply exactly: Not enough evidence.\n"
    "- Prefer concise bullet points (2–5), each with a citation.\n"
    'Return STRICT JSON ONLY (no markdown fences): {"answer":"...","citations":["DOC:PAGE",...],"confidence":0.x}'
)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_LAST  = re.compile(r"\{[\s\S]*\}$")

def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE.sub("", text).strip()

def _coerce_json(text: str):
    """
    Make a best-effort to parse JSON, then normalize:
    - If answer is a list, join into bullets.
    - If citations missing, try extracting tags from answer.
    """
    raw = text.strip()
    m = _JSON_LAST.search(raw)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        # fallback: wrap text
        return {"answer": raw[-800:], "citations": [], "confidence": 0.3}

    # normalize answer
    ans = data.get("answer", "")
    if isinstance(ans, list):
        ans = "\n".join(f"- {s}" for s in ans)
    elif not isinstance(ans, str):
        ans = str(ans)
    data["answer"] = ans.strip()

    # normalize citations
    cits = data.get("citations") or []
    if not isinstance(cits, list):
        cits = [str(cits)]
    data["citations"] = [str(c).strip() for c in cits]
    # confidence
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
    except Exception:
        data["confidence"] = 0.5
    return data

def generate_json_ollama_chat(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: int = 220,
    num_ctx: int = 8192,
):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {
                "role": "user",
                "content": (
                    "Use the PASSAGES and QUESTION below. Output ONLY the JSON object. "
                    "Do NOT include triple backticks.\n" + prompt
                ),
            },
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
            "repeat_penalty": 1.05,
        },
        "stream": False,
    }
    r = requests.post(f"{host}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    txt = (r.json().get("message", {}) or {}).get("content", "")
    txt = _strip_code_fences(txt)
    return _coerce_json(txt)
