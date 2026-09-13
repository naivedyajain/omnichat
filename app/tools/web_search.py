"""Web search via Tavily with detailed step-by-step logging."""
import json
import requests


def web_search(query: str, keys: dict) -> str:
    api_key = keys.get("tavily", "")
    print(f"[Tavily Log] Initiating web search...")
    print(f"[Tavily Log] Query: '{query}'")
    print(f"[Tavily Log] API Key present: {bool(api_key)}, length: {len(api_key)}")

    if not api_key:
        err = "ERROR: No Tavily API key provided in settings."
        print(f"[Tavily Log] {err}")
        return err

    try:
        print("[Tavily Log] Sending POST request to https://api.tavily.com/search...")
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "include_answer": True,
            },
            timeout=30,
        )
        print(f"[Tavily Log] Response HTTP status code: {resp.status_code}")

        data = resp.json()
        print("[Tavily Log] FULL TAVILY API RESPONSE:")
        print(json.dumps(data, indent=2))

        parts = []
        answer = data.get("answer")
        if answer:
            print(f"[Tavily Log] Extracted summary answer: '{answer}'")
            parts.append("Summary: " + answer)

        results = data.get("results", [])
        print(f"[Tavily Log] Processing {len(results)} search results...")
        for idx, r in enumerate(results, start=1):
            title = r.get('title', '')
            content = r.get('content', '')
            url = r.get('url', '')
            print(f"[Tavily Log] Result #{idx}: title='{title}', url='{url}'")
            parts.append(f"- {title}: {content} ({url})")

        output = "\n".join(parts) if parts else "No results found."
        print(f"[Tavily Log] Search parsing complete. Output length: {len(output)} chars.")
        return output

    except Exception as e:
        err_msg = f"ERROR during web search: {repr(e)}"
        print(f"[Tavily Log] {err_msg}")
        return err_msg
