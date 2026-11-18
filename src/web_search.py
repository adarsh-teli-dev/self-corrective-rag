
import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return normalized evidence.

    Each result contains title, URL, and content.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. Add it to your project .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )

    results = []

    for item in response.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")

        # Ignore results that provide no usable evidence.
        if not url or not content:
            continue

        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "source": "web",
            }
        )

    return results


def main():
    print("Fallback Web Search Tester")
    print("-" * 50)

    while True:
        query = input(
            "\nEnter a search query (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a query.")
            continue

        try:
            results = search_web(query, max_results=3)

            if not results:
                print("No usable web results returned.")
                continue

            for index, result in enumerate(results, start=1):
                print(f"\n{'=' * 60}")
                print(f"Result {index}")
                print("Title:", result["title"])
                print("URL:", result["url"])
                print("Content:", result["content"][:1000])

        except Exception as exc:
            print(f"Web search failed: {exc}")


if __name__ == "__main__":
    main()
