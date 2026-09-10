import asyncio
import json

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from inspect_ai.tool import Tool, ToolError, tool


@tool
def web_search(backend: str, max_results: int, timeout: int) -> Tool:
    """Search public websites without a model-specific API or separate search credentials."""
    if any(type(value) is not int or value <= 0 for value in (max_results, timeout)):
        raise ValueError("Search max_results and timeout must be positive integers")

    async def execute(query: str) -> str:
        """Search the Internet and return page titles, URLs, and snippets.

        Args:
            query: Search terms, optionally including a site: filter.
        """
        if not query.strip():
            raise ToolError("Provide a non-empty search query")

        # Run blocking network requests outside Inspect's event loop.
        try:
            results = await asyncio.to_thread(
                DDGS(timeout=timeout).text,
                query,
                backend=backend,
                max_results=max_results,
            )
        except DDGSException as error:
            raise ToolError(f"Web search failed: {error}") from error
        return json.dumps(results, ensure_ascii=False)

    return execute
