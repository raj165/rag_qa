# rag_qa_system/web_search.py
import os
import logging
from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Prompt for general-purpose web QA
WEB_SEARCH_PROMPT = PromptTemplate(
    input_variables=["search_results", "user_query"],
    template="""
You are a helpful assistant that answers using ONLY the search results below.

Recent Search Results:
{search_results}

User Query: {user_query}

Answer in clear, markdown-formatted text. 
If you cannot find relevant info in the results, say "I could not find this in the search results." 
Always include the sources.
"""
)

def web_search_agent(user_query: str, domains: list[str] | None = None) -> str:
    """Run Tavily search + summarize with Azure OpenAI"""
    try:
        search_tool = TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_domains=domains,     # Optional filter
            include_answer=False,
            include_raw_content=True,
            time_range="month",
        )

        search_response: Dict[str, Any] = search_tool.invoke(user_query)
        search_results_list = search_response.get("results", [])

        if not search_results_list:
            return "No relevant search results found."

        # Build context string
        search_context = "\n".join(
            f"- **{res['title']}** ({res['url']}): {res.get('content', '')[:300]}..."
            for res in search_results_list
        )

        # LLM setup
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=0.2,
        )

        runnable = WEB_SEARCH_PROMPT | llm | StrOutputParser()
        answer = runnable.invoke({
            "search_results": search_context,
            "user_query": user_query
        })

        # Add sources at end
        sources = "\n\n### Sources\n" + "\n".join(
            f"- {res['url']}" for res in search_results_list
        )
        return f"{answer}{sources}"

    except Exception as e:
        logger.error(f"Error in web_search_agent: {str(e)}")
        return f"Error: {str(e)}"
