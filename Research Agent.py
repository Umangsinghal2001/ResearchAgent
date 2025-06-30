import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated
import operator
from typing import Literal


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import ArxivLoader, WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for the agent
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    plan: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    summaries: List[str]
    final_report: str
    iteration_count: int
    max_iterations: int

# Pydantic models for structured outputs
class SearchPlan(BaseModel):
    """Plan for research operations"""
    operations: List[Dict[str, str]] = Field(description="List of search operations to perform")
    reasoning: str = Field(description="Reasoning behind the plan")
    needs_more_search: bool = Field(description="Whether more searches might be needed")

class PaperSummary(BaseModel):
    """Summary of a research paper"""
    title: str = Field(description="Paper title")
    authors: str = Field(description="Paper authors")
    summary: str = Field(description="Detailed summary of the paper")
    key_findings: List[str] = Field(description="Key findings from the paper")
    relevance_score: float = Field(description="Relevance score from 0-1")

class DatasetRecommendation(BaseModel):
    """Dataset recommendation"""
    name: str = Field(description="Dataset name")
    description: str = Field(description="Dataset description")
    url: str = Field(description="Dataset URL")
    relevance: str = Field(description="Why this dataset is relevant")

class PlanningPromptOutput(BaseModel):
        next_action: Literal["web_search", "arxiv_search", "load_documents", "end"] = Field(
            ..., description="The next action to perform."
        )
        search_query: str = Field(..., description="The search query to use if the action involves searching.")
        reasoning: str = Field(..., description="Explanation of why this action was chosen.")    

class AIResearchAgent:
    def __init__(self, openai_api_key: str, serpapi_key: str = None):
        """Initialize the AI Research Agent with necessary API keys"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Initialize search tools
        self.setup_tools(serpapi_key)
        
        # Initialize the graph
        self.graph = self.create_graph()
        
    def setup_tools(self, serpapi_key: str = None):
        """Setup search and retrieval tools"""
        # Web search tool
        if serpapi_key:
            search_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)
            self.web_search_tool = Tool(
                name="web_search",
                description="Search the web for recent information and papers",
                func=search_wrapper.run
            )
        else:
            # Fallback to DuckDuckGo if SerpAPI not available
            ddg_search = DuckDuckGoSearchAPIWrapper()
            self.web_search_tool = Tool(
                name="web_search",
                description="Search the web for recent information and papers",
                func=ddg_search.run
            )
        
        # ArXiv search tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=5)
        self.arxiv_search_tool = Tool(
            name="arxiv_search",
            description="Search ArXiv for academic papers",
            func=arxiv_wrapper.run
        )
    
    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("web_searcher", self.web_search_node)
        workflow.add_node("arxiv_searcher", self.arxiv_search_node)
        workflow.add_node("document_loader", self.document_loader_node)
        workflow.add_node("summarizer", self.summarizer_node)
        workflow.add_node("report_generator", self.report_generator_node)
        
        # Define the flow
        workflow.set_entry_point("planner")
        
        # Add conditional edges from planner
        workflow.add_conditional_edges(
            "planner",
            self.should_continue_search,
            {
                "web_search": "web_searcher",
                "arxiv_search": "arxiv_searcher",
                "load_documents": "document_loader",
                "end": END
            }
        )
        
        # Add edges from search nodes back to planner for potential re-planning
        workflow.add_edge("web_searcher", "planner")
        workflow.add_edge("arxiv_searcher", "planner")
        workflow.add_edge("document_loader", "summarizer")
        workflow.add_edge("summarizer", "report_generator")
        workflow.add_edge("report_generator", END)
        
        return workflow.compile()
    

    
    def planner_node(self, state: AgentState) -> AgentState:
        """Plan the research operations based on query and current state"""
        logger.info("Planning research operations...")
        
        if state["iteration_count"] >= state["max_iterations"]:
            logger.info("Max iterations reached, proceeding to document loading")
            state["plan"] = {"next_action": "load_documents"}
            return state
        
        # Create planning prompt
        planning_prompt = ChatPromptTemplate.from_template("""
        You are a research planning assistant. Analyze the user query and current search results to plan next actions.
        
        User Query: {query}
        
        Current Search Results Count: {results_count}
        Current Iteration: {iteration}
        Max Iterations: {max_iterations}
        
        Previous Search Results Summary: {results_summary}
        
        Based on the query and current state, determine what actions to take next:
        1. "web_search" - Search the web for recent papers, articles, or datasets
        2. "arxiv_search" - Search ArXiv for academic papers
        3. "load_documents" - Load and process found documents
        4. "end" - If enough information has been gathered
        
        Provide your plan as JSON with the following structure:
        {{
            "next_action": "web_search|arxiv_search|load_documents|end",
            "search_query": "specific search query if searching",
            "reasoning": "explanation of why this action was chosen"
        }}
        """)
        
        # Prepare context for planning
        results_summary = ""
        if state["search_results"]:
            results_summary = f"Found {len(state['search_results'])} results so far. "
            results_summary += "Recent results include: " + "; ".join([
                result.get("title", result.get("snippet", ""))[:100] 
                for result in state["search_results"][-3:]
            ])
        
        messages = planning_prompt.format_messages(
            query=state["query"],
            results_count=len(state["search_results"]),
            iteration=state["iteration_count"],
            max_iterations=state["max_iterations"],
            results_summary=results_summary
        )
        
        # Get plan from LLM
        response = self.llm.with_structured_output(PlanningPromptOutput).invoke(messages)
        
        try:
            plan = response.dict()
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, defaulting to web search")
            plan = {
                "next_action": "web_search" if not state["search_results"] else "load_documents",
                "search_query": state["query"],
                "reasoning": "Default action due to parsing error"
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            plan = {
                "next_action": "web_search" if not state["search_results"] else "load_documents",
                "search_query": state["query"],
                "reasoning": f"Error during planning: {e}"
            }
                
        
        state["plan"] = plan
        state["iteration_count"] += 1
        
        logger.info(f"Plan: {plan['next_action']} - {plan['reasoning']}")
        return state
    
    def should_continue_search(self, state: AgentState) -> str:
        """Determine the next action based on the plan"""
        plan = state.get("plan", {})
        next_action = plan.get("next_action", "end")
        
        if next_action in ["web_search", "arxiv_search", "load_documents", "end"]:
            return next_action
        else:
            return "end"
    
    def web_search_node(self, state: AgentState) -> AgentState:
        """Perform web search"""
        logger.info("Performing web search...")
        
        search_query = state["plan"].get("search_query", state["query"])
        
        try:
            results = self.web_search_tool.run(search_query)
            
            # Parse results (this will depend on your search tool's output format)
            if isinstance(results, str):
                # Simple parsing for string results
                search_results = [{
                    "title": "Web Search Result",
                    "snippet": results,
                    "source": "web_search",
                    "query": search_query
                }]
            else:
                search_results = results if isinstance(results, list) else [results]
            
            state["search_results"].extend(search_results)
            logger.info(f"Found {len(search_results)} web search results")
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            state["search_results"].append({
                "title": "Search Error",
                "snippet": f"Web search failed: {e}",
                "source": "web_search_error",
                "query": search_query
            })
        
        return state
    
    def arxiv_search_node(self, state: AgentState) -> AgentState:
        """Perform ArXiv search"""
        logger.info("Performing ArXiv search...")
        
        search_query = state["plan"].get("search_query", state["query"])
        
        try:
            results = self.arxiv_search_tool.run(search_query)
            
            # Parse ArXiv results
            if isinstance(results, str):
                # ArXiv tool typically returns formatted string
                arxiv_results = [{
                    "title": "ArXiv Search Results",
                    "content": results,
                    "source": "arxiv_search",
                    "query": search_query
                }]
            else:
                arxiv_results = results if isinstance(results, list) else [results]
            
            state["search_results"].extend(arxiv_results)
            logger.info(f"Found ArXiv search results")
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            state["search_results"].append({
                "title": "ArXiv Search Error",
                "content": f"ArXiv search failed: {e}",
                "source": "arxiv_search_error",
                "query": search_query
            })
        
        return state
    
    def document_loader_node(self, state: AgentState) -> AgentState:
        """Load and parse documents from search results"""
        logger.info("Loading documents...")
        
        documents = []
        
        for result in state["search_results"]:
            try:
                if result.get("source") == "arxiv_search":
                    # For ArXiv results, the content is already loaded
                    documents.append({
                        "title": result.get("title", "ArXiv Paper"),
                        "content": result.get("content", result.get("snippet", "")),
                        "source": "arxiv",
                        "metadata": result
                    })
                elif "url" in result:
                    # Try to load web content
                    loader = WebBaseLoader([result["url"]])
                    docs = loader.load()
                    for doc in docs:
                        documents.append({
                            "title": result.get("title", "Web Document"),
                            "content": doc.page_content,
                            "source": "web",
                            "metadata": result
                        })
                else:
                    # Use snippet/content directly
                    documents.append({
                        "title": result.get("title", "Search Result"),
                        "content": result.get("snippet", result.get("content", "")),
                        "source": result.get("source", "unknown"),
                        "metadata": result
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to load document: {e}")
                # Still add the result with available content
                documents.append({
                    "title": result.get("title", "Failed Load"),
                    "content": result.get("snippet", result.get("content", "")),
                    "source": "error",
                    "metadata": result,
                    "error": str(e)
                })
        
        state["documents"] = documents
        logger.info(f"Loaded {len(documents)} documents")
        return state
    
    def summarizer_node(self, state: AgentState) -> AgentState:
        """Summarize loaded documents"""
        logger.info("Summarizing documents...")
        
        summaries = []
        
        summarization_prompt = ChatPromptTemplate.from_template("""
        Summarize the following research document in the context of this query: {query}
        
        Document Title: {title}
        Document Source: {source}
        Document Content: {content}
        
        Provide a structured summary including:
        1. Main topic and relevance to the query
        2. Key findings or insights
        3. Methodology (if applicable)
        4. Limitations or considerations
        5. Relevance score (0-1) to the original query
        
        Keep the summary concise but comprehensive.
        """)
        
        for doc in state["documents"]:
            try:
                messages = summarization_prompt.format_messages(
                    query=state["query"],
                    title=doc["title"],
                    source=doc["source"],
                    content=doc["content"][:4000]  # Limit content length
                )
                
                response = self.llm.invoke(messages)
                summary = response.content
                
                summaries.append({
                    "title": doc["title"],
                    "summary": summary,
                    "source": doc["source"],
                    "metadata": doc.get("metadata", {})
                })
                
            except Exception as e:
                logger.error(f"Failed to summarize document {doc['title']}: {e}")
                summaries.append({
                    "title": doc["title"],
                    "summary": f"Error summarizing: {e}",
                    "source": doc["source"],
                    "metadata": doc.get("metadata", {})
                })
        
        state["summaries"] = summaries
        logger.info(f"Generated {len(summaries)} summaries")
        return state
    
    def report_generator_node(self, state: AgentState) -> AgentState:
        """Generate final comprehensive report"""
        logger.info("Generating final report...")
        
        report_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive research report based on the following query and summaries:
        
        Original Query: {query}
        
        Document Summaries:
        {summaries}
        
        Generate a well-structured report that includes:
        1. Executive Summary
        2. Key Findings from the research
        3. Detailed Analysis of each relevant source
        4. Dataset Recommendations (if applicable)
        5. Conclusions and Recommendations
        6. Sources and References
        
        Make sure the report directly addresses the original query and provides actionable insights.
        """)
        
        # Format summaries for the prompt
        formatted_summaries = []
        for i, summary in enumerate(state["summaries"], 1):
            formatted_summaries.append(f"""
            {i}. {summary['title']} (Source: {summary['source']})
            {summary['summary']}
            """)
        
        summaries_text = "\n".join(formatted_summaries)
        
        messages = report_prompt.format_messages(
            query=state["query"],
            summaries=summaries_text
        )
        
        try:
            response = self.llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            final_report = f"Error generating final report: {e}\n\nAvailable summaries:\n{summaries_text}"
        
        state["final_report"] = final_report
        logger.info("Final report generated")
        return state
    
    async def run_research(self, query: str) -> Dict[str, Any]:
        """Run the complete research workflow"""
        logger.info(f"Starting research for query: {query}")
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            plan={},
            search_results=[],
            documents=[],
            summaries=[],
            final_report="",
            iteration_count=0,
            max_iterations=5
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "query": query,
            "final_report": result["final_report"],
            "search_results_count": len(result["search_results"]),
            "documents_count": len(result["documents"]),
            "summaries_count": len(result["summaries"]),
            "iterations": result["iteration_count"]
        }
    
    def run_research_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for research workflow"""
        return asyncio.run(self.run_research(query))

# Example usage and demo
def main():
    """Demo function"""
    import os
    
    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")  # Optional
    
    if not openai_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent
    agent = AIResearchAgent(openai_key, serpapi_key)

    query = "Summarize the three latest papers on LLM safety, recommend one open-source dataset, and analyze publication trends"
    
    print(f"Running research query: {query}")
    print("=" * 80)
    
    # Run research
    result = agent.run_research_sync(query)
    
    print(f"\n\nRESEARCH RESULTS:")
    print(f"Query: {result['query']}")
    print(f"Search Results: {result['search_results_count']}")
    print(f"Documents Processed: {result['documents_count']}")
    print(f"Summaries Generated: {result['summaries_count']}")
    print(f"Iterations: {result['iterations']}")
    print("\n" + "="*80)
    print("FINAL REPORT:")
    print("="*80)
    print(result['final_report'])

# if __name__ == "__main__":
#     main()