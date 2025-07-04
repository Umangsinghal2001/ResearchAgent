import asyncio
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import json

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
import arxiv
import requests
from bs4 import BeautifulSoup

# Structured Output Models
class ThoughtProcess(BaseModel):
    """Model for agent's reasoning process"""
    current_step: str = Field(description="What step the agent is currently on")
    reasoning: str = Field(description="Agent's reasoning about what to do next")
    action_needed: Literal["web_search", "arxiv_search", "both_search", "load_documents", "summarize", "generate_report", "end"] = Field(
        description="What action the agent decides to take next"
    )
    search_query: Optional[str] = Field(description="Search query if search action is needed")

class PaperInfo(BaseModel):
    """Model for paper information"""
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(description="List of authors")
    url: str = Field(description="URL to the paper")
    abstract: str = Field(description="Abstract or summary of the paper")
    year: Optional[int] = Field(description="Publication year")
    venue: Optional[str] = Field(description="Publication venue")

class ResearchReport(BaseModel):
    """Model for final research report"""
    title: str = Field(description="Title of the research report")
    summary: str = Field(description="Overall summary of findings")
    papers: List[PaperInfo] = Field(description="List of important papers found")
    methodology: str = Field(description="How the research was conducted")

# State Model
class AgentState(BaseModel):
    """State of the research agent"""
    user_query: str = Field(description="Original user query")
    messages: List[Any] = Field(default_factory=list, description="Conversation history")
    current_step: str = Field(default="planning", description="Current step in the process")
    thought_process: Optional[ThoughtProcess] = Field(description="Current thought process")
    search_results: Dict[str, Any] = Field(default_factory=dict, description="Search results from different sources")
    papers_found: List[PaperInfo] = Field(default_factory=list, description="Papers found during research")
    final_report: Optional[ResearchReport] = Field(description="Final research report")
    iteration_count: int = Field(default=0, description="Number of iterations")


tavily_tool = TavilySearch(max_results=10)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv for academic papers"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "categories": result.categories
            }
            papers.append(paper)
        
        return json.dumps(papers)
    except Exception as e:
        return f"Error in arXiv search: {str(e)}"
    
    

# Node Functions
def planner_node(state: AgentState) -> AgentState:
    """Planning node that decides what to do next"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create structured output LLM
    structured_llm = llm.with_structured_output(ThoughtProcess)
    
    # System message for planning
    system_msg = """You are a research planning agent. Your job is to think through the user's research query and decide what actions to take.

Available actions:
- web_search: Search the web for general information
- arxiv_search: Search arXiv for academic papers
- both_search: Search both web and arXiv in parallel
- load_documents: Process and load found documents
- summarize: Summarize the collected information
- generate_report: Generate the final research report
- end: End the process

Think step by step about what information you need and what tools would be most effective."""

    # Create messages for the current context
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"User query: {state.user_query}"),
        HumanMessage(content=f"Current step: {state.current_step}"),
        HumanMessage(content=f"Papers found so far: {len(state.papers_found)}"),
        HumanMessage(content=f"Search results available: {list(state.search_results.keys())}")
    ]
    
    # Get structured response
    thought_process = structured_llm.invoke(messages)
    
    # Update state
    state.thought_process = thought_process
    state.current_step = thought_process.current_step
    state.iteration_count += 1
    
    # Add to message history
    state.messages.append({
        "role": "assistant",
        "content": f"Thinking: {thought_process.reasoning}",
        "action": thought_process.action_needed
    })
    
    return state

def search_executor_node(state: AgentState) -> AgentState:
    """Execute search operations based on planner's decision"""
    
    if not state.thought_process:
        return state
    
    action = state.thought_process.action_needed
    query = state.thought_process.search_query or state.user_query
    
    if action == "web_search":
        results = web_search.run(query)
        state.search_results["web"] = results
        state.messages.append({
            "role": "system",
            "content": f"Web search completed for query: {query}"
        })
        
    elif action == "arxiv_search":
        results = arxiv_search.run(query)
        state.search_results["arxiv"] = results
        state.messages.append({
            "role": "system",
            "content": f"arXiv search completed for query: {query}"
        })
        
    elif action == "both_search":
        # Execute both searches
        web_results = web_search.run(query)
        arxiv_results = arxiv_search.run(query)
        
        state.search_results["web"] = web_results
        state.search_results["arxiv"] = arxiv_results
        state.messages.append({
            "role": "system",
            "content": f"Both web and arXiv searches completed for query: {query}"
        })
    
    return state

def document_loader_node(state: AgentState) -> AgentState:
    """Process and structure the search results into paper information"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Process arXiv results
    if "arxiv" in state.search_results:
        try:
            arxiv_data = json.loads(state.search_results["arxiv"])
            for paper_data in arxiv_data:
                paper = PaperInfo(
                    title=paper_data["title"],
                    authors=paper_data["authors"],
                    url=paper_data["url"],
                    abstract=paper_data["summary"][:500] + "..." if len(paper_data["summary"]) > 500 else paper_data["summary"],
                    year=int(paper_data["published"][:4]) if paper_data["published"] else None,
                    venue="arXiv"
                )
                state.papers_found.append(paper)
        except Exception as e:
            state.messages.append({
                "role": "system",
                "content": f"Error processing arXiv results: {str(e)}"
            })
    
    # Process web results (extract paper information)
    if "web" in state.search_results:
        try:
            web_data = json.loads(state.search_results["web"])
            
            # Use LLM to extract paper information from web results
            extraction_prompt = f"""
            Extract academic paper information from the following web search results.
            Focus on finding titles, authors, URLs, and abstracts of important research papers.
            
            Web results: {web_data}
            
            Return the information in a structured format.
            """
            
            # This would need additional processing to extract papers from web results
            # For now, we'll add a placeholder
            state.messages.append({
                "role": "system",
                "content": "Web results processed for paper extraction"
            })
            
        except Exception as e:
            state.messages.append({
                "role": "system",
                "content": f"Error processing web results: {str(e)}"
            })
    
    state.current_step = "document_loading_complete"
    return state

def summarizer_node(state: AgentState) -> AgentState:
    """Summarize the collected papers and prepare for report generation"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create summary of findings
    summary_prompt = f"""
    Based on the user query: "{state.user_query}"
    
    Papers found: {len(state.papers_found)}
    
    Create a brief summary of the research findings and methodology used.
    """
    
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    
    state.messages.append({
        "role": "assistant",
        "content": f"Research summary: {summary_response.content}"
    })
    
    state.current_step = "summarization_complete"
    return state

def report_generator_node(state: AgentState) -> AgentState:
    """Generate the final research report"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ResearchReport)
    
    # Generate structured report
    report_prompt = f"""
    Create a comprehensive research report based on the following information:
    
    User Query: {state.user_query}
    Papers Found: {len(state.papers_found)}
    
    Papers Information:
    {json.dumps([paper.dict() for paper in state.papers_found], indent=2)}
    
    Create a well-structured research report with:
    1. A clear title
    2. An executive summary
    3. Details about each important paper found
    4. Methodology explanation
    """
    
    report = structured_llm.invoke([HumanMessage(content=report_prompt)])
    
    state.final_report = report
    state.current_step = "complete"
    
    return state

# Router function
def should_continue(state: AgentState) -> str:
    """Determine the next step based on current state"""
    
    if not state.thought_process:
        return "planner"
    
    action = state.thought_process.action_needed
    
    if action in ["web_search", "arxiv_search", "both_search"]:
        return "search_executor"
    elif action == "load_documents":
        return "document_loader"
    elif action == "summarize":
        return "summarizer"
    elif action == "generate_report":
        return "report_generator"
    elif action == "end" or state.current_step == "complete":
        return END
    else:
        return "planner"

# Create the graph
def create_research_agent():
    """Create the research agent graph"""
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("search_executor", search_executor_node)
    workflow.add_node("document_loader", document_loader_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("report_generator", report_generator_node)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "planner": "planner",
            "search_executor": "search_executor",
            "document_loader": "document_loader",
            "summarizer": "summarizer",
            "report_generator": "report_generator",
            END: END
        }
    )
    
    # Add edges from other nodes back to planner
    workflow.add_edge("search_executor", "planner")
    workflow.add_edge("document_loader", "planner")
    workflow.add_edge("summarizer", "planner")
    workflow.add_edge("report_generator", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app

# Usage example
async def run_research_agent(query: str, thread_id: str = "research_session_1"):
    """Run the research agent with a given query"""
    
    # Create agent
    agent = create_research_agent()
    
    # Initial state
    initial_state = AgentState(
        user_query=query,
        messages=[],
        current_step="start"
    )
    
    # Configuration for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the agent
    try:
        final_state = await agent.ainvoke(initial_state, config)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"RESEARCH COMPLETED")
        print(f"{'='*50}")
        
        if final_state.final_report:
            print(f"\nTitle: {final_state.final_report.title}")
            print(f"\nSummary: {final_state.final_report.summary}")
            print(f"\nPapers Found: {len(final_state.final_report.papers)}")
            
            for i, paper in enumerate(final_state.final_report.papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors)}")
                print(f"   URL: {paper.url}")
                print(f"   Abstract: {paper.abstract}")
                if paper.year:
                    print(f"   Year: {paper.year}")
        
        print(f"\nTotal iterations: {final_state.iteration_count}")
        
        return final_state
        
    except Exception as e:
        print(f"Error running research agent: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example query
    query = "What are the 5 most important research papers that led to the development of Large Language Models?"
    
    # Run the agent
    result = asyncio.run(run_research_agent(query))
    
    if result:
        print("\nAgent completed successfully!")
    else:
        print("\nAgent failed to complete.")