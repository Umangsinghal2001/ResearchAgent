# AI Research Assistant Agent

A sophisticated multi-tool AI agent built with LangGraph that can handle complex research queries by orchestrating web search, ArXiv search, document processing, and intelligent summarization to generate comprehensive research reports.

## 🚀 Features

- **Dynamic Planning**: Intelligent query analysis and multi-step research planning
- **Multi-Tool Orchestration**: Seamlessly combines web search, ArXiv search, and document processing
- **Adaptive Feedback Loop**: Iteratively refines search strategy based on results (max 5 iterations)
- **Comprehensive Summarization**: AI-powered document analysis and synthesis
- **Professional Reporting**: Generates structured, actionable research reports

## 🏗️ Architecture

The agent follows a sophisticated workflow built on LangGraph:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Planner   │───▶│ Web Searcher │───▶│ Document Loader │
│             │    └──────────────┘    └─────────────────┘
│             │    ┌──────────────┐    ┌─────────────────┐
│ (Analyzes   │───▶│ArXiv Searcher│───▶│   Summarizer    │
│  query &    │    └──────────────┘    └─────────────────┘
│  decides    │                        ┌─────────────────┐
│  actions)   │◀───────────────────────│Report Generator │
└─────────────┘                        └─────────────────┘
```

### Key Components:

1. **Planner Node**: Analyzes queries and determines optimal search strategies
2. **Search Nodes**: Web search (SerpAPI/DuckDuckGo) and ArXiv academic search
3. **Document Loader**: Processes and extracts content from found sources
4. **Summarizer**: AI-powered analysis and synthesis of research materials
5. **Report Generator**: Creates comprehensive, structured final reports

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (required)
- SerpAPI key (optional, will fallback to DuckDuckGo)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-research-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here  # Optional
   ```

   Or export them directly:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export SERPAPI_API_KEY="your_serpapi_key_here"  # Optional
   ```

## 🎯 Usage

### Jupyter Notebook Demo

Run the interactive demo notebook:
```bash
jupyter notebook demo.ipynb
```

## 📝 Example Queries

The agent excels at handling complex, multi-faceted research queries:

### Academic Research
```
"Summarize the three latest papers on LLM safety, recommend one open-source dataset, and analyze publication trends"
```

### Technology Analysis
```
"Find recent papers on transformer architecture improvements and summarize the key innovations"
```


## 🔧 Configuration

### Search Tools
- **Primary**: SerpAPI (requires API key)
- **Academic**: ArXiv API (free, no API key required)

### Iteration Limits
The agent has a built-in safety mechanism with a maximum of 5 search iterations to prevent infinite loops while allowing comprehensive research.

### LLM Configuration
- **Model**: GPT-4o-mini (configurable)
- **Temperature**: 0.1 (for consistent, factual outputs)

## 📊 Output Structure

Each research run returns:

```python
{
    "query": "Original research query",
    "final_report": "Comprehensive structured report",
    "search_results_count": 15,
    "documents_count": 12,
    "summaries_count": 10,
    "iterations": 3
}
```

### Report Sections
1. **Executive Summary**
2. **Key Findings**
3. **Detailed Analysis**
4. **Dataset Recommendations** (when applicable)
5. **Conclusions and Recommendations**
6. **Sources and References**

## 🚦 Error Handling

The agent includes robust error handling:
- **Search failures**: Gracefully falls back to alternative search methods
- **Document loading errors**: Continues with available content
- **API rate limits**: Implements retry logic with exponential backoff
- **Malformed responses**: Provides meaningful error messages
