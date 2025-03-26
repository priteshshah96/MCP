"""
MCP Scientific Paper Analyzer - Backend

This application demonstrates the use of Model Context Protocol (MCP) with Google's Gemini API
to create a scientific paper analysis tool with Semantic Scholar integration.

To run:
1. Create a .env file with your GEMINI_API_KEY or paste it in the app
2. Install dependencies: pip install -r requirements.txt or use UV which is way faster
2a. uv pip install -r requirements.txt
3. Run: python backend/main.py
"""
import os
import re
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# We load env
load_dotenv()

# Initialize application
app = FastAPI(
    title="Scientific Paper Analyzer",
    description="A paper analysis tool built with Model Context Protocol and Gemini API",
    version="0.1.0",
)

# Create MCP server, this will create a MCP server , check modelcontextprotocol.io
mcp_server = FastMCP("Scientific Paper Analyzer")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for this demonstartion(strictly for demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We store searches to refer the paperid
last_search_results = []

# request and response model
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, Any]] = []
    api_key: Optional[str] = Field(None, description="Gemini API key (This is optional if set in .env)")

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Dict[str, Any]]

class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., description="Gemini API key")

class ApiKeyResponse(BaseModel):
    success: bool
    message: str

# Cleaning html from the responses if any
def clean_response_text(text):
    """Clean HTML  and other unwanted elements from output/response text."""
    if not text:
        return ""
    
    # Ensure text begins with a newline for consistent formatting
    if not text.startswith('\n'):
        text = '\n' + text
        text = re.sub(r'</?(div|span|p|br)[^>]*>', '', text)
        text = text.replace('```', '')
    
    return text

# From here we start defining our mcp server tool
@mcp_server.tool()
def search_papers(query: str) -> str:
    """
    Search for scientific papers based on a query using Semantic Scholar API.
    """
    global last_search_results
    
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    fields = [
        "title", "authors", "year", "abstract", 
        "venue", "citationCount", "url"
    ]
    
    params = {
        "query": query,
        "limit": 5, # can increase the number of papers to fetch
        "fields": ",".join(fields)
    }
    
    headers = {"User-Agent": "Scientific Paper Analyzer/1.0"}
    
    try:
        response = requests.get(api_url, params=params, headers=headers)
        
        if response.status_code == 200:
            search_results = response.json()
            papers = search_results.get("data", [])
            
            result = f"## Search Results for \"{query}\"\n\n"
            
            if not papers:
                return f"{result}No papers found for this query. Try different search terms."
            
            result += f"I found {len(papers)} papers related to {query}:\n\n"
            
            # Clear previous results, so that we can start fresh
            last_search_results = []
            
            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "Untitled")
                year = paper.get("year", "Unknown year")
                
                # Format for authors
                authors = paper.get("authors", [])
                author_names = []
                for author in authors:
                    if isinstance(author, dict):
                        name = author.get("name", "")
                        if name:
                            author_names.append(name)
                    elif isinstance(author, str):
                        author_names.append(author)
                
                author_text = ", ".join(author_names) if author_names else "Unknown authors"
                venue = paper.get("venue", "Unknown venue")
                citation_count = paper.get("citationCount", 0)
                
                abstract = paper.get("abstract", "No abstract available.")
                if abstract and len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                
                url = paper.get("url", "")
                
                # Extract paper ID from URL
                paper_id = url.split('/')[-1] if url else None
                
                # Store this paper for later usage
                last_search_results.append({
                    "reference_id": f"P{i}",
                    "actual_id": paper_id,
                    "title": title,
                    "url": url
                })
                
                # Format result
                result += f"### Paper {i}: {title} ({year})\n"
                result += f"- **ID**: P{i}\n"
                result += f"- **Authors**: {author_text}\n"
                result += f"- **Venue**: {venue}\n"
                result += f"- **Citations**: {citation_count}\n"
                result += f"- **Abstract**: {abstract}\n"
                if url:
                    result += f"- **URL**: {url}\n"
                result += "\n"
            
            return result
        
        else:
            return f"Error searching for papers: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"An error occurred while searching for papers: {str(e)}"

@mcp_server.resource("paper://{paper_id}")
def get_paper(paper_id: str) -> str:
    """
    Get details about a specific scientific paper.
    """
    global last_search_results
    
    # Handles reference IDs (P1, P2, etc.)
    if paper_id.startswith("P"):
        try:
            index = int(paper_id[1:]) - 1
            
            if not last_search_results or index < 0 or index >= len(last_search_results):
                return f"Paper ID {paper_id} is a reference to search results. Please search for papers first."
            
            # Get the actual paper ID
            paper_info = last_search_results[index]
            actual_paper_id = paper_info["actual_id"]
            
            if not actual_paper_id:
                return f"No valid paper ID found for {paper_id}."
            
            # Use the actual ID instead
            paper_id = actual_paper_id
            
        except ValueError:
            return f"Invalid paper reference: {paper_id}. Expected format is P followed by a number (e.g., P1)."
    
    # Try to fetch from Semantic Scholar
    try:
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        
        fields = ["title", "authors", "year", "abstract", "venue", "citationCount", "url", "references"]
        params = {"fields": ",".join(fields)}
        
        headers = {"User-Agent": "MCP Scientific Paper Analyzer/1.0"}
        
        response = requests.get(api_url, params=params, headers=headers)
        
        if response.status_code == 200:
            paper = response.json()
            
            # Extract paper details
            title = paper.get("title", "Untitled")
            year = paper.get("year", "Unknown year")
            venue = paper.get("venue", "Unknown venue")
            citation_count = paper.get("citationCount", 0)
            abstract = paper.get("abstract", "No abstract available.")
            url = paper.get("url", "")
            
            # Format authors
            authors = paper.get("authors", [])
            author_names = []
            for author in authors:
                if isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        author_names.append(name)
                elif isinstance(author, str):
                    author_names.append(author)
            
            author_text = ", ".join(author_names) if author_names else "Unknown authors"
            
            # Format result
            result = f"## Paper Details: {title}\n\n"
            result += f"- **Authors**: {author_text}\n"
            result += f"- **Year**: {year}\n"
            result += f"- **Venue**: {venue}\n"
            result += f"- **Citations**: {citation_count}\n"
            result += f"- **Abstract**: {abstract}\n"
            if url:
                result += f"- **URL**: {url}\n"
            
            # Add references if available
            references = paper.get("references", [])
            if references and len(references) > 0:
                result += f"\n### References (showing up to 5 of {len(references)})\n"
                for i, ref in enumerate(references[:5], 1):
                    ref_title = ref.get("title", "Untitled reference")
                    ref_year = ref.get("year", "")
                    ref_year_str = f" ({ref_year})" if ref_year else ""
                    result += f"{i}. {ref_title}{ref_year_str}\n"
            
            return result
        
        elif response.status_code == 404:
            return f"Paper with ID '{paper_id}' not found in the Semantic Scholar database."
        
        else:
            return f"Error retrieving paper: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"An error occurred while retrieving paper details: {str(e)}"

# this is currently not working properly, might need to connect to some other tool
@mcp_server.tool()
def analyze_citation_graph(paper_id: str) -> str:
    """
    Analyze the citation graph for a specific paper.
    Works with both P-references (P1, P2) and direct paper IDs.
    """
    global last_search_results
    
    # Handle reference IDs (P1, P2, etc.)
    if paper_id.startswith("P"):
        try:
            index = int(paper_id[1:]) - 1
            
            if not last_search_results:
                return "No search results available. Please search for papers first before analyzing citations."
            
            if index < 0 or index >= len(last_search_results):
                return f"Invalid paper reference: {paper_id}. Please use a valid paper ID from the search results."
            
            # Get the actual paper ID
            paper_info = last_search_results[index]
            actual_paper_id = paper_info["actual_id"]
            paper_title = paper_info["title"]
            
            if not actual_paper_id:
                return f"Cannot analyze citations for {paper_id} ({paper_title}). No valid paper ID found."
            
            # Use the actual ID instead
            paper_id = actual_paper_id
            
        except ValueError:
            return f"Invalid paper reference format: {paper_id}. Expected format is P followed by a number (e.g., P1)."
    
    # Try to fetch the paper and its citations
    try:
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        
        fields = ["title", "year", "citationCount", "citations", "references"]
        params = {"fields": ",".join(fields)}
        
        headers = {"User-Agent": "MCP Scientific Paper Analyzer/1.0"}
        
        response = requests.get(api_url, params=params, headers=headers)
        
        if response.status_code == 200:
            paper = response.json()
            
            # Extract paper information
            title = paper.get("title", "Untitled")
            year = paper.get("year", "Unknown year")
            citation_count = paper.get("citationCount", 0)
            
            # Get citations and references data
            citations = paper.get("citations", [])
            references = paper.get("references", [])
            
            # Format the result
            result = f"## Citation Analysis: {title} ({year})\n\n"
            result += f"- **Total Citations**: {citation_count}\n"
            result += f"- **References**: {len(references)}\n\n"
            
            # Analyze papers that cite this paper
            if citations and len(citations) > 0:
                # Sort citations by count if available
                sorted_citations = sorted(
                    citations, 
                    key=lambda x: x.get("citationCount", 0) if isinstance(x, dict) else 0, 
                    reverse=True
                )
                
                result += f"### Top Citing Papers (showing up to 5 of {len(citations)})\n"
                
                for i, citation in enumerate(sorted_citations[:5], 1):
                    cit_title = citation.get("title", "Untitled")
                    cit_year = citation.get("year", "")
                    cit_count = citation.get("citationCount", "N/A")
                    
                    result += f"{i}. **{cit_title}** ({cit_year}) - Cited {cit_count} times\n"
            else:
                result += "This paper has not been cited yet according to Semantic Scholar.\n"
            
            # Add reference analysis
            if references and len(references) > 0:
                result += f"\n### Key References (showing up to 5 of {len(references)})\n"
                
                # Sort references by citation count if available
                sorted_refs = sorted(
                    references,
                    key=lambda x: x.get("citationCount", 0) if isinstance(x, dict) else 0,
                    reverse=True
                )
                
                for i, ref in enumerate(sorted_refs[:5], 1):
                    ref_title = ref.get("title", "Untitled reference")
                    ref_year = ref.get("year", "")
                    ref_count = ref.get("citationCount", "N/A")
                    ref_year_str = f" ({ref_year})" if ref_year else ""
                    
                    result += f"{i}. **{ref_title}**{ref_year_str} - Cited {ref_count} times\n"
            
            return result
        
        elif response.status_code == 404:
            return f"Paper with ID '{paper_id}' not found in the Semantic Scholar database."
        
        else:
            return f"Error analyzing citations: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"An error occurred while analyzing citation graph: {str(e)}"

# This tool extracts the paper from the last search results
@mcp_server.tool()
def extract_paper_ids() -> str:
    """
    Extracts actual paper IDs from search results for easier citation analysis.
    """
    global last_search_results
    
    if not last_search_results:
        return "No search results available. Please search for papers first using search_papers()."
    
    result = "## Extracted Paper IDs for Citation Analysis\n\n"
    result += "Use these IDs with the analyze_citation_graph function:\n\n"
    
    for paper in last_search_results:
        ref_id = paper["reference_id"]
        actual_id = paper["actual_id"]
        title = paper["title"]
        
        result += f"- **{ref_id}**: {title}\n"
        result += f"  - Reference ID: `{ref_id}`\n"
        result += f"  - Actual Paper ID: `{actual_id}`\n"
        result += f"  - To analyze: `analyze_citation_graph(\"{ref_id}\")`\n\n"
    
    result += "\nYou can now use either the reference ID (e.g., P1) or the actual paper ID with analyze_citation_graph()."
    
    return result

# We integrate MCP server with FastAPI
app.mount("/mcp", mcp_server.sse_app)

# Helper function to update the tool for Gemini
def get_tool_definitions():
    """Get the updated tool definitions for Gemini API"""
    return [{
        "function_declarations": [{
            "name": "search_papers",
            "description": "Search for scientific papers based on a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "Search terms for finding relevant papers"
                    }
                },
                "required": ["query"]
            }
        }, {
            "name": "get_paper",
            "description": "Get details about a specific scientific paper (works with both P1, P2 references and paper IDs)",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string", 
                        "description": "ID of the paper to retrieve (can be P1, P2, etc. from search results)"
                    }
                },
                "required": ["paper_id"]
            }
        }, {
            "name": "analyze_citation_graph",
            "description": "Analyze the citation graph for a specific paper (works with both P1, P2 references and paper IDs)",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string", 
                        "description": "ID of the paper to analyze (can be P1, P2, etc. from search results)"
                    }
                },
                "required": ["paper_id"]
            }
        }, {
            "name": "extract_paper_ids",
            "description": "Extract and display actual paper IDs from the most recent search results",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }]
    }]

# Helper to configure Gemini API with the provided key or from environment
def configure_gemini_api(api_key: Optional[str] = None):
    """Configure the Gemini API with the provided key or from environment."""
    # Use the provided key or fall back to environment variable
    key_to_use = api_key or os.getenv("GEMINI_API_KEY")
    
    if not key_to_use:
        raise ValueError("No Gemini API key provided. Set GEMINI_API_KEY in .env or provide it in the request.")
    
    # Configure Gemini with the key
    genai.configure(api_key=key_to_use)
    return key_to_use

# Routes
@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"message": "MCP Scientific Paper Analyzer API is running"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/set-api-key", response_model=ApiKeyResponse)
async def set_api_key(request: ApiKeyRequest, response: Response):
    """Test a Gemini API key and optionally set it as a cookie."""
    try:
        # Test the API key by configuring Gemini
        genai.configure(api_key=request.api_key)
        
        # Try a simple generation to verify the key works
        model = genai.GenerativeModel("gemini-2.0-flash")
        _ = model.generate_content("Hello")
        
        # Set a cookie with the API key
        response.set_cookie(
            key="gemini_api_key",
            value=request.api_key,
            httponly=True,
            max_age=3600,  # 1 hour
            samesite="lax"
        )
        
        return {"success": True, "message": "API key is valid and has been set"}
    
    except Exception as e:
        return {"success": False, "message": f"Invalid API key: {str(e)}"}

@app.get("/api/tools")
async def list_tools():
    """List all available MCP tools."""
    try:
        tools = mcp_server.list_tools()
        # Check if the result is awaitable
        if hasattr(tools, "__await__"):
            tools = await tools
        return {"tools": tools}
    except Exception as e:
        print(f"Error listing tools: {e}")
        return {"tools": ["search_papers", "analyze_citation_graph", "extract_paper_ids"]}

@app.get("/api/resources")
async def list_resources():
    """List all available MCP resources."""
    try:
        resources = mcp_server.list_resource_templates()
        # Check if the result is awaitable
        if hasattr(resources, "__await__"):
            resources = await resources
        return {"resources": resources}
    except Exception as e:
        print(f"Error listing resources: {e}")
        return {"resources": ["paper://{paper_id}"]}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Process a chat message using Gemini with MCP tool integration.
    """
    global last_search_results
    
    try:
        # Get API key from request, cookie, or environment in that order
        api_key = request.api_key
        
        if not api_key:
            # Try to get from cookie
            api_key = req.cookies.get("gemini_api_key")
        
        # Configure Gemini API with the key
        try:
            configure_gemini_api(api_key)
        except ValueError as e:
            # If no API key is available, return a friendly error message
            error_response = (
                "I need a Gemini API key to work. You can provide one in three ways:\n\n"
                "1. Add it to the .env file as GEMINI_API_KEY=your_key_here\n"
                "2. Enter it in the API Key field in the sidebar\n"
                "3. Include it directly in your chat request, (but you shouldn't use this, ever!) \n\n"
                "You can get a free Gemini API key from https://aistudio.google.com/app/apikey"
            )
            return ChatResponse(
                response=error_response,
                conversation_history=request.conversation_history + [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": error_response}
                ]
            )
        
     # upadted tool definition
        tools = get_tool_definitions()
        
        # Diffrent models 
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception:
            try:
                model = genai.GenerativeModel("gemini-1.0-pro")
            except Exception:
                # Fall back 
                model = genai.GenerativeModel("gemini-pro")
        
        # Build conversation for the model
        chat = model.start_chat(history=[])
        
        # Add history
        for msg in request.conversation_history:
            if msg["role"] == "user":
                chat.history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat.history.append({"role": "model", "parts": [msg["content"]]})
        
        # System instruction that you can play along
        system_message = """You are a helpful research assistant that specializes in scientific papers.

You have access to the following tools:
1. search_papers(query) - Search for papers matching a query using the Semantic Scholar database
2. get_paper(paper_id) - Get detailed information about a paper with a specific ID
3. analyze_citation_graph(paper_id) - Analyze the citation relationships for a paper
4. extract_paper_ids() - Extract and display actual paper IDs from the most recent search results

IMPORTANT GUIDELINES:
- When users ask to search for papers, use search_papers(query)
- Paper IDs start with 'P' for search results, e.g. P1, P2
- You can now use P1, P2, etc. directly with analyze_citation_graph() - this will automatically use the correct paper ID
- Always reply in clear, concise language focusing on the information requested
- Format your responses with clear headings and bullet points
- If a user asks for comparison or details, always use the appropriate tools to retrieve the information
"""
        
        # Intial message with users input/query
        try:
            response = chat.send_message(
                request.message,
                tools=tools
            )
        except Exception as e:
            # If fails, try with system message in content
            try:
                response = chat.send_message(
                    f"{system_message}\n\nUser query: {request.message}",
                    tools=tools
                )
            except Exception as e2:
                print(f"Error sending message: {e2}")
                response = chat.send_message(request.message)
        
        # Process any tool calls
        handled_tool_call = False
        all_tool_results = []
        
        try:
            # Extract function calls based on response structure
            function_calls = []
            
            if hasattr(response, "candidates"):
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call"):
                                function_calls.append(part.function_call)
            elif hasattr(response, "function_call"):
                function_calls.append(response.function_call)
            elif hasattr(response, "parts"):
                for part in response.parts:
                    if hasattr(part, "function_call"):
                        function_calls.append(part.function_call)
            
            # Process each function call
            for function_call in function_calls:
                handled_tool_call = True
                function_name = function_call.name
                
                # Extract arguments safely
                try:
                    args = function_call.args
                except:
                    args = {}
                
                # Call the appropriate function based on name
                if function_name == "search_papers":
                    query = args.get("query", "")
                    result = search_papers(query)
                    all_tool_results.append(result)
                    response = chat.send_message(result)
                
                elif function_name == "get_paper":
                    paper_id = args.get("paper_id", "").replace("'", "").replace('"', "")
                    result = get_paper(paper_id)
                    all_tool_results.append(result)
                    response = chat.send_message(result)
                
                elif function_name == "analyze_citation_graph":
                    paper_id = args.get("paper_id", "").replace("'", "").replace('"', "")
                    result = analyze_citation_graph(paper_id)
                    all_tool_results.append(result)
                    response = chat.send_message(result)
                    
                elif function_name == "extract_paper_ids":
                    result = extract_paper_ids()
                    all_tool_results.append(result)
                    response = chat.send_message(result)
        
        except Exception as e:
            print(f"Error processing tool calls: {e}")
            # Continue without tool results if there was an error
        
        # Get the final response text
        try:
            response_text = response.text
        except AttributeError:
            # Handle different response structures
            try:
                if hasattr(response, "candidates") and response.candidates:
                    response_text = response.candidates[0].content.parts[0].text
                elif hasattr(response, "parts") and response.parts:
                    response_text = response.parts[0].text
                else:
                    response_text = "I encountered an issue processing your request. Please try again."
            except Exception:
                response_text = "I encountered an issue processing your request. Please try again."
        
        # If we handled a tool call but no results were returned to the model,
        # append the tool results directly to the response
        if handled_tool_call and all_tool_results and not any(result in response_text for result in all_tool_results):
            response_text += "\n\n" + "\n\n".join(all_tool_results)
        
        # Clean up any HTML artifacts that might appear in the response
        response_text = clean_response_text(response_text)
        
        # Update conversation history
        new_history = request.conversation_history.copy()
        new_history.append({"role": "user", "content": request.message})
        new_history.append({"role": "assistant", "content": response_text})
        
        return ChatResponse(
            response=response_text,
            conversation_history=new_history
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        error_message = f"\nI encountered an error: {str(e)}\n\nPlease check your API key and try again."
        
        return ChatResponse(
            response=error_message,
            conversation_history=request.conversation_history + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": error_message}
            ]
        )

if __name__ == "__main__":
    import uvicorn
    
    # Start message
    print("=" * 70)
    print(" MCP Scientific Paper Analyzer Backend")
    print("=" * 70)
    
    # Check if API key is configured
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: No Gemini API key found in environment variables")
        print("   Users will need to provide their own API key via the frontend")
        print("   Get a key at: https://aistudio.google.com/app/apikey\n")
    else:
        print("\n✅ Gemini API key found in environment variables\n")
    
    print("✅ Enhanced citation analysis: Can now use P1, P2, etc. with analyze_citation_graph")
    print("✅ Added extract_paper_ids tool to make paper IDs more accessible")
    
    print(f"\nStarting server on http://localhost:8000")
    print("Press Ctrl+C to exit")
    print("-" * 70)
    
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)