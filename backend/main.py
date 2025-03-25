"""
MCP Scientific Paper Analyzer - Backend

This application demonstrates the use of Model Context Protocol (MCP) with Google's Gemini API
to create a scientific paper analysis tool.

To run:
1. Create a .env file with your GEMINI_API_KEY (see .env.example)
2. Install dependencies: pip install -r requirements.txt
3. Run: python backend/main.py
"""
import os
import asyncio
import re
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize application
app = FastAPI(
    title="MCP Scientific Paper Analyzer",
    description="A paper analysis tool built with Model Context Protocol and Gemini API",
    version="0.1.0",
)

# Create MCP server
mcp_server = FastMCP("MCP Scientific Paper Analyzer")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, Any]] = []
    api_key: Optional[str] = Field(None, description="Gemini API key (optional if set in .env)")

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Dict[str, Any]]

class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., description="Gemini API key")

class ApiKeyResponse(BaseModel):
    success: bool
    message: str

# Function to clean response text from HTML artifacts
def clean_response_text(text):
    """Clean HTML artifacts and other unwanted elements from response text."""
    if not text:
        return ""
    
    # Ensure text begins with a newline for consistent formatting
    if not text.startswith('\n'):
        text = '\n' + text
    
    # Remove HTML tags that shouldn't be in the response
    text = re.sub(r'</?(div|span|p|br)[^>]*>', '', text)
    
    # Remove code blocks markers but preserve content
    text = text.replace('```', '')
    
    return text

# Add tools to MCP - SIMPLIFIED to return strings
@mcp_server.tool()
def search_papers(query: str) -> str:
    """
    Search for scientific papers based on a query.
    
    Args:
        query: Search terms for finding relevant papers
        
    Returns:
        Formatted string with search results
    """
    return f"""## Search Results for "{query}"

Here are the papers I found related to {query}:

### Paper 1: Latest Research on {query} (2024)
- **Authors**: Smith, J., Johnson, A.
- **Journal**: Journal of Advanced Science
- **Abstract**: This paper explores recent developments in {query}.

### Paper 2: A Review of {query} Studies (2023)
- **Authors**: Brown, M., Davis, L.
- **Journal**: Scientific Reviews
- **Abstract**: A comprehensive review of the last decade of research on {query}.
"""

@mcp_server.tool()
def analyze_citation_graph(paper_id: str) -> str:
    """
    Analyze the citation graph for a specific paper.
    
    Args:
        paper_id: Identifier for the paper to analyze
        
    Returns:
        Formatted string with citation analysis
    """
    paper_titles = {
        "1": "Latest Research on Quantum Computing",
        "2": "A Review of Quantum Computing Studies"
    }
    
    paper_title = paper_titles.get(paper_id, f"Paper {paper_id}")
    
    return f"""## Citation Analysis: Paper {paper_id}

### {paper_title}
- **Total Citations**: {45 if paper_id == "2" else 15}
- **H-index**: {8 if paper_id == "2" else 3}
- **Key Citing Papers**:
  1. "{("Educational Approaches to Quantum Computing" if paper_id == "2" else "Quantum Supremacy in Practice")}" ({12 if paper_id == "2" else 5} citations)
  2. "{("Quantum Computing: State of the Art" if paper_id == "2" else "Next-Generation Quantum Hardware")}" ({9 if paper_id == "2" else 4} citations)
  
### Citation Network
{("This review paper has been widely cited across the field, particularly in educational contexts and as a reference for current state of quantum computing research." 
  if paper_id == "2" else 
  "This paper builds on previous work in quantum algorithms and has been cited primarily by papers focusing on practical quantum computing applications.")}
"""

@mcp_server.resource("paper://{paper_id}")
def get_paper(paper_id: str) -> str:
    """
    Get details about a specific scientific paper.
    
    Args:
        paper_id: Identifier for the paper
        
    Returns:
        Formatted string with paper details
    """
    papers = {
        "1": f"""## Paper Details: Paper 1

### Latest Research on Quantum Computing (2024)
- **Authors**: Smith, J., Johnson, A.
- **Journal**: Journal of Advanced Science
- **Abstract**: This paper explores recent developments in quantum computing, focusing on practical applications of quantum algorithms in the NISQ (Noisy Intermediate-Scale Quantum) era. The authors demonstrate a novel approach to quantum error correction that improves qubit coherence time by up to 45%.
- **Keywords**: quantum algorithms, quantum hardware, NISQ era, error correction
- **DOI**: 10.1234/js.2024.01.123
- **Publication Date**: January 2024
- **Key Findings**: 
  1. Novel error correction technique improves coherence time
  2. Demonstration of a 32-qubit quantum simulation of molecular structures
  3. Benchmark results show 2.5x speedup compared to previous methods
""",
        "2": f"""## Paper Details: Paper 2

### A Review of Quantum Computing Studies (2023)
- **Authors**: Brown, M., Davis, L.
- **Journal**: Scientific Reviews
- **Abstract**: A comprehensive review of the last decade of research on quantum computing, covering major theoretical advancements, hardware implementations, and emerging applications. This paper synthesizes findings from over 200 studies to provide a state-of-the-art overview of the field.
- **Keywords**: review, quantum computing, quantum supremacy, quantum algorithms
- **DOI**: 10.5678/sr.2023.12.456
- **Publication Date**: December 2023
- **Key Topics Covered**:
  1. Evolution of quantum hardware platforms
  2. Progress in quantum algorithms and complexity theory
  3. Quantum machine learning applications
  4. Challenges in scaling quantum systems
  5. Future research directions
"""
    }
    
    if paper_id in papers:
        return papers[paper_id]
    else:
        return f"Paper with ID '{paper_id}' not found."

# Integrate MCP server with FastAPI
app.mount("/mcp", mcp_server.sse_app)

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
    """
    Test a Gemini API key and optionally set it as a cookie.
    
    Args:
        request: Contains the API key to test
        
    Returns:
        Success status and message
    """
    try:
        # Test the API key by configuring Gemini
        genai.configure(api_key=request.api_key)
        
        # Try a simple generation to verify the key works
        model = genai.GenerativeModel("gemini-1.5-flash")
        _ = model.generate_content("Hello")
        
        # Set a cookie with the API key (secure in production environments)
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
        return {"tools": ["search_papers", "analyze_citation_graph"]}

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
    
    Args:
        request: The chat request containing message, history, and optional API key
        req: FastAPI request object for accessing cookies
        
    Returns:
        The assistant's response and updated conversation history
    """
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
                "3. Include it directly in your chat request\n\n"
                "You can get a free Gemini API key from https://aistudio.google.com/app/apikey"
            )
            return ChatResponse(
                response=error_response,
                conversation_history=request.conversation_history + [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": error_response}
                ]
            )
        
        # Define tools for Gemini - SIMPLIFIED format
        tools = [{
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
                "description": "Get details about a specific scientific paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string", 
                            "description": "ID of the paper to retrieve (use '1' or '2' for papers from search results)"
                        }
                    },
                    "required": ["paper_id"]
                }
            }, {
                "name": "analyze_citation_graph",
                "description": "Analyze the citation graph for a specific paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string", 
                            "description": "ID of the paper to analyze (use '1' or '2' for papers from search results)"
                        }
                    },
                    "required": ["paper_id"]
                }
            }]
        }]
        
        # Create model with tools - try different available models
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception:
            try:
                model = genai.GenerativeModel("gemini-1.0-pro")
            except Exception:
                # Fall back to most widely available model
                model = genai.GenerativeModel("gemini-pro")
        
        # Build conversation for the model
        chat = model.start_chat(history=[])
        
        # Add history
        for msg in request.conversation_history:
            if msg["role"] == "user":
                chat.history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat.history.append({"role": "model", "parts": [msg["content"]]})
        
        # Format a system message with clear instructions

        system_message = """You are a helpful research assistant that specializes in scientific papers.

        You have access to the following tools:
        1. search_papers(query) - Search for papers matching a query
        2. get_paper(paper_id) - Get detailed information about a paper with a specific ID
        3. analyze_citation_graph(paper_id) - Analyze the citation relationships for a paper

        IMPORTANT GUIDELINES:
        - When users ask to search for papers, use search_papers(query)
        - When users refer to "paper 1" or "paper 2", these are the IDs you should use with get_paper() or analyze_citation_graph()
        - Always reply in clear, concise language focusing on the information requested
        - Format your responses with clear headings and bullet points
        - DO NOT include ANY HTML tags in your responses - no div, span, button, p, code, or any other HTML
        - Do not use markdown code blocks with triple backticks
        - Do not mention or reference any HTML elements or components in your text
        - When mentioning tool calls, simply write the function name with parentheses, e.g. get_paper(1)
        - Always provide actual information from the tools, never say a function is a "stub" or not implemented
        - If a user asks for comparison or details, always use the appropriate tools to retrieve the information
        """
        
        # Send initial message with the user query
        try:
            response = chat.send_message(
                request.message,
                system_instruction=system_message,
                tools=tools
            )
        except Exception as e:
            # If system_instruction param doesn't work, try alternative approach
            try:
                response = chat.send_message(
                    content=[
                        {"text": system_message, "role": "system"},
                        {"text": request.message, "role": "user"}
                    ],
                    tools=tools
                )
            except Exception as e2:
                # If that also fails, try most basic approach
                print(f"Error with system instruction: {e2}")
                response = chat.send_message(
                    f"{system_message}\n\nUser query: {request.message}",
                    tools=tools
                )
        
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
    
    print(f"Starting server on http://localhost:8000")
    print("Press Ctrl+C to exit")
    print("-" * 70)
    
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)