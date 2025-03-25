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
import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

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

# Scientific domains for better paper generation
SCIENTIFIC_DOMAINS = {
    "Physics": ["Quantum Physics", "Astrophysics", "Particle Physics", "Condensed Matter", "Nuclear Physics"],
    "Biology": ["Molecular Biology", "Genetics", "Neuroscience", "Ecology", "Microbiology"],
    "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Biochemistry", "Materials Science", "Physical Chemistry"],
    "Computer Science": ["Artificial Intelligence", "Machine Learning", "Computer Vision", "Natural Language Processing", "Quantum Computing"],
    "Medicine": ["Immunology", "Oncology", "Cardiology", "Neurology", "Epidemiology"],
    "Environmental Science": ["Climate Change", "Ecology", "Conservation", "Renewable Energy", "Sustainability"],
    "Mathematics": ["Number Theory", "Algebra", "Topology", "Applied Mathematics", "Combinatorics"],
    "Engineering": ["Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Chemical Engineering", "Biomedical Engineering"]
}

# Paper database (in-memory for demo)
PAPERS_DB = {
    # Nuclear Energy Papers
    "NE1": {
        "id": "NE1",
        "title": "Recent Advances in Nuclear Fusion Containment",
        "authors": ["Takahashi, M.", "Johnson, R.", "Kuznetsov, A."],
        "journal": "Journal of Nuclear Engineering",
        "abstract": "This paper explores recent technological advances in magnetic containment systems for nuclear fusion reactions, with particular focus on improvements to the tokamak design that have led to longer plasma containment times and higher energy efficiency.",
        "year": 2023,
        "keywords": ["nuclear fusion", "tokamak", "magnetic containment", "plasma physics"],
        "doi": "10.1234/jne.2023.05.127",
        "citations": 24,
        "related_papers": ["NE2", "NE4"]
    },
    "NE2": {
        "id": "NE2",
        "title": "Safety Protocols in Modern Nuclear Energy Facilities",
        "authors": ["Smith, L.", "Garcia, C.", "Wong, H."],
        "journal": "International Journal of Energy Safety",
        "abstract": "A comprehensive review of safety protocols implemented in nuclear energy facilities over the past decade, analyzing their effectiveness in preventing incidents and comparing international regulatory frameworks.",
        "year": 2022,
        "keywords": ["nuclear safety", "regulatory compliance", "risk assessment", "emergency protocols"],
        "doi": "10.5678/ijes.2022.11.089",
        "citations": 37,
        "related_papers": ["NE3", "NE5"]
    },
    "NE3": {
        "id": "NE3",
        "title": "Environmental Impact Assessment of Nuclear Waste Storage Solutions",
        "authors": ["Patel, S.", "Müller, J.", "Dubois, F."],
        "journal": "Environmental Science and Technology",
        "abstract": "This study evaluates the long-term environmental impact of various nuclear waste storage solutions, including deep geological repositories and dry cask storage. The paper presents a multi-criteria analysis incorporating geological stability, radiation containment, and monitoring capabilities.",
        "year": 2023,
        "keywords": ["nuclear waste", "environmental impact", "geological repositories", "radiation containment"],
        "doi": "10.9012/est.2023.02.043",
        "citations": 18,
        "related_papers": ["NE2", "NE5"]
    },
    "NE4": {
        "id": "NE4",
        "title": "Next-Generation Small Modular Reactors: Design and Economics",
        "authors": ["Lee, J.", "Novak, P.", "Anderson, T."],
        "journal": "Energy Policy and Economics",
        "abstract": "An analysis of the economic viability of small modular reactors (SMRs) as compared to traditional nuclear plants and other clean energy sources. The paper explores design innovations, deployment strategies, and potential market adoption scenarios.",
        "year": 2024,
        "keywords": ["small modular reactors", "nuclear economics", "energy policy", "clean energy"],
        "doi": "10.3456/epe.2024.01.018",
        "citations": 9,
        "related_papers": ["NE1", "NE5"]
    },
    "NE5": {
        "id": "NE5",
        "title": "Public Perception and Acceptance of Nuclear Energy Post-Fukushima",
        "authors": ["Hernandez, E.", "Ivanova, M.", "Okamoto, T."],
        "journal": "Risk Analysis and Society",
        "abstract": "This paper examines the evolution of public perception and acceptance of nuclear energy in various countries following the Fukushima Daiichi nuclear disaster. Using longitudinal survey data, the authors identify key factors influencing risk perception and propose strategies for science communication.",
        "year": 2022,
        "keywords": ["risk perception", "nuclear accidents", "public opinion", "science communication"],
        "doi": "10.7890/ras.2022.08.112",
        "citations": 42,
        "related_papers": ["NE2", "NE3"]
    },
    
    # Quantum Computing Papers
    "QC1": {
        "id": "QC1",
        "title": "Practical Quantum Error Correction in NISQ Devices",
        "authors": ["Smith, J.", "Johnson, A.", "Zhao, L."],
        "journal": "Journal of Advanced Quantum Computing",
        "abstract": "This paper explores recent developments in quantum computing, focusing on practical applications of quantum algorithms in the NISQ (Noisy Intermediate-Scale Quantum) era. The authors demonstrate a novel approach to quantum error correction that improves qubit coherence time by up to 45%.",
        "year": 2024,
        "keywords": ["quantum algorithms", "quantum hardware", "NISQ era", "error correction"],
        "doi": "10.1234/jaqc.2024.01.123",
        "citations": 15,
        "related_papers": ["QC2", "QC4"]
    },
    "QC2": {
        "id": "QC2",
        "title": "A Comprehensive Review of Quantum Computing Progress",
        "authors": ["Brown, M.", "Davis, L.", "Thompson, R."],
        "journal": "Scientific Reviews in Computing",
        "abstract": "A comprehensive review of the last decade of research on quantum computing, covering major theoretical advancements, hardware implementations, and emerging applications. This paper synthesizes findings from over 200 studies to provide a state-of-the-art overview of the field.",
        "year": 2023,
        "keywords": ["review", "quantum computing", "quantum supremacy", "quantum algorithms"],
        "doi": "10.5678/src.2023.12.456",
        "citations": 45,
        "related_papers": ["QC1", "QC3"]
    }
}

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

# Function to generate paper IDs for a domain
def generate_paper_id(domain_prefix):
    """Generate a unique paper ID for a given domain prefix."""
    existing_ids = [pid for pid in PAPERS_DB.keys() if pid.startswith(domain_prefix)]
    next_num = len(existing_ids) + 1
    return f"{domain_prefix}{next_num}"

# Function to generate realistic paper data
def generate_paper_data(query, domain_prefix="GEN"):
    """Generate realistic paper data based on a query."""
    # Determine the domain and subdomain
    domain = "Computer Science"  # Default domain
    subdomain = "Artificial Intelligence"  # Default subdomain
    
    # Try to map the query to a domain
    for d, subdomains in SCIENTIFIC_DOMAINS.items():
        for sd in subdomains:
            if sd.lower() in query.lower() or d.lower() in query.lower():
                domain = d
                subdomain = sd
                break
    
    # Get current year
    current_year = datetime.now().year
    
    # Generate paper ID
    paper_id = generate_paper_id(domain_prefix)
    
    # Generate author names
    surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", 
                "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", 
                "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", 
                "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", 
                "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", 
                "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts", "Chen", "Kumar", "Singh",
                "Patil", "Muller", "Schmidt", "Fischer", "Weber", "Schneider", "Meyer", "Wagner", "Zhou",
                "Wang", "Liu", "Zhang", "Li", "Khan", "Ivanov", "Smirnov", "Petrov", "Kuznetsov"]
    
    initials = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Generate 2-3 authors
    num_authors = random.randint(2, 3)
    authors = []
    
    for _ in range(num_authors):
        surname = random.choice(surnames)
        initial = random.choice(initials)
        authors.append(f"{surname}, {initial}.")
    
    # Journals by domain
    journals = {
        "Physics": ["Physical Review Letters", "Journal of Physics", "Nature Physics", "Physics Today", "European Physical Journal"],
        "Biology": ["Cell", "Nature Biotechnology", "PLOS Biology", "Molecular Biology and Evolution", "Journal of Theoretical Biology"],
        "Chemistry": ["Journal of the American Chemical Society", "Chemical Science", "Angewandte Chemie", "Chemistry - A European Journal"],
        "Computer Science": ["Journal of Machine Learning Research", "IEEE Transactions on Pattern Analysis", "Communications of the ACM", "Journal of Artificial Intelligence Research"],
        "Medicine": ["The Lancet", "New England Journal of Medicine", "JAMA", "Nature Medicine", "BMJ"],
        "Environmental Science": ["Nature Climate Change", "Environmental Science & Technology", "Global Environmental Change", "Conservation Biology"],
        "Mathematics": ["Annals of Mathematics", "Journal of the American Mathematical Society", "Inventiones Mathematicae", "Mathematical Programming"],
        "Engineering": ["IEEE Transactions on Engineering", "Journal of Engineering Design", "Environmental Engineering Science"]
    }
    
    journal = random.choice(journals.get(domain, journals["Computer Science"]))
    
    # Generate a title related to the query and domain
    capitalized_query = " ".join(word.capitalize() for word in query.split())
    title_templates = [
        f"Recent Advances in {capitalized_query} Research",
        f"A Novel Approach to {capitalized_query} Using {subdomain} Techniques",
        f"Comprehensive Analysis of {capitalized_query} Methods",
        f"The Future of {capitalized_query}: A {subdomain} Perspective",
        f"{capitalized_query}: Challenges and Opportunities",
        f"Optimizing {capitalized_query} Systems through {subdomain}",
        f"Understanding {capitalized_query} Dynamics in Modern {domain}"
    ]
    
    title = random.choice(title_templates)
    
    # Generate publication year (last 3 years)
    year = random.randint(current_year - 3, current_year)
    
    # Generate citation count (more recent papers have fewer citations)
    max_citations = 100 - ((current_year - year) * 30)
    citations = random.randint(5, max(10, max_citations))
    
    # Generate DOI
    doi_prefix = "10." + str(random.randint(1000, 9999))
    doi_suffix = journal.lower().replace(" ", "")[:3] + "." + str(year) + "." + str(random.randint(10, 99)) + "." + str(random.randint(100, 999))
    doi = doi_prefix + "/" + doi_suffix
    
    # Generate keywords based on query and domain
    domain_keywords = {
        "Physics": ["quantum mechanics", "thermodynamics", "relativity", "electromagnetism", "optics"],
        "Biology": ["genomics", "proteomics", "cell signaling", "evolution", "ecology"],
        "Chemistry": ["catalysis", "synthesis", "spectroscopy", "molecular dynamics", "polymers"],
        "Computer Science": ["algorithms", "data structures", "neural networks", "optimization", "machine learning"],
        "Medicine": ["clinical trials", "therapeutics", "diagnostics", "pathology", "epidemiology"],
        "Environmental Science": ["climate modeling", "biodiversity", "pollution", "conservation", "ecosystem"],
        "Mathematics": ["theorem", "proof", "algorithm", "optimization", "topology"],
        "Engineering": ["design", "simulation", "control systems", "materials", "efficiency"]
    }
    
    # Generate keywords by combining query keywords with domain keywords
    query_keywords = [q.lower() for q in query.split() if len(q) > 3]
    all_keywords = query_keywords + random.sample(domain_keywords.get(domain, domain_keywords["Computer Science"]), min(3, len(domain_keywords.get(domain, domain_keywords["Computer Science"]))))
    keywords = list(set(all_keywords))  # Remove duplicates
    
    # Generate an abstract
    abstract_templates = [
        f"This paper explores recent developments in {query}, focusing on applications within {subdomain}. We present a novel approach that improves traditional methods by {random.randint(20, 50)}% in terms of efficiency and accuracy.",
        
        f"A comprehensive review of {query} research over the past {random.randint(3, 10)} years, with particular emphasis on advancements in {subdomain}. This study synthesizes findings from over {random.randint(50, 200)} publications to provide a state-of-the-art overview.",
        
        f"This study presents a new methodology for addressing challenges in {query} using techniques from {subdomain}. Our experiments demonstrate significant improvements over baseline approaches, with performance gains of {random.randint(15, 40)}% on standard benchmarks.",
        
        f"We introduce a novel framework for {query} that leverages recent advances in {subdomain}. Through extensive evaluation on {random.randint(3, 8)} datasets, we demonstrate that our approach outperforms existing methods by a substantial margin while requiring fewer computational resources."
    ]
    
    abstract = random.choice(abstract_templates)
    
    # Generate related papers
    all_papers = list(PAPERS_DB.keys())
    related_papers = []
    if all_papers:
        num_related = min(random.randint(1, 3), len(all_papers))
        related_papers = random.sample(all_papers, num_related)
    
    # Create paper data
    paper_data = {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "journal": journal,
        "abstract": abstract,
        "year": year,
        "keywords": keywords,
        "doi": doi,
        "citations": citations,
        "related_papers": related_papers
    }
    
    return paper_id, paper_data

# Add tools to MCP
@mcp_server.tool()
def search_papers(query: str) -> str:
    """
    Search for scientific papers based on a query.
    
    Args:
        query: Search terms for finding relevant papers
        
    Returns:
        Formatted string with search results
    """
    # Check if it's a nuclear energy query
    if "nuclear" in query.lower() and ("energy" in query.lower() or "power" in query.lower()):
        # Use our predefined nuclear energy papers
        papers = {k: v for k, v in PAPERS_DB.items() if k.startswith("NE")}
    # Check if it's a quantum computing query
    elif "quantum" in query.lower() and "comput" in query.lower():
        # Use our predefined quantum computing papers
        papers = {k: v for k, v in PAPERS_DB.items() if k.startswith("QC")}
    else:
        # Generate papers dynamically
        papers = {}
        for i in range(1, 4):  # Generate 3 papers
            paper_id, paper_data = generate_paper_data(query, "GEN")
            papers[paper_id] = paper_data
            # Add to global database
            PAPERS_DB[paper_id] = paper_data
    
    # Format the results
    result = f"## Search Results for \"{query}\"\n\n"
    result += f"I found {len(papers)} papers related to {query}:\n\n"
    
    for i, (paper_id, paper) in enumerate(papers.items(), 1):
        result += f"### Paper {i}: {paper['title']} ({paper['year']})\n"
        result += f"- **ID**: {paper_id}\n"
        result += f"- **Authors**: {', '.join(paper['authors'])}\n"
        result += f"- **Journal**: {paper['journal']}\n"
        result += f"- **Citations**: {paper['citations']}\n"
        result += f"- **Abstract**: {paper['abstract'][:200]}...\n\n"
    
    return result

@mcp_server.tool()
def analyze_citation_graph(paper_id: str) -> str:
    """
    Analyze the citation graph for a specific paper.
    
    Args:
        paper_id: Identifier for the paper to analyze
        
    Returns:
        Formatted string with citation analysis
    """
    # Check if paper exists
    if paper_id not in PAPERS_DB:
        return f"Paper with ID '{paper_id}' not found."
    
    paper = PAPERS_DB[paper_id]
    
    # Generate citation data
    cited_by_count = paper["citations"]
    h_index = max(1, cited_by_count // 10)  # Simplified h-index calculation
    
    # Create fake citing papers
    citing_papers = []
    for i in range(min(5, cited_by_count)):
        domain = random.choice(list(SCIENTIFIC_DOMAINS.keys()))
        subdomain = random.choice(SCIENTIFIC_DOMAINS[domain])
        citing_title = f"{subdomain} Applications in {domain}: A Reference to {paper['title'].split(':')[0]}"
        citing_citations = random.randint(1, max(2, cited_by_count // 3))
        citing_papers.append((citing_title, citing_citations))
    
    # Sort citing papers by citation count
    citing_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Format the result
    result = f"## Citation Analysis: {paper['title']}\n\n"
    result += f"### Paper ID: {paper_id}\n"
    result += f"- **Total Citations**: {cited_by_count}\n"
    result += f"- **Publication Year**: {paper['year']}\n"
    result += f"- **H-index of Authors**: {h_index}\n"
    
    if citing_papers:
        result += "- **Key Citing Papers**:\n"
        for i, (title, citations) in enumerate(citing_papers[:3], 1):
            result += f"  {i}. \"{title}\" ({citations} citations)\n"
    
    # Citation trends
    current_year = datetime.now().year
    years_since_publication = current_year - paper["year"]
    
    result += "\n### Citation Trends\n"
    if years_since_publication <= 1:
        result += f"This paper was published recently and has already gathered {cited_by_count} citations, indicating significant interest in the field.\n"
    elif paper["citations"] > 30:
        result += f"This paper has been consistently cited since its publication in {paper['year']}, with an average of {paper['citations'] // years_since_publication} citations per year.\n"
    else:
        result += f"This paper has received moderate attention since its publication in {paper['year']}, with citation rates typical for the field.\n"
    
    # Field impact
    result += "\n### Field Impact\n"
    if paper["citations"] > 40:
        result += "This paper appears to be highly influential in its field, with citation patterns suggesting it may be a seminal work in the area.\n"
    elif paper["citations"] > 20:
        result += "This paper has made a notable contribution to its field, as evidenced by steady citation patterns from researchers in related domains.\n"
    else:
        result += "This paper represents a contribution to its specific research niche, with citations primarily from closely related research areas.\n"
    
    return result

@mcp_server.resource("paper://{paper_id}")
def get_paper(paper_id: str) -> str:
    """
    Get details about a specific scientific paper.
    
    Args:
        paper_id: Identifier for the paper
        
    Returns:
        Formatted string with paper details
    """
    # Check if paper exists
    if paper_id not in PAPERS_DB:
        return f"Paper with ID '{paper_id}' not found."
    
    paper = PAPERS_DB[paper_id]
    
    # Format the result
    result = f"## Paper Details: {paper['title']}\n\n"
    result += f"- **ID**: {paper_id}\n"
    result += f"- **Authors**: {', '.join(paper['authors'])}\n"
    result += f"- **Journal**: {paper['journal']}\n"
    result += f"- **Year**: {paper['year']}\n"
    result += f"- **DOI**: {paper['doi']}\n"
    result += f"- **Citations**: {paper['citations']}\n"
    result += f"- **Abstract**: {paper['abstract']}\n"
    
    if paper['keywords']:
        result += f"- **Keywords**: {', '.join(paper['keywords'])}\n"
    
    if paper['related_papers']:
        result += "- **Related Papers**:\n"
        for rel_id in paper['related_papers']:
            if rel_id in PAPERS_DB:
                rel_paper = PAPERS_DB[rel_id]
                result += f"  - {rel_id}: \"{rel_paper['title']}\" ({rel_paper['year']})\n"
    
    return result

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
                            "description": "ID of the paper to retrieve"
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
                            "description": "ID of the paper to analyze"
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
        - When users refer to papers by ID (like NE1, QC2, etc.), use these IDs with get_paper() or analyze_citation_graph()
        - Always reply in clear, concise language focusing on the information requested
        - Format your responses with clear headings and bullet points
        - DO NOT include ANY HTML tags in your responses - no div, span, button, p, code, or any other HTML
        - Do not use markdown code blocks with triple backticks
        - Do not mention or reference any HTML elements or components in your text
        - When mentioning tool calls, simply write the function name with parentheses, e.g. get_paper(NE1)
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