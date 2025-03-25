"""
MCP Scientific Paper Analyzer - Frontend

This Streamlit application provides a user interface for the MCP Scientific Paper Analyzer.
It connects to the backend API to process user queries and display responses.

To run:
1. Make sure the backend is running
2. Run: streamlit run frontend/app.py
"""
import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="MCP Scientific Paper Analyzer",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "tools_and_resources" not in st.session_state:
    st.session_state.tools_and_resources = None

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")

if "backend_status" not in st.session_state:
    st.session_state.backend_status = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .assistant-message {
        background-color: #e9f5ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        margin-top: 20px;
        border: 1px solid #cce5ff;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    .stButton button {
        width: 100%;
    }
    .api-key-input {
        margin-bottom: 10px;
    }
    footer {
        visibility: hidden;
    }
    /* Tool highlighting */
    .tool-call {
        background-color: #25c941;
        color: #000000;
        padding: 3px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        display: inline-block;
    }
    /* Section formatting */
    h3 {
        margin-top: 20px;
        margin-bottom: 15px;
        font-size: 1.3rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: 8px;
    }
    /* Paper formatting */
    .paper-item {
        border-left: 3px solid #4a86e8;
        padding: 12px 15px;
        margin: 15px 0;
        background-color: rgba(74, 134, 232, 0.05);
        border-radius: 4px;
    }
    /* Tool item formatting */
    .tool-item {
        padding: 10px;
        margin-bottom: 12px;
        border-left: 3px solid #25c941;
        background-color: rgba(37, 201, 65, 0.05);
        border-radius: 4px;
    }
    .tool-name {
        font-weight: bold;
        font-family: monospace;
    }
    .tool-description {
        font-size: 0.9rem;
        margin-top: 5px;
    }
    /* Resource item formatting */
    .resource-item {
        padding: 10px;
        margin-bottom: 12px;
        border-left: 3px solid #ff9800;
        background-color: rgba(255, 152, 0, 0.05);
        border-radius: 4px;
    }
    .resource-name {
        font-weight: bold;
        font-family: monospace;
    }
    /* Table formatting */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th {
        background-color: #f5f5f5;
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    td {
        padding: 8px;
        border: 1px solid #ddd;
        vertical-align: top;
    }
</style>
""", unsafe_allow_html=True)

# Function to check backend status
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False

# Update backend status
st.session_state.backend_status = check_backend_status()

# Function to verify API key
def verify_api_key(api_key):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/set-api-key",
            json={"api_key": api_key},
            timeout=5
        )
        data = response.json()
        return data["success"], data["message"]
    except Exception as e:
        return False, f"Error connecting to backend: {str(e)}"

# Function to send chat message
def send_chat_message(message, conversation_history, api_key=None):
    try:
        request_data = {
            "message": message,
            "conversation_history": conversation_history
        }
        
        # Add API key if provided
        if api_key:
            request_data["api_key"] = api_key
        
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=request_data,
            timeout=60  # Longer timeout for chat responses
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, f"Connection error: Could not connect to {BACKEND_URL}. Is the backend server running?"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Function to fetch tools and resources
def fetch_tools_and_resources():
    try:
        tools_response = requests.get(f"{BACKEND_URL}/api/tools", timeout=5)
        resources_response = requests.get(f"{BACKEND_URL}/api/resources", timeout=5)
        
        if tools_response.status_code == 200 and resources_response.status_code == 200:
            tools = tools_response.json().get("tools", [])
            resources = resources_response.json().get("resources", [])
            return {"tools": tools, "resources": resources}
        else:
            return None
    except Exception:
        return None

# Sidebar
with st.sidebar:
    st.markdown("<div class='sub-header'>Settings</div>", unsafe_allow_html=True)
    
    # Backend status indicator
    if st.session_state.backend_status:
        st.success("✅ Backend is running")
    else:
        st.error("❌ Backend not connected")
        st.info(f"Make sure the backend is running at {BACKEND_URL}")
    
    # API Key input
    st.markdown("<div class='api-key-input'>", unsafe_allow_html=True)
    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from https://aistudio.google.com/app/apikey"
    )
    
    # Save API Key button
    if st.button("Save API Key"):
        if api_key:
            with st.spinner("Verifying API key..."):
                success, message = verify_api_key(api_key)
                if success:
                    st.session_state.api_key = api_key
                    st.success("✅ API key saved successfully")
                else:
                    st.error(f"❌ {message}")
        else:
            st.warning("Please enter an API key")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()
    
    # Fetch tools and resources
    if st.session_state.backend_status and not st.session_state.tools_and_resources:
        with st.spinner("Loading tools and resources..."):
            st.session_state.tools_and_resources = fetch_tools_and_resources()
    
    # Display tools and resources
    if st.session_state.tools_and_resources:
        with st.expander("Available Tools", expanded=False):
            tools = st.session_state.tools_and_resources.get("tools", [])
            if tools:
                for tool in tools:
                    description = ""
                    if tool == "search_papers":
                        description = "Search for scientific papers based on keywords"
                    elif tool == "analyze_citation_graph":
                        description = "Analyze citation relationships for a specific paper"
                    elif tool == "get_paper":
                        description = "Retrieve detailed information about a paper"
                    
                    st.markdown(f"""
                    <div class="tool-item">
                        <div class="tool-name">{tool}</div>
                        <div class="tool-description">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No tools available")
        
        with st.expander("Available Resources", expanded=False):
            resources = st.session_state.tools_and_resources.get("resources", [])
            if resources:
                for resource in resources:
                    description = ""
                    if "paper" in resource:
                        description = "Access paper details by ID"
                    
                    st.markdown(f"""
                    <div class="resource-item">
                        <div class="resource-name">{resource}</div>
                        <div class="tool-description">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No resources available")
    
    # About section
    with st.expander("About", expanded=False):
        st.markdown("""
        ## MCP Scientific Paper Analyzer
        
        This application demonstrates the use of Model Context Protocol (MCP) 
        with Google's Gemini API to create a scientific paper analysis tool.
        
        ### Features:
        - Search for scientific papers
        - Analyze citation graphs
        - Get paper details
        
        ### How to use:
        1. Enter your Gemini API key in the sidebar
        2. Ask questions about scientific papers
        3. The AI will use MCP tools to find information and present it to you
        
        ### Example queries:
        - "Search for papers about quantum computing"
        - "Analyze the citation graph for paper ID 1"
        - "Get information about paper 2"
        """)

# Main content
st.markdown("<div class='main-header'>MCP Scientific Paper Analyzer</div>", unsafe_allow_html=True)
st.markdown("Analyze, search, and explore scientific papers using AI with Model Context Protocol (MCP)")

# Display conversation history
chat_container = st.container()

with chat_container:
    if not st.session_state.conversation_history:
        st.info("👋 Hello! I'm your research assistant. Ask me to search for papers, analyze citations, or get paper details!")
    
    # Display the conversation with improved formatting
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header">You:</div>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process the assistant message to ensure proper display
            content = message['content']
            
            # Ensure content begins with a newline for consistent formatting
            if not content.startswith('\n'):
                content = '\n' + content
            
            # Clean up any HTML artifacts
            content = content.replace('</div>', '')
            content = content.replace('```', '')
            
            # Highlight function calls for better readability
            content = content.replace(
                'search_papers(',
                '<span class="tool-call">search_papers('
            )
            content = content.replace(
                'analyze_citation_graph(',
                '<span class="tool-call">analyze_citation_graph('
            )
            content = content.replace(
                'get_paper(',
                '<span class="tool-call">get_paper('
            )
            
            # Make sure all spans are closed properly
            if content.count('<span class="tool-call">') > content.count('</span>'):
                remaining = content.count('<span class="tool-call">') - content.count('</span>')
                content = content + ('</span>' * remaining)
            
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-header">Assistant:</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_message = st.text_area(
        "Your message:",
        height=100, 
        placeholder="Ask about scientific papers, request analysis, or explore citations..."
    )
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        example_queries = st.selectbox(
            "Example queries:",
            [
                "Select an example...",
                "Search for papers about quantum computing",
                "Analyze the citation graph for paper ID 1",
                "Get information about paper 2",
                "Compare papers 1 and 2",
                "What are the latest research trends in machine learning?"
            ]
        )
    
    with col2:
        submit_button = st.form_submit_button("Send")
    
    # Handle example selection
    if example_queries != "Select an example..." and not user_message:
        user_message = example_queries
    
    # Process submission
    if submit_button and user_message:
        if not st.session_state.backend_status:
            st.error("Backend is not connected. Please make sure the backend server is running.")
        else:
            # Show spinner while waiting for response
            with st.spinner("Thinking..."):
                # Send request to backend
                data, error = send_chat_message(
                    user_message, 
                    st.session_state.conversation_history,
                    st.session_state.api_key
                )
                
                if error:
                    st.error(error)
                else:
                    # Update conversation history
                    st.session_state.conversation_history = data["conversation_history"]
                    
                    # Force refresh to display updated conversation
                    st.rerun()

# Add footer with instructions if conversation is empty
if not st.session_state.conversation_history:
    st.markdown("""
    ---
    ### Example queries to try:
    
    - **Search for papers:** "Find papers about quantum computing"
    - **Analyze citations:** "Analyze the citation graph for paper ID 1"
    - **Get paper details:** "Tell me about paper 2"
    - **Compare papers:** "Compare the findings of papers 1 and 2"
    
    The system will use MCP tools to find information and present it to you.
    """)

# Add a footer crediting original author
st.markdown("""
---
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    Created with ❤️ using MCP | Fork this project on GitHub
</div>
""", unsafe_allow_html=True)