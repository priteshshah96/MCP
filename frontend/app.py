"""
MCP Scientific Paper Analyzer - Gradio Frontend (Gradio 5.23.0)

This application provides a user interface for the MCP Scientific Paper Analyzer using Gradio.
It connects to the backend API to process user queries and display responses.

To run:
1. Make sure the backend is running
2. Run: python frontend/app.py
"""
import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize state
conversation_history = []

# Function to clean text
def clean_text(text):
    """Clean response text of any unwanted HTML or formatting"""
    if not text:
        return ""
    
    # Ensure text begins with a newline for consistent formatting
    if not text.startswith('\n'):
        text = '\n' + text
    
    # Remove HTML tags that shouldn't be in the response
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    
    return text

# Function to check backend status
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            return "Backend is running ✓"
        return "Backend not connected ✗"
    except Exception:
        return "Backend not connected ✗"

# Function to verify and save API key
def save_api_key(api_key):
    global API_KEY
    if not api_key:
        return "Please enter an API key ✗"
        
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/set-api-key",
            json={"api_key": api_key},
            timeout=5
        )
        data = response.json()
        if data["success"]:
            API_KEY = api_key
            return "API key saved successfully ✓"
        else:
            return f"Invalid API key: {data['message']} ✗"
    except Exception as e:
        return f"Error connecting to backend: {str(e)} ✗"

# Function to send message to backend and get response
def send_message(message, chatbot, api_key_input):
    global conversation_history, API_KEY
    
    if not message or message.strip() == "":
        return chatbot
    
    # Use API key from input if provided, else use stored key
    current_api_key = api_key_input if api_key_input else API_KEY
    
    # Check if backend is available
    backend_status = check_backend_status()
    if "not connected" in backend_status:
        return chatbot + [[message, f"Error: Backend is not connected. Make sure the backend server is running at {BACKEND_URL}."]]
    
    # Send request to backend
    try:
        request_data = {
            "message": message,
            "conversation_history": conversation_history,
            "api_key": current_api_key
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            conversation_history = data["conversation_history"]
            # Clean response text
            response_text = clean_text(data["response"])
            return chatbot + [[message, response_text]]
        else:
            error_message = f"Error: {response.text}"
            return chatbot + [[message, error_message]]
    
    except requests.exceptions.ConnectionError:
        connection_error = f"Connection error: Could not connect to {BACKEND_URL}. Is the backend server running?"
        return chatbot + [[message, connection_error]]
    except Exception as e:
        general_error = f"Error: {str(e)}"
        return chatbot + [[message, general_error]]

# Function to handle example selection
def use_example(example, chatbot):
    if example != "Select an example...":
        return example, chatbot
    return "", chatbot

# Function to clear conversation
def clear_conversation():
    global conversation_history
    conversation_history = []
    return None, []

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
            # MCP Scientific Paper Analyzer
            ### Analyze, search, and explore scientific papers using AI with Model Context Protocol (MCP)
            """
        )
    
    with gr.Row():
        # Main chat column
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                value=[[None, "👋 Hello! I'm your research assistant. Ask me to search for papers, analyze citations, or get paper details!"]],
                height=500,
                show_copy_button=True,
            )
            
            # Input box
            with gr.Group():
                message = gr.Textbox(
                    placeholder="Ask about scientific papers, request analysis, or explore citations...",
                    lines=2,
                    show_label=False
                )
            
            # Buttons
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Conversation")
                    
            # Example queries
            with gr.Accordion("Example Queries", open=False):
                example_dropdown = gr.Dropdown(
                    choices=[
                        "Select an example...",
                        "Search for papers about machine learning",
                        "Search for papers about climate change",
                        "Search for papers about quantum computing",
                        "Analyze the citation graph for paper P1",
                        "Get information about paper P2",
                        "Compare papers P1 and P2",
                        "What are the latest research trends in artificial intelligence?"
                    ],
                    label="Try one of these examples",
                    value="Select an example..."
                )
        
        # Settings and info column
        with gr.Column(scale=3):
            # Settings section
            with gr.Group():
                gr.Markdown("## Settings")
                backend_status_text = gr.Textbox(
                    value=check_backend_status(),
                    label="Backend Status",
                    interactive=False
                )
                
                api_key_input = gr.Textbox(
                    value=API_KEY,
                    label="Gemini API Key",
                    placeholder="Enter your Gemini API key",
                    type="password"
                )
                
                save_key_button = gr.Button("Save API Key")
            
            # About section
            with gr.Group():
                gr.Markdown("## About")
                gr.Markdown(
                    """
                    This application demonstrates the use of Model Context Protocol (MCP) 
                    with Google's Gemini API to create a scientific paper analysis tool.
                    
                    **Features:**
                    - Search for scientific papers
                    - Analyze citation graphs
                    - Get paper details
                    
                    **How to use:**
                    1. Enter your Gemini API key
                    2. Ask questions about scientific papers
                    3. The AI will use MCP tools to find information
                    """
                )
    
    # Set up event handlers
    submit.click(send_message, [message, chatbot, api_key_input], [chatbot]).then(
        lambda: "", None, [message]
    )
    message.submit(send_message, [message, chatbot, api_key_input], [chatbot]).then(
        lambda: "", None, [message]
    )
    
    clear.click(clear_conversation, None, [message, chatbot])
    
    example_dropdown.change(
        use_example, 
        [example_dropdown, chatbot], 
        [message, chatbot]
    )
    
    save_key_button.click(
        save_api_key,
        [api_key_input],
        [backend_status_text]
    )

# Launch the app
if __name__ == "__main__":
    print("=" * 70)
    print(" MCP Scientific Paper Analyzer - Gradio Frontend")
    print("=" * 70)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    print("\nStarting Gradio interface...")
    
    # Launch with Gradio options - simplified for compatibility
    demo.launch(server_name="0.0.0.0", server_port=8501, share=False)