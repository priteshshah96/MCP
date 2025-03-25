"""
MCP Scientific Paper Analyzer - Gradio Frontend

This application provides a user interface for the MCP Scientific Paper Analyzer using Gradio.
It connects to the backend API to process user queries and display responses.

To run:
1. Make sure the backend is running
2. Run: python frontend/gradio_app.py
"""
import os
import re
import json
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
    
    # Remove code blocks markers but preserve content
    text = text.replace('```', '')
    
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
    
    # Use API key from input if provided, else use stored key
    current_api_key = api_key_input if api_key_input else API_KEY
    
    # Check if backend is available
    backend_status = check_backend_status()
    if "not connected" in backend_status:
        return chatbot + [
            [message, f"Error: Backend is not connected. Make sure the backend server is running at {BACKEND_URL}."]
        ]
    
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

# Function to clear conversation
def clear_conversation():
    global conversation_history
    conversation_history = []
    return None, []

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# MCP Scientific Paper Analyzer")
    gr.Markdown("Analyze, search, and explore scientific papers using AI with Model Context Protocol (MCP)")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                bubble_full_width=False,
                show_copy_button=True,
                show_share_button=False,
                avatar_images=(None, "https://em-content.zobj.net/thumbs/120/apple/354/robot_1f916.png")
            )
            
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Ask about scientific papers, request analysis, or explore citations...",
                    lines=2,
                    show_label=False
                )
                submit = gr.Button("Send", variant="primary")
            
            with gr.Row():
                clear = gr.Button("Clear Conversation")
                example_dropdown = gr.Dropdown(
                    choices=[
                        "Select an example...",
                        "Search for papers about quantum computing",
                        "Analyze the citation graph for paper ID 1",
                        "Get information about paper 2",
                        "Compare papers 1 and 2",
                        "What are the latest research trends in machine learning?"
                    ],
                    label="Example queries",
                    value="Select an example..."
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            
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
            
            gr.Markdown("### About")
            gr.Markdown("""
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
            """)
    
    # Set up event handlers
    submit.click(send_message, [message, chatbot, api_key_input], [chatbot], queue=False).then(
        lambda: "", None, [message], queue=False
    )
    message.submit(send_message, [message, chatbot, api_key_input], [chatbot], queue=False).then(
        lambda: "", None, [message], queue=False
    )
    
    clear.click(clear_conversation, None, [message, chatbot], queue=False)
    
    example_dropdown.change(
        use_example, 
        [example_dropdown, chatbot], 
        [message, chatbot], 
        queue=False
    )
    
    save_key_button.click(
        save_api_key,
        [api_key_input],
        [backend_status_text],
        queue=False
    )
    
    # Initialize with welcome message
    demo.load(
        lambda: [[None, "👋 Hello! I'm your research assistant. Ask me to search for papers, analyze citations, or get paper details!"]],
        None,
        [chatbot],
        queue=False
    )

# Launch the app
if __name__ == "__main__":
    # Update requirements.txt to include gradio
    print("Starting MCP Scientific Paper Analyzer with Gradio...")
    print(f"Backend URL: {BACKEND_URL}")
    print("Press Ctrl+C to exit")
    
    demo.launch(server_name="0.0.0.0", server_port=8501, share=False)