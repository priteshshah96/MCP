# Basic MCP Application

A simple application demonstrating Model Context Protocol (MCP) integration with FastAPI and Streamlit.

## Overview

This project shows how to build a basic MCP server with a Streamlit frontend. The application allows users to interact with LLMs through a simple interface.

## Technology Stack

- **Backend**: FastAPI + MCP Python SDK
- **Frontend**: Streamlit
- **Authentication**: JWT
- **LLM Integration**: Anthropic Claude API

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/basic-mcp-app.git
   cd basic-mcp-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

6. Start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```

7. Open your browser and navigate to http://localhost:8501

## Project Structure


```
basic-mcp-app/
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
├── backend/
│   └── main.py            # Backend API with MCP implementation
└── frontend/
    └── app.py             # Streamlit frontend
```


## Features

- Basic MCP server implementation
- Simple authentication system
- Streamlit frontend with basic UI
- Claude API integration
- Basic error handling

## License

MIT

## Acknowledgments

- Model Context Protocol (MCP) team
- FastAPI framework
- Streamlit team