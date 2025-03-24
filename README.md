# MCP Scientific Paper Analyzer

A learning project for building a complete application using Model Context Protocol (MCP), FastAPI, and Streamlit to analyze scientific papers.

## Project Overview

This project demonstrates how to build a secure Model Context Protocol (MCP) server and client integration with a Streamlit frontend. The application allows users to upload scientific papers, analyze them using LLMs, and visualize the results.

## Learning Objectives

- Build a Model Context Protocol (MCP) server using the Python SDK
- Implement secure authentication and authorization
- Create a usable frontend with Streamlit
- Connect to LLM services for document analysis
- Understand context management for effective LLM integration

## Technology Stack

- **Backend**: FastAPI + MCP Python SDK
- **Frontend**: Streamlit
- **Authentication**: JWT with secure cookie storage
- **LLM Integration**: Claude API (or alternative LLM provider)
- **Database**: PostgreSQL

## Project Structure

```
mcp-scientific-analyzer/
├── backend/
│   ├── app/
│   │   ├── mcp/                # MCP server implementation
│   │   │   ├── resources/      # Resource handlers
│   │   │   ├── tools/          # Tool implementations
│   │   │   └── server.py       # MCP server configuration
│   │   ├── api/                # FastAPI routes
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── papers.py       # Paper management endpoints
│   │   │   └── analysis.py     # Analysis results endpoints
│   │   ├── core/               # Core application modules
│   │   │   ├── config.py       # Application configuration
│   │   │   ├── security.py     # Security utilities
│   │   │   └── logging.py      # Logging configuration
│   │   ├── models/             # Database models
│   │   │   ├── user.py         # User model
│   │   │   ├── paper.py        # Scientific paper model
│   │   │   └── analysis.py     # Analysis results model
│   │   └── services/           # Business logic
│   │       ├── llm.py          # LLM integration service
│   │       └── papers.py       # Paper processing service
│   ├── main.py                 # Application entry point
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── app.py                  # Main Streamlit application
│   ├── pages/                  # Streamlit pages
│   │   ├── home.py             # Home page
│   │   ├── upload.py           # Upload page
│   │   ├── analysis.py         # Analysis page
│   │   └── settings.py         # Settings page
│   ├── components/             # Reusable components
│   │   ├── auth.py             # Authentication components
│   │   ├── visualizations.py   # Data visualization components
│   │   └── feedback.py         # User feedback components
│   └── utils/                  # Utility functions
│       ├── api.py              # API client
│       └── session.py          # Session management
└── docs/                       # Documentation
    ├── setup.md                # Setup instructions
    ├── mcp-concepts.md         # MCP concepts documentation
    └── development.md          # Development guide
```

## Feature Roadmap

### Phase 1: Basic Implementation
- [x] Project structure setup
- [ ] Basic FastAPI backend with MCP integration
- [ ] Simple Streamlit frontend with file upload
- [ ] Basic authentication system
- [ ] Paper storage and retrieval

### Phase 2: Core Functionality
- [ ] Complete MCP resource handlers for papers
- [ ] LLM integration for paper analysis
- [ ] Structured data extraction from papers
- [ ] Basic visualization of analysis results
- [ ] User profile management

### Phase 3: Advanced Features
- [ ] Advanced context management for LLMs
- [ ] Interactive analysis editing
- [ ] Batch processing of multiple papers
- [ ] Comparison of different papers
- [ ] Export functionality for analysis results

## Installation and Setup

### Prerequisites
- Python 3.11+
- pip
- virtualenv or conda (recommended)

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your specific settings
   ```

4. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Install Streamlit and dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## MCP Implementation Details

The Model Context Protocol implementation in this project focuses on providing secure access to scientific paper data and analysis tools. The key components include:

### Resources
- `DocumentResource`: Provides access to uploaded scientific papers
- `AnalysisResource`: Gives access to previous analysis results

### Tools
- `ExtractInfoTool`: Extracts structured information from papers
- `AnalyzeAbstractTool`: Performs detailed analysis of paper abstracts
- `CompareResultsTool`: Compares multiple analysis results

### Security
The MCP server implements authentication using JWT tokens and authorizes access to resources based on user permissions. All file access is securely managed through the MCP protocol.

## Learning Path

To make the most of this project for learning purposes, follow this suggested path:

1. **MCP Basics**: Start by understanding the MCP protocol and its components
2. **Backend Development**: Implement the FastAPI server and MCP integration
3. **Frontend Implementation**: Build the Streamlit interface
4. **Security Implementation**: Add authentication and authorization
5. **LLM Integration**: Connect to AI services for analysis
6. **Advanced Features**: Implement additional features as you learn

## Contributing

This is a learning project, but contributions are welcome. Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Model Context Protocol (MCP) documentation and team
- FastAPI framework
- Streamlit team
- Scientific paper analysis community
