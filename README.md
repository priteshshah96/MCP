# Basic MCP Application

A simple app that shows how Model Context Protocol (MCP) works with FastAPI and Gradio (because i am not a dev who enjoys Streamlit headaches).

## Overview

This project demonstrates a basic MCP server with a Gradio frontend (Streamlit was a headache, and life's too short for unnecessary pain). Users can chat with AI models through what marketing people would call a "simple interface" and what developers know as "the best I could do before I moveon 🥲 ."

## Technology Stack

- **Backend**: FastAPI + MCP Python SDK (a match made in heaven, unlike pineapple on pizza)
- **Frontend**: Gradio (because pretty buttons make dopamine go brrr)
- **AI Integration**: Google Gemini API (not the horoscope sign, the sundar pichai's AI AI AI AI AI AI thingy)

## Known Issues

⚠️ **Please Note**
- The citation tool is not working properly at the moment. You may see errors when trying to analyze paper citations or when using some of the advanced search features. I am working on fixing this issue. When will it be fixed? Who knows ¯\\\_(ツ)\_/¯. Maybe when we have AGI.
- Semantic Scholar API has rate limitations, which may cause the search functionality to sometimes return an error message stating "I cannot directly search for and provide you with papers." This is what happens when free APIs meet enthusiastic users - we love them to death. Just wait a bit and try again (or distract yourself with coffee while the rate limits reset).

## Speed Up Your Setup

This project works great with `uv`, a super fast Python package installer! Instead of waiting for pip to finish sometime next century, you can use `uv` to install dependencies in seconds. I highly recommend this for a smoother experience (and to reclaim hours of your life staring at progress bars).

## Quick Start

### What You'll Need

- Python 3.11 or newer (sorry dinosaurs still using Python 2)
- pip package manager (or its cooler, faster cousin `uv`)
- The patience of a RCB (Google them) (optional but recommended)

### Setup Steps

1. Clone this project:
   ```bash
   git clone https://github.com/yourusername/basic-mcp-app.git
   cd basic-mcp-app
   ```

2. Create a virtual environment (because global dependencies are the path to emotional damage):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   
   **Using pip (the tortoise way):**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Using uv (the hare way that actually wins the race):**
   ```bash
   # Install uv first if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   or 
   ```bash
   pip install uv
   ```
   
   Then install dependencies with uv:
   ```bash
   uv pip install -r requirements.txt
   ```
   
   Using `uv` makes Python package installation go brrrr! It's much faster than regular pip (motherpromise🤞). In the time it takes pip to realize what you've asked it to do, uv has already finished, made coffee, and started writing your next app for you.


4. Set up your API keys (the things you should never commit to GitHub but someone always does anyway):
   ```bash
   cp .env.example .env
   # Open .env and add your API keys
   ```

5. Run both servers with one command (like magic, but with more semicolons):
   ```bash
   python run.py
   ```
   
   This starts both the backend and frontend at once. It's like having your cake and eating it too, but with fewer calories.
   
   You can also start them separately if needed (for the control freaks among us):
   - Backend: `uvicorn backend.main:app --reload`
   - Frontend: `python frontend/app.py`

6. Open your web browser and go to http://localhost:8501 (if this doesn't work, try turning it off and on again 😑)

## Project Files

Look at this beautiful directory structure that will never stay this clean once development really starts:

```
basic-mcp-app/
├── .env.example           # Template for your API keys(Please don't make your api keys public🙏)
├── .gitignore             # Files to ignore in git (like the emotional baggage)
├── README.md              # This help file that nobody reads until desperate
├── requirements.txt       # Required packages (aka dependency hell)
├── run.py                 # Script to start both servers
├── backend/
│   └── main.py            # Backend server code with MCP (where the real magic happens)
└── frontend/
    └── app.py             # Gradio frontend interface (pretty buttons go here)
```


## Features

- Scientific paper search using Semantic Scholar (for when Google Scholar is just too mainstream)
- Paper analysis tools (that work 60% of the time, every time)
- Simple chat interface (simple for users, nightmare for developers)
- Easy setup process (if you've ever climbed Everest, this will feel like a walk in the park)

## License

MIT (Because I'm nice and don't want to read long licenses either)

## Thanks

- Anthropic for MCP -  https://www.anthropic.com/news/model-context-protocol
- https://modelcontextprotocol.io/introduction
- Claude for vibe coding the parts of demo, not completely, just tiny bit 🤏.