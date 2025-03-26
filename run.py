import os
import subprocess
import sys
import time

def main():
    # Start the backend server
    print("Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Time given for backend to start
    time.sleep(2)
    
    # Start the frontend 
    print("Starting frontend...")
    frontend_process = subprocess.Popen(
        [sys.executable, "frontend/app.py"]
    )
    
    print("\nApplication started!")
    print("- Backend running at: http://localhost:8000")
    print("- Frontend running at: http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers...\n")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # clean shtdown
        print("\nShutting down...")
        frontend_process.terminate()
        backend_process.terminate()
        print("Done!")

if __name__ == "__main__":
    main()