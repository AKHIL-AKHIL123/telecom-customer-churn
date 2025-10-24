"""
Run both FastAPI backend and Streamlit frontend simultaneously
"""
import subprocess
import sys
import time
from pathlib import Path

def check_model_exists():
    """Check if model is trained"""
    model_path = Path("models/churn_model.pkl")
    if not model_path.exists():
        print("WARNING: Model not found!")
        print("Please train the model first:")
        print("  python src/ml/train.py")
        sys.exit(1)

def main():
    print("=" * 50)
    print("Starting Telecom Churn Prediction System")
    print("=" * 50)
    
    # Check if model exists
    check_model_exists()
    
    print("\nModel found")
    print("\nStarting services...")
    print("  Backend:  http://localhost:8000")
    print("  Frontend: http://localhost:8501")
    print("\nPress Ctrl+C to stop both services\n")
    
    # Start backend process
    backend = subprocess.Popen(
        [sys.executable, "src/api/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait a bit for backend to start
    time.sleep(3)
    
    # Start frontend process
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    try:
        # Monitor both processes
        while True:
            # Check if processes are still running
            if backend.poll() is not None:
                print("\nERROR: Backend stopped unexpectedly")
                frontend.terminate()
                break
            
            if frontend.poll() is not None:
                print("\nERROR: Frontend stopped unexpectedly")
                backend.terminate()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        
        # Wait for processes to terminate
        backend.wait(timeout=5)
        frontend.wait(timeout=5)
        
        print("Services stopped successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
