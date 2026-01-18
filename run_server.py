"""
Production Server Runner for CLV Prediction System
Supports Waitress (Windows) and Gunicorn (Linux/Mac).

Author: Ali Abbass (OTE22)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_waitress():
    """Run with Waitress (Windows production)."""
    try:
        from waitress import serve
        from backend.api import app
        
        host = os.getenv("CLV_API_HOST", "0.0.0.0")
        port = int(os.getenv("CLV_API_PORT", "8000"))
        threads = int(os.getenv("CLV_API_WORKERS", "4"))
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CLV Prediction System - Production Server (Waitress)        ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {host:<54} ║
║  Port: {port:<54} ║
║  Threads: {threads:<51} ║
║                                                              ║
║  API Docs: http://{host}:{port}/api/docs                     ║
║  Frontend: http://{host}:{port}/                             ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        serve(app, host=host, port=port, threads=threads)
        
    except ImportError:
        print("Waitress not installed. Run: pip install waitress")
        sys.exit(1)


def run_uvicorn():
    """Run with Uvicorn (development/async production)."""
    try:
        import uvicorn
        
        host = os.getenv("CLV_API_HOST", "0.0.0.0")
        port = int(os.getenv("CLV_API_PORT", "8000"))
        workers = int(os.getenv("CLV_API_WORKERS", "1"))
        reload = os.getenv("CLV_DEBUG", "false").lower() == "true"
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CLV Prediction System - Development Server (Uvicorn)        ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {host:<54} ║
║  Port: {port:<54} ║
║  Reload: {str(reload):<52} ║
║                                                              ║
║  API Docs: http://{host}:{port}/api/docs                     ║
║  Frontend: http://{host}:{port}/                             ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        uvicorn.run(
            "backend.api:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1
        )
        
    except ImportError:
        print("Uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


def run_gunicorn():
    """Run with Gunicorn (Linux/Mac production)."""
    try:
        import subprocess
        
        host = os.getenv("CLV_API_HOST", "0.0.0.0")
        port = int(os.getenv("CLV_API_PORT", "8000"))
        workers = int(os.getenv("CLV_API_WORKERS", "4"))
        
        cmd = [
            "gunicorn",
            "backend.api:app",
            "-w", str(workers),
            "-k", "uvicorn.workers.UvicornWorker",
            "-b", f"{host}:{port}",
            "--access-logfile", "-",
            "--error-logfile", "-"
        ]
        
        print(f"Starting Gunicorn on {host}:{port} with {workers} workers...")
        subprocess.run(cmd)
        
    except FileNotFoundError:
        print("Gunicorn not installed. Run: pip install gunicorn")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLV Prediction System Server")
    parser.add_argument(
        "--server",
        choices=["waitress", "uvicorn", "gunicorn"],
        default="waitress" if sys.platform == "win32" else "uvicorn",
        help="Server to use (default: waitress on Windows, uvicorn otherwise)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CLV_API_HOST", "0.0.0.0"),
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CLV_API_PORT", "8000")),
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    # Set environment variables from args
    os.environ["CLV_API_HOST"] = args.host
    os.environ["CLV_API_PORT"] = str(args.port)
    
    if args.server == "waitress":
        run_waitress()
    elif args.server == "gunicorn":
        run_gunicorn()
    else:
        run_uvicorn()
