#!/usr/bin/env python
"""
Build and serve the documentation locally.

Usage:
    python build_docs.py [--serve] [--port PORT]
"""
import argparse
import subprocess
import sys
from pathlib import Path


def build_docs():
    """Build the documentation."""
    print("🏗️  Building documentation...")
    result = subprocess.run(["mkdocs", "build", "--clean"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Documentation built successfully!")
        print(f"📁 Output directory: {Path.cwd() / 'site'}")
        return True
    else:
        print("❌ Documentation build failed!")
        print(result.stderr)
        return False


def serve_docs(port=8000):
    """Serve the documentation locally."""
    print(f"🌐 Starting documentation server on port {port}...")
    print(f"📖 Documentation will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["mkdocs", "serve", "--dev-addr", f"localhost:{port}"])
    except KeyboardInterrupt:
        print("\n👋 Documentation server stopped")


def main():
    parser = argparse.ArgumentParser(description="Build and serve documentation")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve documentation locally after building"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve documentation on (default: 8000)"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy documentation to GitHub Pages"
    )
    
    args = parser.parse_args()
    
    if args.deploy:
        print("🚀 Deploying documentation to GitHub Pages...")
        result = subprocess.run(["mkdocs", "gh-deploy", "--force"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Documentation deployed successfully!")
        else:
            print("❌ Deployment failed!")
            print(result.stderr)
            sys.exit(1)
    elif args.serve:
        serve_docs(args.port)
    else:
        if build_docs():
            print("\n💡 Tip: Use --serve to start a local server")
            print("   Example: python build_docs.py --serve")


if __name__ == "__main__":
    main()
