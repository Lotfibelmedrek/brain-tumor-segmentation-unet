#!/usr/bin/env python3
"""Convenience entry point: uv run python main.py"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
