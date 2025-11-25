#!/bin/bash

# Stop RAG Demo Services
# Kills Streamlit and Ollama processes

echo "Stopping RAG Demo services..."

# Stop Streamlit
if lsof -i :8501 > /dev/null 2>&1; then
    echo "Stopping Streamlit on port 8501..."
    pkill -f streamlit
    sleep 1
    if lsof -i :8501 > /dev/null 2>&1; then
        PID=$(lsof -ti :8501)
        kill -9 $PID 2>/dev/null
        echo "Streamlit stopped"
    else
        echo "Streamlit stopped"
    fi
else
    echo "Streamlit is not running"
fi

# Stop Ollama
if lsof -i :11434 > /dev/null 2>&1; then
    echo "Stopping Ollama on port 11434..."
    pkill -f ollama
    sleep 1
    if lsof -i :11434 > /dev/null 2>&1; then
        PID=$(lsof -ti :11434)
        kill -9 $PID 2>/dev/null
        echo "Ollama stopped"
    else
        echo "Ollama stopped"
    fi
else
    echo "Ollama is not running"
fi

echo "Done"

