# Qwen3-VL Image Analysis with Streamlit

A Streamlit web interface for analyzing images using Ollama's Qwen3-VL model.

## Features

- 🖼️ Upload images (JPG, PNG, BMP, GIF, WebP)
- 💬 Ask questions about uploaded images
- 🔄 Chat history with context
- ⚙️ Configurable settings (temperature, model, API URL)

## Prerequisites

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai to install Ollama for your system
   ```

2. **Pull the Qwen3-VL model**
   ```bash
   ollama pull qwen3-vl:4b
   ```

3. **Make sure Ollama is running**
   ```bash
   ollama serve
   ```

## Installation

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the URL shown (usually http://localhost:8501)

3. **Upload an image** using the file uploader

4. **Ask questions** about the image in the chat interface

## Example Questions

- "What's in this image?"
- "Describe the scene in detail"
- "What colors are dominant in this image?"
- "Are there any people in this photo?"
- "What text can you see in this image?"
- "What's the mood or atmosphere of this image?"

## Troubleshooting

- **Connection Error**: Make sure Ollama is running (`ollama serve`)
- **Model Not Found**: Pull the model first (`ollama pull qwen3-vl:4b`)
- **Slow Response**: The 4B model is reasonably fast, but larger images may take longer to process

## Configuration

You can adjust settings in the sidebar:
- **Ollama API URL**: Default is `http://localhost:11434`
- **Model Name**: Default is `qwen3-vl:4b` (you can try `qwen3-vl:7b` for better results)
- **Temperature**: Controls response randomness (0.0 = deterministic, 1.0 = creative)
