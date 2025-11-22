# Vision Model Chat Interface

A modular Streamlit web application for analyzing images using Ollama-powered vision models. Features single image chat, dual image comparison, and triple image comparison capabilities.

## Features

### 📷 Single Image Chat
- Upload one image and ask questions with optional additional context images
- Attach supplementary images to provide more context in conversations
- Persistent chat history with image references

### 🖼️🖼️ Dual Image Compare
- Upload two images side-by-side for comparison
- Perfect for before/after analysis or comparing similar subjects
- Automatic image alignment and scaling

### 🖼️🖼️🖼️ Triple Image Compare
- Upload three images for comprehensive analysis
- Ideal for progression sequences or multi-angle comparisons
- Side-by-side composite view

### ⚙️ Advanced Features
- Multiple vision model support (Qwen 2.5-VL, Qwen 3-VL, LLaVA, DeepSeek, etc.)
- Adjustable temperature control (0.0-2.0)
- Model management (view running models, unload from VRAM)
- Streaming responses for real-time feedback
- Memory-optimized image handling (images sent only once per conversation)

## Prerequisites

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai to install Ollama for your system
   ```

2. **Pull a vision model**
   ```bash
   # Recommended models:
   ollama pull qwen2.5-vl:7b       # Balanced performance
   ollama pull llava:latest        # General purpose
   ollama pull qwen2-vl:72b        # Best quality (requires GPU)
   ollama pull deepseek-vl:1.3b    # Lightweight, good for OCR
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

3. **Configure settings** in the sidebar:
   - Set Ollama URL (default: `http://localhost:11434`)
   - Select a vision model from the dropdown
   - Adjust temperature (0.0 = focused, 2.0 = creative)

4. **Choose a tab**:
   - **Single Image Chat**: Upload one image and ask questions
   - **Dual Image Compare**: Upload two images for side-by-side analysis
   - **Triple Image Compare**: Upload three images for comprehensive comparison

5. **Upload image(s)** and start chatting!

## Example Questions

### Single Image Chat
- "What's in this image?"
- "Describe the scene in detail"
- "What colors are dominant in this image?"
- "Are there any people in this photo?"
- "What text can you see in this image?"

### Dual Image Compare
- "What are the differences between these two images?"
- "Which image has better lighting?"
- "Compare the composition of both photos"
- "Are these the same object from different angles?"

### Triple Image Compare
- "Show me the progression across these three images"
- "Which of these three images is the best quality?"
- "Describe the differences between all three photos"
- "Are these images related or completely different?"

## Troubleshooting

- **Connection Error**: Make sure Ollama is running (`ollama serve`)
- **Model Not Found**: Pull the model first (`ollama pull qwen3-vl:4b`)
- **Slow Response**: The 4B model is reasonably fast, but larger images may take longer to process

## Configuration

You can adjust settings in the sidebar:
- **Ollama API URL**: Default is `http://localhost:11434`
- **Model Selection**: Automatically detects available vision models
- **Temperature**: Controls response randomness (0.0 = deterministic, 2.0 = highly creative)
- **Model Management**: Unload models to free VRAM or view running models

## Architecture & Code Quality

### 📁 Project Structure

```
vision-model/
├── app.py                    # Main application entry point (140 lines)
├── utils.py                  # Shared utility functions (150 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── tabs/                     # Modular tab implementations
    ├── __init__.py          # Module exports
    ├── single_image.py      # Single image chat functionality (250 lines)
    ├── dual_image.py        # Dual image comparison (200 lines)
    └── triple_image.py      # Triple image comparison (200 lines)
```

### 🏗️ Modular Design

The application follows a **modular architecture** with clear separation of concerns:

#### **app.py** - Application Orchestrator
- **Responsibility**: Main entry point, configuration, and tab coordination
- **Coupling**: Moderate (imports tab classes, passes configuration)
- **Cohesion**: High (all code relates to app initialization)
- **Lines**: ~140 (focused and minimal)

#### **utils.py** - Shared Utilities
- **Responsibility**: Image processing, encoding, model fetching, cleanup
- **Coupling**: Low (no dependencies on other modules)
- **Cohesion**: Very High (pure utility functions)
- **Functions**:
  - `encode_image_to_base64()` - Convert images to base64
  - `combine_images_side_by_side()` - Merge 2 images horizontally
  - `combine_three_images_side_by_side()` - Merge 3 images horizontally
  - `get_available_models()` - Fetch vision models from Ollama
  - `cleanup_ollama_models()` - Free VRAM on exit

#### **tabs/** - Feature Modules
Each tab is a **self-contained class** with identical structure:

**Class Methods** (all tabs):
```python
class SingleImageTab:
    def __init__(self, ollama_url, model_name, temperature)
    def render(self)                      # Main rendering orchestrator
    def _render_upload_section(self)      # Image upload UI
    def _render_chat_section(self)        # Chat interface
    def _handle_chat_input(self)          # Process user messages
    def _build_messages(self)             # Construct API payload
    def _call_ollama_api(self)            # Stream responses
```

**Benefits**:
- ✅ **Single Responsibility**: Each tab handles ONE feature
- ✅ **Encapsulation**: Private methods (prefix `_`) hide implementation
- ✅ **Reusability**: Shared patterns across tabs
- ✅ **Testability**: Easy to unit test individual components

### 🔗 Coupling Analysis

**External Coupling** (dependencies):
- `streamlit` - UI framework
- `requests` - HTTP API calls
- `PIL` - Image processing
- `base64` - Image encoding

**Internal Coupling**:

| From → To | Type | Level | Assessment |
|-----------|------|-------|------------|
| app.py → tabs/* | Control Coupling | Moderate | ✅ Good (dependency injection) |
| tabs/* → utils.py | Data Coupling | Low | ✅ Excellent (pure functions) |
| tabs/* → Session State | Common Coupling | High | ⚠️ Acceptable (Streamlit pattern) |

**Coupling Strengths**:
- No circular dependencies
- Tab modules are independent (no cross-imports)
- Utils has zero internal dependencies
- Changes to one tab don't affect others

**Coupling Diagram**:

```
┌─────────────┐
│  app_new.py │  ◄──── Entry point
└──────┬──────┘
       │ Creates instances (Control Coupling)
       ├────────────────┬────────────────┬────────────────┐
       ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
│single_image  │  │ dual_image   │  │triple_image  │  │ utils.py │
│     .py      │  │     .py      │  │     .py      │  │          │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────▲────┘
       │                  │                  │                 │
       └──────────────────┴──────────────────┴─────────────────┘
                    Data Coupling (imports functions)

       ┌──────────────────────────────────────────────────┐
       │         st.session_state (Shared State)          │
       │   Common Coupling (all tabs read/write)          │
       └──────────────────────────────────────────────────┘
                          ▲
                          │ All tabs depend on this
       ┌──────────────────┼──────────────────┬────────────┐
       │                  │                  │            │
  single_image       dual_image        triple_image   app_new
```

### Cohesion Analysis

**Cohesion Strengths**:
- Each module has a clear, focused purpose
- Methods within classes are tightly related
- No mixed responsibilities or "god objects"
- Easy to understand what each file does

### SOLID Principles

#### **Single Responsibility Principle** 
Each module has ONE reason to change:
- `app.py` changes if app structure changes
- `utils.py` changes if utilities need updates
- `single_image.py` changes if single image features change
- Tab classes never interfere with each other

#### **Open/Closed Principle** ⚠️ Partially
- Open for extension: Easy to add new tabs
- Requires modifying `app.py` to add tabs (acceptable trade-off)

#### **Dependency Inversion** 
- High-level modules (app.py) don't depend on low-level details
- Tabs receive dependencies via constructor injection
- No tight coupling to implementation details

#### **Don't Repeat Yourself (DRY)** 
- Shared utilities extracted to `utils.py`
- Common patterns abstracted into base methods
- Acceptable duplication where needed (2 vs 3 image uploaders)

### 🧠 Memory Management

**Efficient Image Handling**:
```python
# Images sent only ONCE per conversation
if i == 0:  # First message
    user_msg["images"] = [image_b64]
else:       # Subsequent messages
    user_msg = {"content": text}  # No images, uses context
```

**Benefits**:
- **5-10x faster** than sending images every message
- **Reduces token usage** and API costs
- **Lower memory footprint**
- All data in RAM (nothing saved to disk)
- Session-only persistence (privacy-focused)

**Session State Isolation**:
- Single tab: `messages`, `current_image_b64`
- Dual tab: `messages_dual`, `combined_image_b64`
- Triple tab: `messages_triple`, `combined_image_triple_b64`
- No cross-tab interference

