import streamlit as st
import requests
from tabs import SingleImageTab, DualImageTab, TripleImageTab
from utils import get_available_models

# Page configuration
st.set_page_config(
    page_title="Vision Model Chat",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Ollama URL
    st.markdown("### 🌐 Ollama Connection")
    ollama_url = st.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="The URL where Ollama is running"
    )
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    
    # Get available models
    available_models = get_available_models(ollama_url)
    
    if available_models:
        model_name = st.selectbox(
            "Choose a vision model",
            options=available_models,
            help="Select which vision model to use for analysis"
        )
        # Display selected model
        st.info(f"**Selected Model:** `{model_name}`")
    else:
        st.warning("⚠️ No vision models found. Make sure Ollama is running and you have vision models installed.")
        model_name = st.text_input(
            "Model name",
            value="llava:latest",
            help="Manually enter the model name"
        )
        if model_name:
            st.info(f"**Selected Model:** `{model_name}`")
    
    # Temperature setting
    st.markdown("### 🌡️ Temperature")
    temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more focused"
    )
    
    # Thinking/reasoning toggle
    st.markdown("### 🧠 Model Thinking")
    enable_thinking_api = st.checkbox(
        "Enable model thinking",
        value=False,
        help="Send 'think' parameter to API. Models like Qwen3, DeepSeek-R1 will generate reasoning traces."
    )
    show_thinking = st.checkbox(
        "Show thinking process",
        value=True,
        help="Display the model's thinking/reasoning output in the chat (only if model generates it)",
        disabled=not enable_thinking_api
    )
    
    # Model management section
    st.markdown("---")
    st.markdown("### 🔧 Model Management")
    
    # Model download section
    with st.expander("📥 Download Vision Model"):
        st.markdown("Enter a vision model name to download from Ollama library:")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            model_to_download = st.text_input(
                "Model name",
                placeholder="e.g., llava:latest, qwen2.5-vl:7b",
                key="download_model_input",
                label_visibility="collapsed"
            )
        with col2:
            download_btn = st.button("📥 Pull", key="download_model_btn", use_container_width=True)
        
        if download_btn and model_to_download:
            try:
                # Use streaming API to show progress
                response = requests.post(
                    f"{ollama_url}/api/pull",
                    json={"name": model_to_download},
                    stream=True,
                    timeout=600
                )
                
                if response.status_code == 200:
                    import json
                    
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    total_size = 0
                    completed_size = 0
                    
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get('status', '')
                            
                            # Extract progress information
                            if 'total' in data and 'completed' in data:
                                total_size = data['total']
                                completed_size = data['completed']
                                
                                if total_size > 0:
                                    progress_percent = completed_size / total_size
                                    progress_bar.progress(progress_percent)
                                    
                                    # Convert to human-readable sizes
                                    completed_mb = completed_size / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    progress_text.text(f"{completed_mb:.1f} MB / {total_mb:.1f} MB")
                            
                            # Show current status
                            if status:
                                status_placeholder.info(f"📦 {status}")
                    
                    progress_bar.progress(1.0)
                    progress_text.text("Download complete!")
                    st.success(f"✅ Successfully downloaded {model_to_download}")
                    
                    # Clear cache to update model list immediately
                    get_available_models.clear()
                    
                    import time
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to download model: {response.status_code}")
            except requests.exceptions.Timeout:
                st.error("❌ Download timed out. The model may be too large or connection is slow.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.markdown("**Popular vision models:**")
        st.markdown("""
        - `llava:latest` - General purpose
        - `llava:13b` - Higher quality
        - `qwen2.5-vl:7b` - Balanced performance
        - `moondream:latest` - Lightweight
        - `minicpm-v:latest` - Fast inference
        """)
    
    # Model removal section
    with st.expander("🗑️ Remove Vision Model"):
        # Get all available models
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                all_models = response.json().get('models', [])
                model_names = [m['name'] for m in all_models]
                
                if model_names:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        model_to_remove = st.selectbox(
                            "Select model to remove",
                            options=model_names,
                            key="remove_model_select",
                            label_visibility="collapsed"
                        )
                    with col2:
                        remove_btn = st.button("🗑️ Remove", key="remove_model_btn", use_container_width=True)
                    
                    if remove_btn and model_to_remove:
                        try:
                            delete_response = requests.delete(
                                f"{ollama_url}/api/delete",
                                json={"name": model_to_remove},
                                timeout=30
                            )
                            
                            if delete_response.status_code == 200:
                                st.success(f"✅ Successfully removed {model_to_remove}")
                                
                                # Clear cache to update model list immediately
                                get_available_models.clear()
                                
                                import time
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to remove model: {delete_response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.info("No models available to remove")
            else:
                st.error("❌ Failed to fetch models")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Unload all models button
    if st.button("🔄 Unload All Models", help="Free VRAM by unloading all loaded models"):
        try:
            response = requests.get(f"{ollama_url}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get('models', [])
                if running_models:
                    for model in running_models:
                        model_name_to_unload = model.get('name', '')
                        if model_name_to_unload:
                            unload_response = requests.post(
                                f"{ollama_url}/api/generate",
                                json={
                                    "model": model_name_to_unload,
                                    "keep_alive": 0
                                }
                            )
                            if unload_response.status_code == 200:
                                st.success(f"✅ Unloaded: {model_name_to_unload}")
                            else:
                                st.error(f"❌ Failed to unload: {model_name_to_unload}")
                    st.info("All models have been unloaded from memory.")
                else:
                    st.info("No models currently loaded in memory.")
            else:
                st.error(f"Failed to get running models: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to Ollama at {ollama_url}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Show running models
    if st.button("👀 Show Running Models", help="Display all currently loaded models"):
        try:
            response = requests.get(f"{ollama_url}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get('models', [])
                if running_models:
                    st.markdown("**Currently loaded models:**")
                    for model in running_models:
                        model_name_display = model.get('name', 'Unknown')
                        model_size = model.get('size', 0) / (1024**3)  # Convert to GB
                        st.write(f"• {model_name_display} ({model_size:.2f} GB)")
                else:
                    st.info("No models currently loaded in memory.")
            else:
                st.error(f"Failed to get running models: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to Ollama at {ollama_url}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.info("""
    **Tips:**
    - Single Image Chat: Upload one image and ask questions with optional additional context images
    - Dual Image Compare: Upload two images side-by-side for comparison
    - Triple Image Compare: Upload three images for comprehensive analysis
    - Use temperature to control response randomness
    - Set a custom system prompt to guide the model's behavior across all modes
    - Unload models when switching to free VRAM
    """)

# Main content area
st.title("👁️ Vision Model Chat Interface")
st.markdown("Upload images and chat with vision models powered by Ollama")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📷 Single Image Chat",
    "🖼️🖼️ Dual Image Compare",
    "🖼️🖼️🖼️ Triple Image Compare"
])

# Render each tab
with tab1:
    single_image_tab = SingleImageTab(ollama_url, model_name, temperature, enable_thinking_api, show_thinking)
    single_image_tab.render()

with tab2:
    dual_image_tab = DualImageTab(ollama_url, model_name, temperature, enable_thinking_api, show_thinking)
    dual_image_tab.render()

with tab3:
    triple_image_tab = TripleImageTab(ollama_url, model_name, temperature, enable_thinking_api, show_thinking)
    triple_image_tab.render()
