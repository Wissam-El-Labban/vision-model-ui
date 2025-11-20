import streamlit as st
import requests
import base64
from pathlib import Path
import json
import atexit
import signal
import sys
from PIL import Image
import io

# Function to unload all running Ollama models
def cleanup_ollama_models():
    """Unload all running Ollama models on exit"""
    try:
        ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
        # Get list of running models
        response = requests.get(f"{ollama_url}/api/ps", timeout=5)
        if response.status_code == 200:
            running_models = response.json().get("models", [])
            # Unload each model
            for model_info in running_models:
                model_name = model_info.get("name", "")
                if model_name:
                    try:
                        requests.post(
                            f"{ollama_url}/api/generate",
                            json={"model": model_name, "keep_alive": 0},
                            timeout=5
                        )
                        print(f"Unloaded model: {model_name}")
                    except:
                        pass
    except:
        pass

# Register cleanup function
atexit.register(cleanup_ollama_models)

# Function to combine two images side by side
def combine_images_side_by_side(image1_bytes, image2_bytes):
    """Combine two images horizontally into one composite image"""
    try:
        # Open images
        img1 = Image.open(io.BytesIO(image1_bytes))
        img2 = Image.open(io.BytesIO(image2_bytes))
        
        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Resize images to same height (use smaller height)
        target_height = min(img1.height, img2.height)
        
        # Calculate new widths maintaining aspect ratio
        img1_new_width = int(img1.width * (target_height / img1.height))
        img2_new_width = int(img2.width * (target_height / img2.height))
        
        # Resize images
        img1_resized = img1.resize((img1_new_width, target_height), Image.LANCZOS)
        img2_resized = img2.resize((img2_new_width, target_height), Image.LANCZOS)
        
        # Create new combined image
        combined_width = img1_resized.width + img2_resized.width
        combined_image = Image.new('RGB', (combined_width, target_height))
        
        # Paste images side by side
        combined_image.paste(img1_resized, (0, 0))
        combined_image.paste(img2_resized, (img1_resized.width, 0))
        
        # Convert to bytes
        output = io.BytesIO()
        combined_image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.getvalue(), combined_image
    except Exception as e:
        st.error(f"Error combining images: {str(e)}")
        return None, None

# Register cleanup function
atexit.register(cleanup_ollama_models)

# Page configuration
st.set_page_config(
    page_title="Vision Model Image Analysis",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Vision Model Image Analysis")
st.markdown("Upload an image and ask questions about it using Qwen3-VL or LLaVA models")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    ollama_url = st.text_input("Ollama API URL", value="http://localhost:11434")
    
    # Store ollama_url in session state for cleanup function
    st.session_state.ollama_url = ollama_url
    
    # Function to get available models from Ollama
    @st.cache_data(ttl=60)
    def get_available_models(url):
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                # Filter for vision models (those with "vision", "vl", "llava", "qwen" in name)
                vision_keywords = ["vision", "vl", "llava", "qwen", "moondream", "minicpm"]
                vision_models = [
                    m["name"] for m in models 
                    if any(keyword in m["name"].lower() for keyword in vision_keywords)
                ]
                return sorted(vision_models) if vision_models else []
            return []
        except:
            return []
    
    # Get available models
    available_models = get_available_models(ollama_url)
    
    if available_models:
        st.success(f"✅ Found {len(available_models)} vision model(s)")
        
        # Add a refresh button
        if st.button("🔄 Refresh Models"):
            st.cache_data.clear()
            st.rerun()
        
        # Model selection from available models
        model_name = st.selectbox(
            "Select Vision Model",
            options=available_models,
            index=0
        )
        st.info(f"Using: `{model_name}`")
    else:
        st.warning("⚠️ No vision models detected. Using manual entry.")
        model_name = st.text_input("Model Name", value="qwen3-vl:4b")
        st.caption("Make sure Ollama is running and you have vision models installed.")
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    
    # Model cleanup section
    st.markdown("### 🧹 Model Management")
    col_cleanup1, col_cleanup2 = st.columns(2)
    
    with col_cleanup1:
        if st.button("🗑️ Unload Models", help="Unload all running Ollama models to free up VRAM"):
            with st.spinner("Unloading models..."):
                try:
                    response = requests.get(f"{ollama_url}/api/ps", timeout=5)
                    if response.status_code == 200:
                        running_models = response.json().get("models", [])
                        if running_models:
                            for model_info in running_models:
                                model_name_to_unload = model_info.get("name", "")
                                if model_name_to_unload:
                                    requests.post(
                                        f"{ollama_url}/api/generate",
                                        json={"model": model_name_to_unload, "keep_alive": 0},
                                        timeout=5
                                    )
                            st.success(f"Unloaded {len(running_models)} model(s)")
                        else:
                            st.info("No models currently loaded")
                    else:
                        st.error("Could not connect to Ollama")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col_cleanup2:
        if st.button("📊 Show Running", help="Show currently loaded models"):
            try:
                response = requests.get(f"{ollama_url}/api/ps", timeout=5)
                if response.status_code == 200:
                    running_models = response.json().get("models", [])
                    if running_models:
                        st.write("**Running Models:**")
                        for model_info in running_models:
                            st.text(f"• {model_info.get('name', 'Unknown')}")
                    else:
                        st.info("No models currently loaded")
                else:
                    st.error("Could not connect to Ollama")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    st.markdown("### About")
    st.markdown("This app uses Ollama's vision models to analyze images and answer questions.")
    st.markdown("Make sure Ollama is running and your chosen model is installed:")
    st.code(f"ollama pull {model_name}", language="bash")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_b64" not in st.session_state:
    st.session_state.current_image_b64 = None

# Initialize session state for dual image mode
if "messages_dual" not in st.session_state:
    st.session_state.messages_dual = []
if "combined_image_b64" not in st.session_state:
    st.session_state.combined_image_b64 = None

# Create tabs
tab1, tab2 = st.tabs(["📷 Single Image", "🖼️🖼️ Dual Image Compare"])

# TAB 1: Single Image Analysis
with tab1:
    # Main layout
    col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    
    # Add custom CSS for better drag and drop visibility
    st.markdown("""
        <style>
        [data-testid="stFileUploader"] {
            border: 3px dashed #4CAF50;
            border-radius: 10px;
            padding: 30px;
            background-color: rgba(76, 175, 80, 0.05);
            text-align: center;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #45a049;
            background-color: rgba(76, 175, 80, 0.1);
        }
        [data-testid="stFileUploader"] section {
            padding: 20px;
        }
        [data-testid="stFileUploader"] section > div {
            font-size: 1.1em;
            color: #4CAF50;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop an image here, or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        help="Upload an image to analyze",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")
        
        # Convert image to base64
        image_bytes = uploaded_file.read()
        st.session_state.current_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        st.session_state.current_image = uploaded_file.name
        
        # Reset to beginning for potential re-reading
        uploaded_file.seek(0)
    elif st.session_state.current_image:
        st.info(f"Current image: {st.session_state.current_image}")

with col2:
    st.header("Chat")
    
    # Display chat history with full height
    chat_container = st.container(height=800)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the image..."):
        if not st.session_state.current_image_b64:
            st.error("Please upload an image first!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Call Ollama API with streaming and conversation memory
            try:
                api_endpoint = f"{ollama_url}/api/chat"  # Use chat endpoint instead of generate
                
                # Build messages array with full conversation history
                messages = []
                
                # Add all previous messages from chat history with image on user messages
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        # Include image with every user message for better context
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "images": [st.session_state.current_image_b64]
                        })
                    else:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Add the current user message with image
                messages.append({
                    "role": "user",
                    "content": prompt,
                    "images": [st.session_state.current_image_b64]
                })
                
                payload = {
                    "model": model_name,
                    "messages": messages,  # Send full conversation history
                    "stream": True,
                    "options": {
                        "temperature": temperature
                    }
                }
                
                # Create placeholder for streaming response
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                
                # Stream the response
                full_response = ""
                response = requests.post(api_endpoint, json=payload, stream=True, timeout=120)
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    full_response += chunk["message"]["content"]
                                    # Update the placeholder with current response
                                    message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Add complete response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to Ollama. Make sure it's running on " + ollama_url)
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The model might be taking too long to respond.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History", key="clear_single"):
            st.session_state.messages = []
            st.rerun()

# TAB 2: Dual Image Analysis
with tab2:
    st.markdown("### Compare Two Images Side-by-Side")
    st.caption("Upload two images and they will be combined into one for comparison analysis")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader(
            "Upload first image",
            type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
            help="First image for comparison",
            key="dual_image1"
        )
        if uploaded_file1:
            st.image(uploaded_file1, caption="Image 1", use_container_width=True)
    
    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader(
            "Upload second image",
            type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
            help="Second image for comparison",
            key="dual_image2"
        )
        if uploaded_file2:
            st.image(uploaded_file2, caption="Image 2", use_container_width=True)
    
    with col3:
        st.subheader("Combined View")
        if uploaded_file1 and uploaded_file2:
            # Read image bytes
            image1_bytes = uploaded_file1.read()
            image2_bytes = uploaded_file2.read()
            uploaded_file1.seek(0)
            uploaded_file2.seek(0)
            
            # Combine images
            combined_bytes, combined_img = combine_images_side_by_side(image1_bytes, image2_bytes)
            
            if combined_bytes:
                # Display combined image
                st.image(combined_img, caption="Combined Side-by-Side", use_container_width=True)
                
                # Convert to base64
                st.session_state.combined_image_b64 = base64.b64encode(combined_bytes).decode('utf-8')
                st.success("✅ Images combined! Ask questions about both images below.")
        else:
            st.info("Upload both images to see combined view")
    
    # Chat interface for dual image
    st.divider()
    st.subheader("Chat About Both Images")
    
    # Display chat history
    chat_container_dual = st.container(height=600)
    with chat_container_dual:
        for message in st.session_state.messages_dual:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt_dual := st.chat_input("Ask about both images (e.g., 'What are the differences?')", key="dual_chat"):
        if not st.session_state.combined_image_b64:
            st.error("Please upload both images first!")
        else:
            # Add user message to chat history
            st.session_state.messages_dual.append({"role": "user", "content": prompt_dual})
            
            # Display user message
            with chat_container_dual:
                with st.chat_message("user"):
                    st.markdown(prompt_dual)
            
            # Call Ollama API with combined image
            try:
                api_endpoint = f"{ollama_url}/api/chat"
                
                # Build messages for dual mode
                messages_dual = []
                for msg in st.session_state.messages_dual:
                    if msg["role"] == "user":
                        messages_dual.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "images": [st.session_state.combined_image_b64]
                        })
                    else:
                        messages_dual.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Add current message
                messages_dual.append({
                    "role": "user",
                    "content": prompt_dual,
                    "images": [st.session_state.combined_image_b64]
                })
                
                payload = {
                    "model": model_name,
                    "messages": messages_dual,
                    "stream": True,
                    "options": {
                        "temperature": temperature
                    }
                }
                
                # Create placeholder for streaming response
                with chat_container_dual:
                    with st.chat_message("assistant"):
                        message_placeholder_dual = st.empty()
                
                # Stream the response
                full_response_dual = ""
                response = requests.post(api_endpoint, json=payload, stream=True, timeout=120)
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    full_response_dual += chunk["message"]["content"]
                                    message_placeholder_dual.markdown(full_response_dual + "▌")
                            except json.JSONDecodeError:
                                continue
                    
                    # Final update
                    message_placeholder_dual.markdown(full_response_dual)
                    
                    # Add to chat history
                    st.session_state.messages_dual.append({
                        "role": "assistant",
                        "content": full_response_dual
                    })
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to Ollama. Make sure it's running on " + ollama_url)
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Clear dual chat button
    if st.session_state.messages_dual:
        if st.button("🗑️ Clear Dual Chat History", key="clear_dual"):
            st.session_state.messages_dual = []
            st.rerun()
    
    # Example questions
    with st.expander("💡 Example Questions for Dual Images"):
        st.markdown("""
        - "What are the main differences between these two images?"
        - "Compare and contrast the images on the left and right"
        - "Which image has better quality?"
        - "Describe what you see in each image"
        - "What's similar and what's different?"
        - "Which image is more professional looking?"
        """)

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Powered by Ollama Vision Models | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
