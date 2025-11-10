import streamlit as st
import requests
import base64
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Qwen3-VL Image Analysis",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Qwen3-VL Image Analysis")
st.markdown("Upload an image and ask questions about it using the Qwen3-VL:4b model")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    ollama_url = st.text_input("Ollama API URL", value="http://localhost:11434")
    model_name = st.text_input("Model Name", value="qwen3-vl:4b")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    st.markdown("### About")
    st.markdown("This app uses Ollama's Qwen3-VL model to analyze images and answer questions.")
    st.markdown("Make sure Ollama is running and the model is installed:")
    st.code("ollama pull qwen3-vl:4b", language="bash")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_b64" not in st.session_state:
    st.session_state.current_image_b64 = None

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        help="Upload an image to analyze"
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
    
    # Display chat history
    chat_container = st.container(height=400)
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
            
            # Call Ollama API
            with st.spinner("Analyzing image..."):
                try:
                    api_endpoint = f"{ollama_url}/api/generate"
                    
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "images": [st.session_state.current_image_b64],
                        "stream": False,
                        "options": {
                            "temperature": temperature
                        }
                    }
                    
                    response = requests.post(api_endpoint, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        result = response.json()
                        assistant_response = result.get("response", "No response received")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_response
                        })
                        
                        # Display assistant response
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(assistant_response)
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
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Powered by Qwen3-VL and Ollama | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
