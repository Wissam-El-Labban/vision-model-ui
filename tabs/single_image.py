import streamlit as st
import requests
import json
import base64
from utils import encode_image_to_base64

class SingleImageTab:
    def __init__(self, ollama_url, model_name, temperature):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
        if "current_image_b64" not in st.session_state:
            st.session_state.current_image_b64 = None
        if "pending_attachment" not in st.session_state:
            st.session_state.pending_attachment = None
    
    def render(self):
        """Render the single image tab"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_upload_section()
        
        with col2:
            self._render_chat_section()
    
    def _render_upload_section(self):
        """Render the image upload section"""
        st.header("Upload Image")
        
        # Add custom CSS
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
            label_visibility="collapsed",
            key="single_image_uploader"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image_bytes = uploaded_file.read()
            st.session_state.current_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            st.session_state.current_image = uploaded_file.name
            uploaded_file.seek(0)
        elif st.session_state.current_image:
            st.info(f"Current image: {st.session_state.current_image}")
    
    def _render_chat_section(self):
        """Render the chat interface"""
        st.header("Chat")
        
        # Display chat history
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "attached_image" in message and message["attached_image"]:
                        st.image(message["attached_image"], width=200, caption="Attached image")
        
        # Optional image attachment
        with st.expander("📎 Attach additional image (optional)", expanded=False):
            additional_image = st.file_uploader(
                "Add context image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="additional_image_single",
                help="Upload an additional image to include with your next question"
            )
            if additional_image:
                st.session_state.pending_attachment = additional_image
                st.image(additional_image, width=200, caption="Will be attached to next message")
            elif st.session_state.pending_attachment is None and "additional_image_single" in st.session_state:
                st.session_state.pending_attachment = None
        
        attachment_to_use = st.session_state.pending_attachment
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the image..."):
            if not st.session_state.current_image_b64:
                st.error("Please upload an image first!")
            else:
                self._handle_chat_input(prompt, attachment_to_use, chat_container)
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", key="clear_single"):
                st.session_state.messages = []
                st.rerun()
    
    def _handle_chat_input(self, prompt, attachment_to_use, chat_container):
        """Handle user chat input and API call"""
        additional_image_b64 = None
        if attachment_to_use:
            additional_image_b64 = encode_image_to_base64(attachment_to_use)
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "attached_image": attachment_to_use if attachment_to_use else None
        })
        
        st.session_state.pending_attachment = None
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                if attachment_to_use:
                    st.image(attachment_to_use, width=200, caption="Attached image")
        
        # Build messages for API
        messages = self._build_messages(prompt, additional_image_b64)
        
        # Call API
        try:
            self._call_ollama_api(messages, chat_container)
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to Ollama. Make sure it's running on {self.ollama_url}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    def _build_messages(self, prompt, additional_image_b64):
        """Build messages array for API call"""
        messages = []
        
        # Add conversation history
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                user_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                if i == 0:
                    user_msg["images"] = [st.session_state.current_image_b64]
                elif "attached_image" in msg and msg["attached_image"] is not None:
                    try:
                        attached_b64 = encode_image_to_base64(msg["attached_image"])
                        user_msg["images"] = [
                            st.session_state.current_image_b64,
                            attached_b64
                        ]
                    except:
                        pass
                
                messages.append(user_msg)
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        current_message = {
            "role": "user",
            "content": prompt
        }
        
        if len(st.session_state.messages) == 1:
            current_message["images"] = [st.session_state.current_image_b64]
        elif additional_image_b64:
            current_message["images"] = [
                st.session_state.current_image_b64,
                additional_image_b64
            ]
        
        messages.append(current_message)
        return messages
    
    def _call_ollama_api(self, messages, chat_container):
        """Call Ollama API and stream response"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature
            }
        }
        
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
        
        full_response = ""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            full_response += chunk["message"]["content"]
                            message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
