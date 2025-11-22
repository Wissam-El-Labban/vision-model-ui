import streamlit as st
import requests
import json
from utils import encode_image_to_base64, combine_images_side_by_side

class DualImageTab:
    def __init__(self, ollama_url, model_name, temperature):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize session state
        if "messages_dual" not in st.session_state:
            st.session_state.messages_dual = []
        if "combined_image_b64" not in st.session_state:
            st.session_state.combined_image_b64 = None
        if "combined_image_pil" not in st.session_state:
            st.session_state.combined_image_pil = None
    
    def render(self):
        """Render the dual image tab"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_upload_section()
        
        with col2:
            self._render_chat_section()
    
    def _render_upload_section(self):
        """Render the dual image upload section"""
        st.header("Upload Two Images")
        
        # Add custom CSS
        st.markdown("""
            <style>
            [data-testid="stFileUploader"] {
                border: 3px dashed #2196F3;
                border-radius: 10px;
                padding: 20px;
                background-color: rgba(33, 150, 243, 0.05);
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #1976D2;
                background-color: rgba(33, 150, 243, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### First Image")
        image1 = st.file_uploader(
            "Upload first image",
            type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
            key="dual_image1",
            label_visibility="collapsed"
        )
        
        st.markdown("### Second Image")
        image2 = st.file_uploader(
            "Upload second image",
            type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
            key="dual_image2",
            label_visibility="collapsed"
        )
        
        if image1 and image2:
            combined_pil = combine_images_side_by_side(image1, image2)
            st.session_state.combined_image_pil = combined_pil
            st.session_state.combined_image_b64 = encode_image_to_base64(combined_pil)
            
            st.markdown("### Combined View")
            st.image(combined_pil, caption="Images Side by Side", use_container_width=True)
        elif st.session_state.combined_image_pil:
            st.markdown("### Combined View")
            st.image(st.session_state.combined_image_pil, caption="Images Side by Side", use_container_width=True)
    
    def _render_chat_section(self):
        """Render the chat interface"""
        st.header("Chat")
        
        # Display chat history
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages_dual:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about both images..."):
            if not st.session_state.combined_image_b64:
                st.error("Please upload both images first!")
            else:
                self._handle_chat_input(prompt, chat_container)
        
        # Clear chat button
        if st.session_state.messages_dual:
            if st.button("🗑️ Clear Chat History", key="clear_dual"):
                st.session_state.messages_dual = []
                st.rerun()
    
    def _handle_chat_input(self, prompt, chat_container):
        """Handle user chat input and API call"""
        # Add user message to history
        st.session_state.messages_dual.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Build messages for API
        messages = self._build_messages(prompt)
        
        # Call API
        try:
            self._call_ollama_api(messages, chat_container)
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to Ollama. Make sure it's running on {self.ollama_url}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    def _build_messages(self, prompt):
        """Build messages array for API call"""
        messages = []
        
        # Add conversation history
        for i, msg in enumerate(st.session_state.messages_dual):
            if msg["role"] == "user":
                user_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                # Only include image on first message
                if i == 0:
                    user_msg["images"] = [st.session_state.combined_image_b64]
                
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
        
        # Only include image on first message
        if len(st.session_state.messages_dual) == 1:
            current_message["images"] = [st.session_state.combined_image_b64]
        
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
            
            st.session_state.messages_dual.append({
                "role": "assistant",
                "content": full_response
            })
            
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
