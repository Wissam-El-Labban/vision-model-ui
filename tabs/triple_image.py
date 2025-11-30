import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
from utils import combine_three_images_side_by_side

class TripleImageTab:
    def __init__(self, ollama_url, model_name, temperature):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize session state
        if "messages_triple" not in st.session_state:
            st.session_state.messages_triple = []
        if "combined_image_triple_b64" not in st.session_state:
            st.session_state.combined_image_triple_b64 = None
        if "combined_image_triple_pil" not in st.session_state:
            st.session_state.combined_image_triple_pil = None
        if "triple_image1_rotation" not in st.session_state:
            st.session_state.triple_image1_rotation = 0
        if "triple_image2_rotation" not in st.session_state:
            st.session_state.triple_image2_rotation = 0
        if "triple_image3_rotation" not in st.session_state:
            st.session_state.triple_image3_rotation = 0
    
    def render(self):
        """Render the triple image tab"""
        # Upload section at the top
        self._render_upload_inputs()
        
        # Combined view and chat side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_combined_view()
        
        with col2:
            self._render_chat_section()
    
    def _render_upload_inputs(self):
        """Render the image upload inputs at the top"""
        st.header("Upload Three Images")
        
        # Add custom CSS
        st.markdown("""
            <style>
            [data-testid="stFileUploader"] {
                border: 3px dashed #FF9800;
                border-radius: 10px;
                padding: 15px;
                background-color: rgba(255, 152, 0, 0.05);
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #F57C00;
                background-color: rgba(255, 152, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### First Image")
            image1 = st.file_uploader(
                "Upload first image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="triple_image1",
                label_visibility="collapsed"
            )
            if image1:
                # Rotation controls
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("↻ 90°", key="tri_img1_rot"):
                        st.session_state.triple_image1_rotation = (st.session_state.triple_image1_rotation + 90) % 360
                with r2:
                    if st.button("Reset", key="tri_img1_reset"):
                        st.session_state.triple_image1_rotation = 0
                if st.session_state.triple_image1_rotation != 0:
                    st.caption(f"{st.session_state.triple_image1_rotation}°")
                st.image(image1, caption="First Image", use_container_width=True)
        
        with col2:
            st.markdown("### Second Image")
            image2 = st.file_uploader(
                "Upload second image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="triple_image2",
                label_visibility="collapsed"
            )
            if image2:
                # Rotation controls
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("↻ 90°", key="tri_img2_rot"):
                        st.session_state.triple_image2_rotation = (st.session_state.triple_image2_rotation + 90) % 360
                with r2:
                    if st.button("Reset", key="tri_img2_reset"):
                        st.session_state.triple_image2_rotation = 0
                if st.session_state.triple_image2_rotation != 0:
                    st.caption(f"{st.session_state.triple_image2_rotation}°")
                st.image(image2, caption="Second Image", use_container_width=True)
        
        with col3:
            st.markdown("### Third Image")
            image3 = st.file_uploader(
                "Upload third image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="triple_image3",
                label_visibility="collapsed"
            )
            if image3:
                # Rotation controls
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("↻ 90°", key="tri_img3_rot"):
                        st.session_state.triple_image3_rotation = (st.session_state.triple_image3_rotation + 90) % 360
                with r2:
                    if st.button("Reset", key="tri_img3_reset"):
                        st.session_state.triple_image3_rotation = 0
                if st.session_state.triple_image3_rotation != 0:
                    st.caption(f"{st.session_state.triple_image3_rotation}°")
                st.image(image3, caption="Third Image", use_container_width=True)
        
        if image1 and image2 and image3:
            image1_bytes = image1.read()
            image2_bytes = image2.read()
            image3_bytes = image3.read()
            image1.seek(0)
            image2.seek(0)
            image3.seek(0)
            
            # Apply rotation to image 1
            if st.session_state.triple_image1_rotation != 0:
                img1 = Image.open(io.BytesIO(image1_bytes))
                img1 = img1.rotate(-st.session_state.triple_image1_rotation, expand=True)
                buffer1 = io.BytesIO()
                img1.save(buffer1, format=img1.format if img1.format else 'PNG')
                image1_bytes = buffer1.getvalue()
            
            # Apply rotation to image 2
            if st.session_state.triple_image2_rotation != 0:
                img2 = Image.open(io.BytesIO(image2_bytes))
                img2 = img2.rotate(-st.session_state.triple_image2_rotation, expand=True)
                buffer2 = io.BytesIO()
                img2.save(buffer2, format=img2.format if img2.format else 'PNG')
                image2_bytes = buffer2.getvalue()
            
            # Apply rotation to image 3
            if st.session_state.triple_image3_rotation != 0:
                img3 = Image.open(io.BytesIO(image3_bytes))
                img3 = img3.rotate(-st.session_state.triple_image3_rotation, expand=True)
                buffer3 = io.BytesIO()
                img3.save(buffer3, format=img3.format if img3.format else 'PNG')
                image3_bytes = buffer3.getvalue()
            
            combined_bytes, combined_pil = combine_three_images_side_by_side(image1_bytes, image2_bytes, image3_bytes)
            st.session_state.combined_image_triple_pil = combined_pil
            st.session_state.combined_image_triple_b64 = base64.b64encode(combined_bytes).decode('utf-8')
        
        st.markdown("---")
    
    def _render_combined_view(self):
        """Render the combined image view"""
        st.header("Combined View")
        if st.session_state.combined_image_triple_pil:
            st.image(st.session_state.combined_image_triple_pil, caption="Three Images Side by Side", use_container_width=True)
        else:
            st.info("Upload all three images to see the combined view")
    
    def _render_chat_section(self):
        """Render the chat interface"""
        st.header("Chat")
        
        # Display chat history
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages_triple:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about all three images..."):
            if not st.session_state.combined_image_triple_b64:
                st.error("Please upload all three images first!")
            else:
                self._handle_chat_input(prompt, chat_container)
        
        # Clear chat button
        if st.session_state.messages_triple:
            if st.button("🗑️ Clear Chat History", key="clear_triple"):
                st.session_state.messages_triple = []
                st.rerun()
    
    def _handle_chat_input(self, prompt, chat_container):
        """Handle user chat input and API call"""
        # Add user message to history
        st.session_state.messages_triple.append({
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
        for i, msg in enumerate(st.session_state.messages_triple):
            if msg["role"] == "user":
                user_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                # Only include image on first message
                if i == 0:
                    user_msg["images"] = [st.session_state.combined_image_triple_b64]
                
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
        if len(st.session_state.messages_triple) == 1:
            current_message["images"] = [st.session_state.combined_image_triple_b64]
        
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
            
            st.session_state.messages_triple.append({
                "role": "assistant",
                "content": full_response
            })
            
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
