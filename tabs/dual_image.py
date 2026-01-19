import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
from utils import combine_images_side_by_side

class DualImageTab:
    def __init__(self, ollama_url, model_name, temperature, enable_thinking_api=False, show_thinking=True,
                 context_limit=0, repeat_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0, top_p=0.9):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        self.enable_thinking_api = enable_thinking_api
        self.show_thinking = show_thinking
        self.context_limit = context_limit
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        
        # Initialize session state
        if "messages_dual" not in st.session_state:
            st.session_state.messages_dual = []
        if "combined_image_b64" not in st.session_state:
            st.session_state.combined_image_b64 = None
        if "combined_image_pil" not in st.session_state:
            st.session_state.combined_image_pil = None
        if "dual_image1_rotation" not in st.session_state:
            st.session_state.dual_image1_rotation = 0
        if "dual_image2_rotation" not in st.session_state:
            st.session_state.dual_image2_rotation = 0
    
    def render(self):
        """Render the dual image tab"""
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First Image")
            image1 = st.file_uploader(
                "Upload first image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="dual_image1",
                label_visibility="collapsed"
            )
            if image1:
                # Rotation controls
                r1, r2, r3 = st.columns(3)
                with r1:
                    if st.button("↺ 90°", key="dual_img1_rot90"):
                        st.session_state.dual_image1_rotation = (st.session_state.dual_image1_rotation - 90) % 360
                with r2:
                    if st.button("↻ 90°", key="dual_img1_rotn90"):
                        st.session_state.dual_image1_rotation = (st.session_state.dual_image1_rotation + 90) % 360
                with r3:
                    if st.button("Reset", key="dual_img1_reset"):
                        st.session_state.dual_image1_rotation = 0
                
                if st.session_state.dual_image1_rotation != 0:
                    st.caption(f"Rotation: {st.session_state.dual_image1_rotation}°")
                
                st.image(image1, caption="First Image", use_container_width=True)
        
        with col2:
            st.markdown("### Second Image")
            image2 = st.file_uploader(
                "Upload second image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="dual_image2",
                label_visibility="collapsed"
            )
            if image2:
                # Rotation controls
                r1, r2, r3 = st.columns(3)
                with r1:
                    if st.button("↺ 90°", key="dual_img2_rot90"):
                        st.session_state.dual_image2_rotation = (st.session_state.dual_image2_rotation - 90) % 360
                with r2:
                    if st.button("↻ 90°", key="dual_img2_rotn90"):
                        st.session_state.dual_image2_rotation = (st.session_state.dual_image2_rotation + 90) % 360
                with r3:
                    if st.button("Reset", key="dual_img2_reset"):
                        st.session_state.dual_image2_rotation = 0
                
                if st.session_state.dual_image2_rotation != 0:
                    st.caption(f"Rotation: {st.session_state.dual_image2_rotation}°")
                
                st.image(image2, caption="Second Image", use_container_width=True)
        
        if image1 and image2:
            image1_bytes = image1.read()
            image2_bytes = image2.read()
            image1.seek(0)
            image2.seek(0)
            
            # Apply rotation to image 1
            if st.session_state.dual_image1_rotation != 0:
                img1 = Image.open(io.BytesIO(image1_bytes))
                img1 = img1.rotate(-st.session_state.dual_image1_rotation, expand=True)
                buffer1 = io.BytesIO()
                img1.save(buffer1, format=img1.format if img1.format else 'PNG')
                image1_bytes = buffer1.getvalue()
            
            # Apply rotation to image 2
            if st.session_state.dual_image2_rotation != 0:
                img2 = Image.open(io.BytesIO(image2_bytes))
                img2 = img2.rotate(-st.session_state.dual_image2_rotation, expand=True)
                buffer2 = io.BytesIO()
                img2.save(buffer2, format=img2.format if img2.format else 'PNG')
                image2_bytes = buffer2.getvalue()
            
            combined_bytes, combined_pil = combine_images_side_by_side(image1_bytes, image2_bytes)
            st.session_state.combined_image_pil = combined_pil
            st.session_state.combined_image_b64 = base64.b64encode(combined_bytes).decode('utf-8')
        
        st.markdown("---")
    
    def _render_combined_view(self):
        """Render the combined image view"""
        st.header("Combined View")
        if st.session_state.combined_image_pil:
            st.image(st.session_state.combined_image_pil, caption="Images Side by Side", use_container_width=True)
        else:
            st.info("Upload both images to see the combined view")
    
    def _render_chat_section(self):
        """Render the chat interface"""
        st.header("Chat")
        
        # System prompt setting
        with st.expander("💬 Custom System Prompt (optional)", expanded=False):
            system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get("system_prompt_dual", ""),
                height=100,
                key="system_prompt_input_dual",
                placeholder="Enter a custom system prompt to guide the model's behavior...\n\nExample: You are a helpful assistant that compares and contrasts images with precision.",
                help="Set a custom system prompt to control how the model responds.",
                label_visibility="collapsed"
            )
            st.session_state.system_prompt_dual = system_prompt
        
        # Display chat history
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages_dual:
                with st.chat_message(message["role"]):
                    # Show thinking if available and enabled
                    if message["role"] == "assistant" and message.get("thinking") and self.show_thinking:
                        with st.expander("🧠 View Thinking Process", expanded=False):
                            st.markdown(message["thinking"])
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
        
        # Add system prompt if provided
        if st.session_state.get('system_prompt_dual', '').strip():
            messages.append({
                "role": "system",
                "content": st.session_state.system_prompt_dual.strip()
            })
        
        # Apply context limit to prevent repetition
        history = st.session_state.messages_dual
        if self.context_limit > 0 and len(history) > self.context_limit:
            history = history[-self.context_limit:]
        
        # Add conversation history
        for i, msg in enumerate(history):
            # Get the original index for image attachment logic
            original_i = st.session_state.messages_dual.index(msg)
            if msg["role"] == "user":
                user_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                # Only include image on first message
                if original_i == 0:
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
                "temperature": self.temperature,
                "repeat_penalty": self.repeat_penalty,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "top_p": self.top_p
            },
            "think": self.enable_thinking_api
        }
        
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                message_placeholder = st.empty()
        
        full_thinking = ""
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
                        
                        # Handle thinking/reasoning output
                        if self.enable_thinking_api and "message" in chunk and "thinking" in chunk["message"]:
                            full_thinking += chunk["message"]["thinking"]
                            if full_thinking and self.show_thinking:
                                thinking_placeholder.markdown(f"**🧠 Thinking:**\n\n{full_thinking}▌")
                        
                        # Handle regular content
                        if "message" in chunk and "content" in chunk["message"]:
                            full_response += chunk["message"]["content"]
                            if full_thinking and self.show_thinking:
                                thinking_placeholder.markdown(f"**🧠 Thinking:**\n\n{full_thinking}")
                            message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
            
            # Final display
            if full_thinking and self.show_thinking:
                with thinking_placeholder.expander("🧠 View Thinking Process", expanded=False):
                    st.markdown(full_thinking)
                thinking_placeholder = st.empty()  # Clear the thinking placeholder
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages_dual.append({
                "role": "assistant",
                "content": full_response,
                "thinking": full_thinking if full_thinking else None
            })
            
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
