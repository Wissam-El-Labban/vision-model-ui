import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
from utils import encode_image_to_base64

class SingleImageTab:
    def __init__(self, ollama_url, model_name, temperature, enable_thinking_api=False, show_thinking=True,
                 enable_anti_repetition=False, context_limit=0, repeat_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0, top_p=0.9, num_ctx=32768):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        self.enable_thinking_api = enable_thinking_api
        self.show_thinking = show_thinking
        self.enable_anti_repetition = enable_anti_repetition
        self.context_limit = context_limit
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.num_ctx = num_ctx
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
        if "current_image_b64" not in st.session_state:
            st.session_state.current_image_b64 = None
        if "pending_attachment" not in st.session_state:
            st.session_state.pending_attachment = None
        if "single_image_rotation" not in st.session_state:
            st.session_state.single_image_rotation = 0
        if "system_image_single_b64" not in st.session_state:
            st.session_state.system_image_single_b64 = None
        if "system_image_single_name" not in st.session_state:
            st.session_state.system_image_single_name = None
    
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
            # Rotation controls
            st.markdown("**Rotate Image:**")
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                if st.button("↺ 90°", key="rotate_single_90"):
                    st.session_state.single_image_rotation = (st.session_state.single_image_rotation - 90) % 360
            with col_r2:
                if st.button("↻ 90°", key="rotate_single_neg90"):
                    st.session_state.single_image_rotation = (st.session_state.single_image_rotation + 90) % 360
            with col_r3:
                if st.button("↻ 180°", key="rotate_single_180"):
                    st.session_state.single_image_rotation = (st.session_state.single_image_rotation + 180) % 360
            with col_r4:
                if st.button("Reset", key="rotate_single_reset"):
                    st.session_state.single_image_rotation = 0
            
            if st.session_state.single_image_rotation != 0:
                st.caption(f"Current rotation: {st.session_state.single_image_rotation}°")
            
            # Read and rotate image
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            if st.session_state.single_image_rotation != 0:
                # Apply rotation
                image = Image.open(io.BytesIO(image_bytes))
                rotated_image = image.rotate(-st.session_state.single_image_rotation, expand=True)
                
                # Convert back to bytes
                buffer = io.BytesIO()
                rotated_image.save(buffer, format=image.format if image.format else 'PNG')
                image_bytes = buffer.getvalue()
            
            # Display rotated image
            st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
            
            # Encode rotated image
            st.session_state.current_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            st.session_state.current_image = uploaded_file.name
        elif st.session_state.current_image:
            st.info(f"Current image: {st.session_state.current_image}")
    
    def _render_chat_section(self):
        """Render the chat interface"""
        st.header("Chat")
        
        # System prompt setting
        with st.expander("💬 Custom System Prompt (optional)", expanded=False):
            system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get("system_prompt_single", ""),
                height=100,
                key="system_prompt_input_single",
                placeholder="Enter a custom system prompt to guide the model's behavior...\n\nExample: You are a helpful assistant that provides detailed image analysis with technical accuracy.",
                help="Set a custom system prompt to control how the model responds.",
                label_visibility="collapsed"
            )
            st.session_state.system_prompt_single = system_prompt
            
            st.markdown("**📎 Attach Image to System Prompt (optional)**")
            st.caption("This image will be sent with every message to provide persistent visual context.")
            
            system_image = st.file_uploader(
                "System prompt image",
                type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
                key="system_image_single",
                help="Upload an image to attach to the system prompt. It will be sent with every message.",
                label_visibility="collapsed"
            )
            
            if system_image:
                st.session_state.system_image_single_b64 = encode_image_to_base64(system_image)
                st.session_state.system_image_single_name = system_image.name
                st.image(system_image, caption=f"System Image: {system_image.name}", width=200)
            elif st.session_state.system_image_single_b64:
                st.image(f"data:image/png;base64,{st.session_state.system_image_single_b64}", 
                        caption=f"System Image: {st.session_state.system_image_single_name}", width=200)
                if st.button("🗑️ Remove System Image", key="remove_system_image_single"):
                    st.session_state.system_image_single_b64 = None
                    st.session_state.system_image_single_name = None
                    st.rerun()
        
        # Display chat history
        chat_container = st.container(height=600)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Show thinking if available and enabled
                    if message["role"] == "assistant" and message.get("thinking") and self.show_thinking:
                        with st.expander("🧠 View Thinking Process", expanded=False):
                            st.markdown(message["thinking"])
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
        
        # Add system prompt if provided
        if st.session_state.get('system_prompt_single', '').strip():
            system_msg = {
                "role": "system",
                "content": st.session_state.system_prompt_single.strip()
            }
            # Add system image if attached
            if st.session_state.system_image_single_b64:
                system_msg["images"] = [st.session_state.system_image_single_b64]
            messages.append(system_msg)
        
        # Apply context limit to prevent repetition (only if enabled)
        history = st.session_state.messages
        if self.enable_anti_repetition and self.context_limit > 0 and len(history) > self.context_limit:
            history = history[-self.context_limit:]
        
        # Add conversation history
        for i, msg in enumerate(history):
            # Get the original index for image attachment logic
            original_i = st.session_state.messages.index(msg)
            if msg["role"] == "user":
                user_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                
                if original_i == 0:
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
                "temperature": self.temperature,
                "num_ctx": self.num_ctx
            },
            "think": self.enable_thinking_api
        }
        
        # Add anti-repetition parameters only if enabled
        if self.enable_anti_repetition:
            payload["options"]["repeat_penalty"] = self.repeat_penalty
            payload["options"]["frequency_penalty"] = self.frequency_penalty
            payload["options"]["presence_penalty"] = self.presence_penalty
            payload["options"]["top_p"] = self.top_p
        
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
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "thinking": full_thinking if full_thinking else None
            })
            
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
