import streamlit as st
import requests
import base64
from PIL import Image
import io
import atexit

def cleanup_ollama_models():
    """Unload all running Ollama models on exit"""
    try:
        ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/ps", timeout=5)
        if response.status_code == 200:
            running_models = response.json().get("models", [])
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

def encode_image_to_base64(image_file):
    """Convert uploaded image to base64"""
    image_bytes = image_file.read()
    image_file.seek(0)
    return base64.b64encode(image_bytes).decode('utf-8')

def combine_images_side_by_side(image1_bytes, image2_bytes):
    """Combine two images horizontally into one composite image"""
    try:
        img1 = Image.open(io.BytesIO(image1_bytes))
        img2 = Image.open(io.BytesIO(image2_bytes))
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        target_height = min(img1.height, img2.height)
        img1_new_width = int(img1.width * (target_height / img1.height))
        img2_new_width = int(img2.width * (target_height / img2.height))
        
        img1_resized = img1.resize((img1_new_width, target_height), Image.LANCZOS)
        img2_resized = img2.resize((img2_new_width, target_height), Image.LANCZOS)
        
        combined_width = img1_resized.width + img2_resized.width
        combined_image = Image.new('RGB', (combined_width, target_height))
        
        combined_image.paste(img1_resized, (0, 0))
        combined_image.paste(img2_resized, (img1_resized.width, 0))
        
        output = io.BytesIO()
        combined_image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.getvalue(), combined_image
    except Exception as e:
        st.error(f"Error combining images: {str(e)}")
        return None, None

def combine_three_images_side_by_side(image1_bytes, image2_bytes, image3_bytes):
    """Combine three images horizontally into one composite image"""
    try:
        img1 = Image.open(io.BytesIO(image1_bytes))
        img2 = Image.open(io.BytesIO(image2_bytes))
        img3 = Image.open(io.BytesIO(image3_bytes))
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        if img3.mode != 'RGB':
            img3 = img3.convert('RGB')
        
        target_height = min(img1.height, img2.height, img3.height)
        
        img1_new_width = int(img1.width * (target_height / img1.height))
        img2_new_width = int(img2.width * (target_height / img2.height))
        img3_new_width = int(img3.width * (target_height / img3.height))
        
        img1_resized = img1.resize((img1_new_width, target_height), Image.LANCZOS)
        img2_resized = img2.resize((img2_new_width, target_height), Image.LANCZOS)
        img3_resized = img3.resize((img3_new_width, target_height), Image.LANCZOS)
        
        combined_width = img1_resized.width + img2_resized.width + img3_resized.width
        combined_image = Image.new('RGB', (combined_width, target_height))
        
        combined_image.paste(img1_resized, (0, 0))
        combined_image.paste(img2_resized, (img1_resized.width, 0))
        combined_image.paste(img3_resized, (img1_resized.width + img2_resized.width, 0))
        
        output = io.BytesIO()
        combined_image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.getvalue(), combined_image
    except Exception as e:
        st.error(f"Error combining images: {str(e)}")
        return None, None

def is_vision_model(url, model_name):
    """Check if a model supports vision by inspecting its details"""
    try:
        response = requests.post(
            f"{url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        
        if response.status_code == 200:
            model_info = response.json()
            
            # Check multiple indicators of vision capability
            modelfile = model_info.get('modelfile', '').lower()
            template = model_info.get('template', '').lower()
            parameters = model_info.get('parameters', '').lower()
            
            # Vision models typically have:
            # 1. Image-related parameters in modelfile
            # 2. Vision-related keywords in template
            # 3. Multimodal projector parameters
            vision_indicators = [
                'image' in modelfile,
                'vision' in modelfile,
                'visual' in modelfile,
                'image' in template,
                '[img' in template,  # Common image token pattern
                'clip' in modelfile,  # CLIP vision encoder
                'mm_projector' in modelfile,  # Multimodal projector
                'vision_tower' in modelfile,
                'image_processor' in modelfile,
            ]
            
            return any(vision_indicators)
        return False
    except:
        return False

@st.cache_data(ttl=60)
def get_available_models(url):
    """Get available vision models from Ollama by checking model details"""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            if not models:
                return []
            
            vision_models = []
            
            # Check each model for vision capabilities
            for model in models:
                model_name = model["name"]
                
                # Check if model supports vision
                if is_vision_model(url, model_name):
                    vision_models.append(model_name)
            
            return sorted(vision_models) if vision_models else []
        return []
    except:
        return []

# Register cleanup function
atexit.register(cleanup_ollama_models)
