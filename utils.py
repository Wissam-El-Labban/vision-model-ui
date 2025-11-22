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

@st.cache_data(ttl=60)
def get_available_models(url):
    """Get available vision models from Ollama"""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            vision_keywords = ["vision", "vl", "llava", "qwen", "moondream", "minicpm"]
            vision_models = [
                m["name"] for m in models 
                if any(keyword in m["name"].lower() for keyword in vision_keywords)
            ]
            return sorted(vision_models) if vision_models else []
        return []
    except:
        return []

# Register cleanup function
atexit.register(cleanup_ollama_models)
