import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import re
from datetime import datetime

class AdvancedImageCaptioner:
    """Advanced image captioning class with comprehensive analysis for AI chatbots."""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """Initialize the captioner with a pre-trained BLIP model."""
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def extract_tags_from_caption(self, caption: str) -> list:
        """Extract relevant tags from the caption for better AI understanding."""
        # Common objects and activities that might appear in images
        tag_patterns = {
            # People and actions
            'people': r'\b(?:person|people|man|woman|child|boy|girl|baby|adult)\b',
            'actions': r'\b(?:running|walking|sitting|standing|jumping|dancing|playing|working|exercising|cooking|reading|writing|driving|riding|swimming|flying|climbing|lifting|pushing|pulling|throwing|catching|kicking|hitting)\b',
            'sports': r'\b(?:football|basketball|tennis|soccer|baseball|golf|hockey|volleyball|boxing|wrestling|cycling|skiing|surfing|skateboarding|yoga|gym|fitness|workout|exercise|bench press|squats|deadlift)\b',
            'locations': r'\b(?:park|beach|street|road|building|house|office|school|hospital|restaurant|store|mall|gym|stadium|field|court|track|pool|lake|river|mountain|forest|desert|city|town|village)\b',
            'objects': r'\b(?:car|truck|bike|bicycle|motorcycle|bus|train|plane|boat|chair|table|bed|computer|phone|camera|book|ball|bottle|cup|plate|food|tree|flower|animal|dog|cat|bird)\b',
            'weather': r'\b(?:sunny|cloudy|rainy|snowy|foggy|windy|storm|clear|bright|dark|day|night|morning|afternoon|evening|sunset|sunrise)\b',
            'colors': r'\b(?:red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|silver|gold)\b',
            'emotions': r'\b(?:happy|sad|angry|excited|surprised|calm|peaceful|energetic|tired|focused|concentrated|relaxed)\b'
        }
        
        tags = []
        caption_lower = caption.lower()
        
        for category, pattern in tag_patterns.items():
            matches = re.findall(pattern, caption_lower)
            tags.extend(matches)
        
        # Remove duplicates and return unique tags
        return list(set(tags))
    
    def analyze_scene_context(self, caption: str) -> dict:
        """Analyze the scene context for better AI understanding."""
        caption_lower = caption.lower()
        
        context = {
            "setting": "unknown",
            "time_of_day": "unknown",
            "activity_level": "unknown",
            "social_context": "unknown",
            "mood": "neutral"
        }
        
        # Determine setting
        if any(word in caption_lower for word in ['indoor', 'inside', 'room', 'office', 'house', 'building']):
            context["setting"] = "indoor"
        elif any(word in caption_lower for word in ['outdoor', 'outside', 'park', 'street', 'beach', 'field', 'forest']):
            context["setting"] = "outdoor"
        
        # Determine time of day
        if any(word in caption_lower for word in ['night', 'evening', 'dark']):
            context["time_of_day"] = "night"
        elif any(word in caption_lower for word in ['morning', 'sunrise']):
            context["time_of_day"] = "morning"
        elif any(word in caption_lower for word in ['afternoon', 'day', 'sunny', 'bright']):
            context["time_of_day"] = "day"
        elif any(word in caption_lower for word in ['sunset', 'dusk']):
            context["time_of_day"] = "evening"
        
        # Determine activity level
        if any(word in caption_lower for word in ['running', 'jumping', 'dancing', 'playing', 'exercising', 'working out']):
            context["activity_level"] = "high"
        elif any(word in caption_lower for word in ['walking', 'standing', 'working']):
            context["activity_level"] = "medium"
        elif any(word in caption_lower for word in ['sitting', 'lying', 'sleeping', 'resting']):
            context["activity_level"] = "low"
        
        # Determine social context
        if any(word in caption_lower for word in ['group', 'people', 'crowd', 'team', 'family']):
            context["social_context"] = "group"
        elif any(word in caption_lower for word in ['person', 'man', 'woman', 'individual']):
            context["social_context"] = "individual"
        
        # Determine mood
        if any(word in caption_lower for word in ['smiling', 'happy', 'celebrating', 'laughing']):
            context["mood"] = "positive"
        elif any(word in caption_lower for word in ['focused', 'concentrated', 'serious', 'determined']):
            context["mood"] = "focused"
        elif any(word in caption_lower for word in ['relaxed', 'calm', 'peaceful']):
            context["mood"] = "calm"
        
        return context
    
    def generate_ai_optimized_description(self, image_path: str) -> dict:
        """Generate comprehensive image analysis optimized for AI chatbot prompts."""
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            
            # Get image dimensions for context
            width, height = image.size
            
            # Process the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate multiple captions with different parameters for variety
            captions = []
            
            # Standard caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)
            standard_caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append({
                "type": "standard",
                "text": standard_caption,
                "confidence": 0.85  # Estimated confidence
            })
            
            # Detailed caption with higher max length
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=75, num_beams=3, temperature=0.7, do_sample=True)
            detailed_caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append({
                "type": "detailed",
                "text": detailed_caption,
                "confidence": 0.80
            })
            
            # Use the standard caption as primary
            primary_caption = standard_caption
            
            # Extract tags and analyze context
            tags = self.extract_tags_from_caption(primary_caption)
            scene_context = self.analyze_scene_context(primary_caption)
            
            # Generate AI-optimized description
            ai_description = self.create_ai_description(primary_caption, scene_context, tags)
            
            # Create comprehensive result
            result = {
                "filename": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "image_properties": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 2),
                    "format": image.format or "JPEG"
                },
                "ai_description": ai_description,
                "captions": captions,
                "tags": sorted(tags),
                "scene_context": scene_context,
                "related_topics": self.generate_related_topics(tags, scene_context),
                "metadata": {
                    "model_used": "BLIP-base",
                    "processing_device": self.device,
                    "analysis_version": "1.0"
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {
                "filename": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "ai_description": "Unable to process this image. Please try again or use a different image.",
                "captions": [],
                "tags": [],
                "scene_context": {},
                "related_topics": ["image analysis", "technical support"]
            }
    
    def create_ai_description(self, caption: str, context: dict, tags: list) -> str:
        """Create a rich description optimized for AI understanding."""
        description_parts = [caption]
        
        # Add context information
        if context["setting"] != "unknown":
            description_parts.append(f"This appears to be an {context['setting']} scene.")
        
        if context["time_of_day"] != "unknown":
            description_parts.append(f"The time appears to be {context['time_of_day']}.")
        
        if context["activity_level"] != "unknown":
            description_parts.append(f"The activity level in the image is {context['activity_level']}.")
        
        if context["social_context"] != "unknown":
            if context["social_context"] == "group":
                description_parts.append("This involves multiple people or a group setting.")
            else:
                description_parts.append("This focuses on an individual person.")
        
        # Add relevant tags context
        if tags:
            key_tags = [tag for tag in tags if tag not in ['person', 'people', 'man', 'woman']]
            if key_tags:
                description_parts.append(f"Key elements include: {', '.join(key_tags[:5])}.")
        
        return " ".join(description_parts)
    

    def generate_related_topics(self, tags: list, context: dict) -> list:
        """Generate related topics for conversation."""
        topics = []
        
        # Add topics based on tags
        if any(tag in tags for tag in ['running', 'exercise', 'gym', 'fitness']):
            topics.extend(['fitness', 'health', 'exercise routines', 'nutrition', 'wellness'])
        
        if any(tag in tags for tag in ['beach', 'outdoor', 'park']):
            topics.extend(['outdoor activities', 'nature', 'travel', 'recreation'])
        
        if any(tag in tags for tag in ['sports', 'basketball', 'football', 'tennis']):
            topics.extend(['sports', 'athletics', 'competition', 'team activities'])
        
        # Add context-based topics
        if context["setting"] == "outdoor":
            topics.extend(['outdoor lifestyle', 'fresh air benefits'])
        
        if context["activity_level"] == "high":
            topics.extend(['active lifestyle', 'energy', 'motivation'])
        
        # Remove duplicates and return unique topics
        return list(set(topics))[:5]

# Global captioner instance (lazy loading)
_captioner = None

def get_captioner():
    """Get or create the global captioner instance."""
    global _captioner
    if _captioner is None:
        _captioner = AdvancedImageCaptioner()
    return _captioner

def image_to_caption(image_path: str) -> str:
    """Generate a simple caption for an image (backward compatibility)."""
    captioner = get_captioner()
    result = captioner.generate_ai_optimized_description(image_path)
    return result.get("captions", [{}])[0].get("text", "Unable to generate caption")

def image_to_ai_description(image_path: str) -> dict:
    """Generate comprehensive AI-optimized description for an image."""
    captioner = get_captioner()
    return captioner.generate_ai_optimized_description(image_path)