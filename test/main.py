import json
import os
from src.captioner import image_to_ai_description
from src.utils import list_images

def process_folder(input_folder="data", output_file="outputs/ai_captions.json"):
    """Process all images in a folder and generate comprehensive AI-optimized JSON."""
    print(f"Processing images from: {input_folder}")
    images = list_images(input_folder)
    
    if not images:
        print("No images found in the data folder.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process images and collect results
    results = {"images": []}
    
    print(f"Found {len(images)} images to process...")
    print("Generating comprehensive AI-optimized descriptions...\n")
    
    for i, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        print(f"[{i}/{len(images)}] Processing: {filename}")
        
        # Generate comprehensive AI description
        ai_result = image_to_ai_description(img_path)
        
        results["images"].append(ai_result)
        
        # Show preview of results
        print(f"  AI Description: {ai_result.get('ai_description', 'N/A')[:100]}...")
        print(f"  Tags: {', '.join(ai_result.get('tags', [])[:5])}")
        print(f"  Conversation Starter: {ai_result.get('chatbot_prompts', {}).get('conversation_starter', 'N/A')[:80]}...")
        print()
    
    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Results saved to: {output_file}")
    print(f"Generated AI-optimized descriptions for {len(results['images'])} images.")
    
    # Also create a simple version for backward compatibility
    simple_output = output_file.replace("ai_captions.json", "captions.json")
    simple_results = {"images": []}
    
    for img_data in results["images"]:
        simple_results["images"].append({
            "filename": img_data["filename"],
            "caption": img_data.get("captions", [{}])[0].get("text", "No caption available")
        })
    
    with open(simple_output, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, indent=2, ensure_ascii=False)
    
    print(f"Simple format also saved to: {simple_output}")

if __name__ == "__main__":
    process_folder()
