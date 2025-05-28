import replicate
import cv2
import os
import numpy as np
import base64
from dotenv import load_dotenv

load_dotenv()
print("API_KEY:", os.getenv("REPLICATE_API_TOKEN"))


def detect_edges(image_path, output_path):
    """Detects edges using Canny edge detection and saves the sketch."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Apply Canny edge detection
        edges = cv2.Canny(img, 100, 200)
        
        # Invert colors (white background, black edges) for sketch effect
        sketch = cv2.bitwise_not(edges)
        
        cv2.imwrite(output_path, sketch)
        print(f"Sketch saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in edge detection: {e}")
        return None

def generate_illustration(sketch_path, prompt):
    """Generates an illustration from a sketch using Replicate API."""
    try:
        # The R script should have loaded .env, so os.getenv should work
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
             raise ValueError("REPLICATE_API_TOKEN environment variable not set. Ensure .env file is loaded correctly in R.")

        # Encode the sketch image to base64 data URI
        with open(sketch_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded_string}"

        # Using a ControlNet scribble model
        model_version = "jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117"
        
        print(f"Sending sketch to Replicate model: {model_version}...")
        
        # Initialize client (it reads token from env var by default)
        client = replicate.Client(api_token=api_token)

        output = client.run(
            model_version,
            input={
                "image": data_uri, 
                "prompt": prompt,
                "num_samples": "1",
                "image_resolution": "512",
                "scale": 7.5,
            }
        )
        
        print("Illustration generated successfully.")
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            print(f"Unexpected output format from Replicate: {output}")
            return None
            
    except Exception as e:
        print(f"Error calling Replicate API: {e}")
        return None
    

def main():
    sketch_path = "my_sketch.png"
    prompt = "A vibrant, artistic representation of car efficiency trends, perhaps using stylized vehicles or abstract energy flows."
    illustration_path = generate_illustration(sketch_path, prompt)
    print(f"Illustration saved to: {illustration_path}")

if __name__ == "__main__":
    main()


