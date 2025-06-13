from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import sys

from transformers import AutoTokenizer, AutoImageProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"


# Separate tokenizer and image processor
tokenizer = AutoTokenizer.from_pretrained(model_id)
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id)

# Read image path from command-line
if len(sys.argv) < 2:
    print("Usage: python describe_image_with_llava.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

# Prompt for captioning
prompt = "Describe the image in detail."

# Process and generate
print(f"Generating description for: {image_path}")
inputs = processor(prompt, image, return_tensors="pt").to(model.device, torch.float16)
output = model.generate(**inputs, max_new_tokens=100)

# Decode output
description = processor.decode(output[0], skip_special_tokens=True)
print("\n🖼️ Description:")
print(description)
