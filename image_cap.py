
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Specify your image filename
img_path = "lexus.jpg"

# Open the image and convert it to RGB
image = Image.open(img_path).convert("RGB")

# Prepare the image and text for BLIP
text = "the image of"

inputs = processor(
    images=image,
    text=text,
    return_tensors="pt"
)

# Generate the caption
outputs = model.generate(
    **inputs,
    max_length=50
)

# Convert generated tokens into readable text
caption = processor.decode(
    outputs[0],
    skip_special_tokens=True
)

# Display the result
print(caption)
