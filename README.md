# PerceptionUnderConstraint  
**Stress-Testing Vision-Language Models Under Real-World Occlusion**

---

## Objective

This project investigates how vision-language models behave when visual input departs from idealised conditions. Using simulated ski environments that introduce occlusion, glare, and exposure shifts, I evaluate the degradation of facial embeddings and examine whether this degradation differs across skin tone groups. The work bridges technical robustness testing with ethical AI considerations, showing how environmental messiness translates into governance-relevant risk.

---

## Project Motivation

Most computer vision systems are evaluated on clean, front-facing, well-lit images. These conditions rarely reflect real-world use. Faces are obscured by equipment, lighting is harsh or inconsistent, and environments introduce noise that models are not explicitly trained to handle.

The core question guiding this project was simple:

**How does AI perception fail when the environment becomes adversarial rather than cooperative?**

This is not only a technical concern. When AI systems are used for identification, access, or assessment, fragile perception becomes an ethical and societal issue.

---

## Dataset Overview

- **Total images:** 60  
  - 30 darker skin  
  - 30 lighter skin  

Each subject has:
- One reference image (baseline condition)
- One ski-simulated image (constrained condition)

### Simulated Ski Conditions

The following constraints were applied to approximate real skiing environments:

- Partial facial occlusion (helmet and goggles)
- Reduced brightness (overcast or shaded snow)
- Increased contrast and glare
- Cold-environment exposure and colour shifts

These transformations were designed to stress model robustness rather than maximise visual realism.

---

## Methodology

### Model

- **Model:** CLIP (ViT-B/32)
- **Task:** Image embedding similarity under environmental constraint

### Pipeline

1. Encode reference image into an embedding  
2. Encode ski-simulated image into an embedding  
3. Compute cosine similarity between embeddings  
4. Apply a similarity threshold to detect degradation  
5. Compare results across skin tone groups  

---

## Implementation

### Embedding Extraction

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b)
