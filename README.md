# PerceptionUnderConstraint  
**Stress-Testing Vision-Language Models Under Real-World Occlusion**

---

## Objective

This project investigates how vision-language models behave when visual input departs from idealised conditions. Using simulated ski environments that introduce occlusion, glare, and exposure shifts, I evaluate the degradation of facial embeddings and examine whether this degradation differs across skin tone groups. The work bridges technical robustness testing with ethical AI considerations, showing how environmental messiness translates into governance-relevant risk.

---

## Project Motivation

## Project Motivation

AI is often treated as neutral and universal, yet its deployment intersects with lived experience, identity, and power. Drawing on *The Human Alignment* and Joy Buolamwini’s work on algorithmic bias, this project reflects on how AI systems—particularly facial recognition technologies like RIDs used at ski resorts for access and tracking—encode assumptions that may invisibilize or misidentify individuals from minority groups. bell hooks’ *Ain’t I a Woman?* reminds us that technology, like society, does not treat all bodies equally.

As someone positioned at the intersection of being both a consumer and potential subject of AI systems—profiting from technology yet vulnerable to its errors—this project explores how real-world constraints (goggles, glare, and lighting on ski slopes) expose fragility in AI perception. It asks: how does AI respond when it encounters bodies and faces that do not conform to its training assumptions, and what does this reveal about fairness, robustness, and ethical deployment?

The core question guiding this work is:

**How does AI perception fail when environmental and social realities challenge its assumptions?**

This is both a technical and ethical inquiry, bridging robustness testing with lived experience and critical theory, demonstrating that real-world AI is never abstract—it interacts with identity, power, and opportunity.

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
