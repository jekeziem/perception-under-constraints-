# PerceptionUnderConstraint  
**Stress-Testing Vision-Language Models Under Ski Conditions**

---

## Objective

This project investigates how vision-language models behave when visual input departs from idealised conditions. Using simulated ski environments that introduce occlusion, glare, and exposure shifts, I evaluate the degradation of facial embeddings and examine whether this degradation differs across skin tone groups. The work bridges technical robustness testing with ethical AI considerations, showing how environmental messiness translates into governance-relevant risk.

---

## Project Motivation

Modern AI systems are often evaluated on clean benchmarks, yet real-world social environments are messy and contextual. Joy Buolamwini’s research on algorithmic bias demonstrated that commercial facial recognition systems showed significantly higher error rates for darker‑skinned women compared to lighter‑skinned men, this project reflects on how AI systems—particularly facial recognition technologies like RIDs used at ski resorts for access and tracking—encode assumptions that may misidentify individuals from minority groups. 

As someone positioned at the intersection of being both a consumer and potential subject of AI systems—profiting from technology yet vulnerable to its errors—this project explores how real-world constraints (goggles, glare, and lighting on ski slopes) expose fragility in AI perception. It asks: how does AI respond when it encounters bodies and faces that do not conform to its training assumptions, and what does this reveal about fairness, robustness, and ethical deployment?

The core question guiding this work is:

**How does AI perception fail when environmental and social realities challenge its assumptions?**

This inquiry is both technical and ethical, situating model robustness alongside questions of equity, representation, and accountability in AI deployment.


## Dataset Overview

- **Total images:** 60 retrieved from **fairface** & **utkfaces** database 
  - 30 darker skin  
  - 30 lighter skin  

Each subject has:
- One reference image (baseline condition)
- One ski-simulated image (constrained condition)

### Simulated Ski Conditions

The following constraints were applied to approximate real-world skiing conditions for robustness testing:

- **Partial facial occlusion** via a dark rectangle over the eye area  
- **Reduced brightness** to simulate overcast or shadowed conditions  
- **Increased contrast** to mimic glare from snow and reflective surfaces
- **Snow** to mimic snowy condition

No helmet, goggles, or photorealistic environmental objects were added. These transformations were designed solely to stress-test the model’s embedding stability rather than to create realistic images.

---

## Methodology

### Development Environment

This project was initially prototyped in **PyCharm** and executed via **Jupyter Notebooks** to allow step-by-step exploration of data preprocessing, embedding extraction, and analysis. After encountering memory and kernel stability issues on local hardware, the workflow was migrated to **Google Colab** for execution on a higher-memory environment with GPU support.  

Python virtual environments were used to ensure dependency management and reproducibility:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch transformers pillow numpy pandas


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

---

**Rationale:**

- CLIP provides robust vision-language embeddings widely used in research (Radford et al., 2021)

- Normalization ensures cosine similarity is meaningful and comparable across images

- Using pre-trained embeddings avoids overfitting on the small dataset


**Ethical Framing**

The methodology integrates technical rigor with ethical reflection:

- Evaluates robustness across 2 skin tone groups, namely black and white skin colours.

- Highlights potential differential performance in real-world deployment

- Ensures reproducibility via notebooks and controlled environment setup

- Connects results with human alignment principles and Buolamwini’s critique of algorithmic bias

---


### Results

Cosine Similarity Statistics

| Group        | Mean  | Std   | Min   | Max   |
| ------------ | ----- | ----- | ----- | ----- |
| Darker skin  | 0.667 | 0.061 | 0.515 | 0.746 |
| Lighter skin | 0.671 | 0.064 | 0.468 | 0.797 |




