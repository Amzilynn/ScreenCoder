# UI-to-Code: Automatic UI Interface Code Generation from Screenshots and Sketches


---

## Project Overview

**UI-to-Code** is an innovative project that leverages computer vision and multimodal AI to automatically generate HTML/CSS code from website screenshots or hand-drawn UI sketches.

The core idea is to bridge the gap between visual UI design and actual code implementation by:

- Detecting common UI elements (buttons, input fields, text, images, icons, etc.) in an input image
- Extracting their positions, types, and relationships
- Generating clean, semantic, and responsive HTML/CSS code based on the detected structure

### Approaches

This project explores two complementary approaches:

1. **YOLOv8-based Object Detection (Primary backend approach)**  
   Fast and accurate detection of UI components using a custom-trained YOLOv8 model.

2. **Multimodal LLM Approach with Ollama/LLaVA**  
   Using vision-language models (LLaVA) to generate detailed textual descriptions of UI elements from images.


---

## Key Features 

- Custom dataset of ~1,900 manually annotated UI screenshots (train/validation/test splits)
- YOLOv8 model trained to detect **9 UI element classes**:
  - Button
  - Checkbox
  - Image
  - Input-Field
  - Radio-Button
  - Search-Box
  - Select-Dropdown
  - Text
  - Icon

- Scripts for:
  - Dataset exploration and visualization (class distribution, bounding boxes, statistics)
  - Model training and evaluation
  - Inference on new images with bounding box extraction and formatted output (positions, class, confidence)
  - Experimental CSS-style generation from bounding boxes

- Well-structured prompts for guiding LLMs to generate HTML/CSS from descriptions or detections

---

## Project Structure
UI_interface_code_generation/
├── Ollama/ # Multimodal LLM approach
│ └── *.py # Scripts for image description using LLaVA/Ollama
│
├── YOLOV8_Approach/
│ └── BackEnd/
│ ├── data/ # (Local) Annotated datasets and data.yaml
│ ├── runs/ # Training runs and weights (best.pt, etc.)
│ ├── src/ # Source code, weights, scripts
│ ├── results/ # Inference outputs, formatted data
│ └── Various .ipynb/.py # Training, detection, visualization notebooks/scripts
│
├── .gitignore
└── README.md


---
# Dataset Details

- **Manually Annotated Dataset V2/V3 (~1,955 images total)**
  - Train: 1,506 images
  - Validation: 228 images
  - Test: 221 images
- Annotations in YOLO format (bounding boxes + class IDs)
- Classes are imbalanced (Text and Input-Field dominant, Search-Box rare)
- Preprocessing: Grayscale conversion + resizing experimented for better sketch handling

---

## How to Run (Local Setup)

```bash
# Clone the repository
git clone https://github.com/Amzilynn/UI_interface_code_generation.git

# Install dependencies
pip install ultralytics opencv-python matplotlib pandas ollama transformers torch pillow
