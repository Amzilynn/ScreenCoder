from tensorflow.keras.models import load_model
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image
import cv2
import subprocess
from flask import Flask, request, render_template, redirect, send_from_directory, jsonify
import csv
import pytesseract
from CodeGenerator2 import generate_code_with_GGUF  # Make sure this import works

# ==============================
# Directory Setup
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
uploads_dir = os.path.join(static_dir, 'uploads')
cropped_dir = os.path.join(uploads_dir, 'cropped')
results_dir = os.path.join(uploads_dir, 'results')

# Ensure directories exist
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ==============================
# Flask App
# ==============================
app = Flask(__name__)
app.config['STATIC_FOLDER'] = static_dir

# ==============================
# Load Models
# ==============================
classification_model = load_model("../../models/modelclasssification.h5")
yolo_screenshot = YOLO("../../models/screenshot.pt")
yolo_sketch = YOLO("../../models/sketch.pt")

# ==============================
# Helper Functions
# ==============================
def classify_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = classification_model.predict(img_array)
    return "Sketch" if prediction[0][0] > 0.5 else "Screenshot"

def extract_text_from_bbox(image, bbox):
    left, top = int(bbox['X']), int(bbox['Y'])
    right, bottom = left + int(bbox['Width']), top + int(bbox['Height'])
    cropped_img = image.crop((left, top, right, bottom))
    return pytesseract.image_to_string(cropped_img).strip()

def process_with_yolo(model, img_path, output_dir):
    img = cv2.imread(img_path)
    results = model.predict(img, show=True, save=True, project=output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    text_file_path = os.path.join(output_dir, base_filename + ".txt")
    csv_file_path = os.path.join(output_dir, base_filename + ".csv")

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Element ID", "Class", "X", "Y", "Width", "Height"])

        element_counter = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_label = model.names[int(box.cls)]
                element_counter += 1
                element_id = f"element_{element_counter}"

                width, height = x2 - x1, y2 - y1

                css_style = f"{{top:{y1}px; left:{x1}px; width:{width}px; height:{height}px;}}"

                text_content = ""
                if class_label.lower() in ['text', 'button']:
                    pil_image = Image.open(img_path)
                    bbox = {'X': x1, 'Y': y1, 'Width': width, 'Height': height}
                    text_content = extract_text_from_bbox(pil_image, bbox)

                with open(text_file_path, 'a') as txt_file:
                    txt_file.write(f"{element_id}\nClass: {class_label}\nContent: {text_content}\n{css_style}\n**************\n")

                csv_writer.writerow([element_id, class_label, int(x1), int(y1), int(width), int(height)])

    return text_file_path

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        img_path = os.path.join(uploads_dir, file.filename)
        file.save(img_path)

        predicted_class = classify_image(img_path)
        if predicted_class == "Sketch":
            txt_path = process_with_yolo(yolo_sketch, img_path, results_dir)
        else:
            txt_path = process_with_yolo(yolo_screenshot, img_path, results_dir)

        result_image_path = os.path.join(results_dir, 'predict', 'image0.jpg')
        return render_template("extraction.html",
                               prediction=predicted_class,
                               img_path=os.path.basename(img_path),
                               result_image_path=os.path.basename(result_image_path),
                               txt_filename=os.path.basename(txt_path))

    return render_template("extraction.html", img_path=None)

@app.route("/view_elements")
def view_elements():
    filename = request.args.get('filename')
    if not filename:
        return "No file specified."
    txt_path = os.path.join(static_dir, filename)
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

@app.route("/generate_code")
def generate_code():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "No filename specified."})

    txt_path = os.path.join(static_dir, filename)
    with open(txt_path, 'r', encoding='utf-8') as f:
        UI_description = f.read()

    html_code = generate_code_with_GGUF(UI_description)

    output_html_path = os.path.join(static_dir, 'generated_code.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_code)

    return jsonify({'html_code': html_code})

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
