
from flask import Flask, render_template, request, send_from_directory, jsonify, flash
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import zipfile
import cv2
import json
import fitz  # PyMuPDF
from collections import defaultdict
from werkzeug.utils import secure_filename
import csv
from io import StringIO
import traceback


def export_json(results, output_path_json, conf_threshold=0.25):
    """
    Exports YOLO detections to a structured JSON file.
    Schema:
    {
        "detections": [
            {
                "class_id": 1,
                "class_name": "signature",
                "confidence": 0.94,
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }
    """
    detections = []

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue
            
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
        cls = int(box.cls)

        detections.append({
            "class_id": cls,
            "class_name": results[0].names[cls],
            "confidence": round(conf, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    data = {"detections": detections}

    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return detections


def export_csv(detections_list, output_path_csv):
    """Export detections to CSV format."""
    if not detections_list:
        return
    
    with open(output_path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Page', 'Class ID', 'Class Name', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        
        for page_num, detections in enumerate(detections_list, 1):
            for det in detections:
                writer.writerow([
                    page_num,
                    det['class_id'],
                    det['class_name'],
                    det['confidence'],
                    det['bbox'][0],
                    det['bbox'][1],
                    det['bbox'][2],
                    det['bbox'][3]
                ])



def draw_pretty_boxes(image, results):
    """
    Pretty bounding boxes with BIG readable text.
    Text size auto-scales based on image resolution.
    """
    img = image.copy()

    # Color per class (signature, stamp, qr)
    class_colors = {
        0: (92, 180, 255),   # signature – light blue
        1: (80, 220, 120),   # stamp – greenish
        2: (255, 150, 60),   # qr – orange
    }

    # Dynamic scaling relative to image diagonal
    diag = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    font_scale = diag / 1500   # adjust if needed
    font_thickness = max(2, int(diag / 2000))

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls)
        conf = float(box.conf)

        color = class_colors.get(cls, (200, 200, 200))
        label = f"{results[0].names[cls].upper()}  {int(conf*100)}%"

        # ----- Draw main bounding box -----
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        # ----- Compute label box -----
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_y = y1 - 12 if y1 - 12 > 0 else y1 + text_h + 12

        # Label background (bigger, cleaner)
        cv2.rectangle(
            img,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 20, label_y + 5),
            color,
            -1,
            cv2.LINE_AA
        )

        # ----- Put text -----
        cv2.putText(
            img,
            label,
            (x1 + 10, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return img


app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'gif', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model with error handling
try:
    model = YOLO("my_model.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file):
    """Validate uploaded file."""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "OK"
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Check if model is loaded
            if model is None:
                return render_template(
                    "index.html",
                    error="Model file not found. Please ensure my_model.pt exists in the project root.",
                    input_image=None,
                    results_with_time=None,
                    json_zip=None,
                    stats=None
                )
            
            # Validate file upload
            if "image" not in request.files:
                return render_template(
                    "index.html",
                    error="No file uploaded. Please select a file.",
                    input_image=None,
                    results_with_time=None,
                    json_zip=None,
                    stats=None
                )

            file = request.files["image"]
            is_valid, error_msg = validate_file(file)
            
            if not is_valid:
                return render_template(
                    "index.html",
                    error=error_msg,
                    input_image=None,
                    results_with_time=None,
                    json_zip=None,
                    stats=None
                )

            # Get confidence threshold from form (default 0.25)
            conf_threshold = float(request.form.get('conf_threshold', 0.25))
            conf_threshold = max(0.0, min(1.0, conf_threshold))  # Clamp between 0 and 1

            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            safe_filename = f"{timestamp}_{filename}"
            input_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            file.save(input_path)
            
            # Detect if PDF or image
            filename_lower = filename.lower()
            images = []
            
            try:
                if filename_lower.endswith(".pdf"):
                    # Convert PDF to images using PyMuPDF
                    doc = fitz.open(input_path)
                    if len(doc) == 0:
                        return render_template(
                            "index.html",
                            error="PDF file is empty or corrupted.",
                            input_image=None,
                            results_with_time=None,
                            json_zip=None,
                            stats=None
                        )
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        images.append((img, page_num + 1))
                    doc.close()
                else:
                    # Regular image
                    img = Image.open(input_path)
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append((img, 1))
            except Exception as e:
                return render_template(
                    "index.html",
                    error=f"Error reading file: {str(e)}",
                    input_image=None,
                    results_with_time=None,
                    json_zip=None,
                    stats=None
                )

            # Process each image/page with YOLO
            output_paths = []
            json_results = []
            csv_results = []
            detection_times = []
            results_with_time = []
            all_detections = defaultdict(int)
            all_detections_list = []  # For CSV export
            
            for img, page_number in images:
                try:
                    img_np = np.array(img)
                    
                    # Time the detection
                    start_time = time.time()
                    results = model(img_np, conf=conf_threshold)  # Apply confidence threshold
                    end_time = time.time()
                    
                    detection_time = round(end_time - start_time, 2)
                    detection_times.append(detection_time)
                    
                    # Get annotated image
                    annotated = results[0].plot()
                    
                    # Count detections per class for this page
                    page_stats = defaultdict(int)
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        class_name = results[0].names[cls]
                        all_detections[class_name] += 1
                        page_stats[class_name] += 1
                    
                    # Save output image
                    base_name = os.path.splitext(safe_filename)[0]
                    output_filename = f"result_page{page_number}_{base_name}.png"
                    output_path = os.path.join(RESULT_FOLDER, output_filename)
                    Image.fromarray(annotated).save(output_path)
                    output_paths.append(output_path)
                    
                    # Save JSON
                    json_filename = f"{base_name}_page{page_number}.json"
                    json_output_path = os.path.join(RESULT_FOLDER, json_filename)
                    detections = export_json(results, json_output_path, conf_threshold)
                    all_detections_list.append(detections)
                    json_results.append(json_filename)
                    
                    # Store results with time and page stats
                    results_with_time.append((
                        output_path,
                        detection_time,
                        {
                            'signatures': page_stats.get('signature', 0),
                            'stamps': page_stats.get('stamp', 0),
                            'qr_codes': page_stats.get('qr', 0)
                        }
                    ))
                except Exception as e:
                    return render_template(
                        "index.html",
                        error=f"Error processing page {page_number}: {str(e)}",
                        input_image=None,
                        results_with_time=None,
                        json_zip=None,
                        stats=None
                    )
            
            # Create ZIP file with all JSON results
            base_name = os.path.splitext(safe_filename)[0]
            zip_filename = f"{base_name}_results.zip"
            zip_path = os.path.join(RESULT_FOLDER, zip_filename)
            
            try:
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for json_file in json_results:
                        zipf.write(os.path.join(RESULT_FOLDER, json_file), arcname=json_file)
                    
                    # Add CSV to ZIP
                    csv_filename = f"{base_name}_results.csv"
                    csv_path = os.path.join(RESULT_FOLDER, csv_filename)
                    export_csv(all_detections_list, csv_path)
                    if os.path.exists(csv_path):
                        zipf.write(csv_path, arcname=csv_filename)
                        csv_results.append(csv_filename)
            except Exception as e:
                print(f"Error creating ZIP: {e}")
            
            # Calculate statistics
            total_detections = sum(all_detections.values())
            avg_time = round(sum(detection_times) / len(detection_times), 2) if detection_times else 0
            
            stats = {
                'total_detections': total_detections,
                'signatures': all_detections.get('signature', 0),
                'stamps': all_detections.get('stamp', 0),
                'qr_codes': all_detections.get('qr', 0),
                'avg_time': avg_time,
                'pages_processed': len(images),
                'conf_threshold': conf_threshold
            }
            
            return render_template(
                "index.html",
                input_image=input_path,
                results_with_time=results_with_time,
                json_zip=zip_filename,
                csv_file=csv_results[0] if csv_results else None,
                stats=stats,
                error=None
            )
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Unexpected error: {error_trace}")
            return render_template(
                "index.html",
                error=f"An unexpected error occurred: {str(e)}",
                input_image=None,
                results_with_time=None,
                json_zip=None,
                stats=None
            )

    return render_template("index.html", input_image=None, results_with_time=None, json_zip=None, stats=None, error=None)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


# >>> REQUIRED FOR FLASK TO RUN <<<
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
