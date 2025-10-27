# Automated Defect Detection (Image Analysis) - Demo Project

**What this is**
- A lightweight demo project (Flask) that implements classical image-processing-based defect detection (edge detection + contour analysis).
- Optional: integrates YOLOv4-Tiny **if** you download the model files (weights + cfg + names) into the `models/` folder (not included here).
- Includes a simple user-login, image upload, preprocessing, defect visualization, and report generation (CSV).

**Features implemented**
- FR1: User authentication (simple, local users.json).
- FR2: Image upload & local storage (`uploads/`).
- FR3: Preprocessing (resize, denoise, histogram equalization).
- FR4: Optional YOLOv4-Tiny integration (code included; weights *not* included).
- FR5/FR6: Processed images saved to `processed/` with detected defects highlighted.
- FR7: Report generation (CSV) saved to `reports/`.

**How to run (VS Code / local machine)**
1. Create and activate a Python virtual environment:
   - Linux / macOS:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) To enable YOLO detection, download these files and put them in `models/`:
   - `yolov4-tiny.cfg`
   - `yolov4-tiny.weights`
   - `coco.names`
   These files can be obtained from the Darknet / YOLOv4 resources (AlexeyAB/darknet repo or other mirrors).
   **Note:** weights are large (~20+ MB for tiny; main weights are much larger). They are *not* included here.

4. Run the app:
   ```
   python app.py
   ```
   Then open `http://127.0.0.1:5000` in your browser.

**Default login**
- username: `admin`
- password: `admin123`

**Where things are**
- `uploads/` : uploaded and sample images
- `processed/` : images with bounding boxes drawn
- `reports/` : CSV reports for each run
- `models/` : (place YOLO files here to enable YOLO)

**Notes & next steps**
- This is a starting point. You can:
  - Swap the classical detector for a trained segmentation model.
  - Add persistent DB for users/reports.
  - Improve UI and add PDF reporting.
