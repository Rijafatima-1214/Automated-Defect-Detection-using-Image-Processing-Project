import os, json, uuid, datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from processing import detect_defects, allowed_file

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-random-key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
for d in [UPLOAD_FOLDER, PROCESSED_FOLDER, REPORTS_FOLDER, os.path.join(BASE_DIR,'models')]:
    os.makedirs(d, exist_ok=True)

USERS_FILE = os.path.join(BASE_DIR, 'users.json')
if not os.path.exists(USERS_FILE):
    users = {'admin': generate_password_hash('admin123')}
    with open(USERS_FILE,'w') as f:
        json.dump(users,f)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        with open(USERS_FILE,'r') as f:
            users = json.load(f)
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials','danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method=='POST':
        files = request.files.getlist('images')
        if not files:
            flash('No files uploaded','warning')
            return redirect(request.url)
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                fname = secure_filename(file.filename)
                uid = str(uuid.uuid4())[:8] + '_' + fname
                save_path = os.path.join(UPLOAD_FOLDER, uid)
                file.save(save_path)
                detection = detect_defects(save_path)
                # normalize returned info for template
                detection['input_filename'] = os.path.basename(detection['input_path'])
                detection['processed_filename'] = os.path.basename(detection['processed_path'])
                results.append(detection)
        # create report CSV
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_csv = os.path.join(REPORTS_FOLDER, f'report_{timestamp}.csv')
        rows=[]
        for r in results:
            if r.get('defects'):
                for d in r['defects']:
                    rows.append({
                        'image': r['input_filename'],
                        'defect_type': d.get('type','unknown'),
                        'confidence': d.get('score', ''),
                        'bbox': d.get('bbox')
                    })
            else:
                rows.append({'image': r['input_filename'], 'defect_type': 'none', 'confidence': '', 'bbox': ''})
        import pandas as pd
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['image','defect_type','confidence','bbox'])
        df.to_csv(report_csv, index=False)
        return render_template('results.html', results=results, report_file=os.path.basename(report_csv))
    # GET -> show upload form and sample images (images placed in uploads/)
    sample_images = []
    for fn in os.listdir(UPLOAD_FOLDER):
        if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
            sample_images.append(fn)
    return render_template('upload.html', sample_images=sample_images)

@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/reports/<path:filename>')
def report_file(filename):
    return send_from_directory(REPORTS_FOLDER, filename)

if __name__=='__main__':
    app.run(debug=True)
