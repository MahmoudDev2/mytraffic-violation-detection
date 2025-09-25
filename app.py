import os
import sys
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Add yolov5_repo to the system path
sys.path.append("yolov5_repo")

# Import the legacy pipeline and the parameter class
from tvdr.utils.params import Parameter
from tvdr.core.pipeline_legacy import TrafficViolationDetectionPipelines

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary folders if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Configure parameters using the legacy Parameter class
        params = Parameter()
        params.video_path = video_path
        params.detect_helmet_violation = 'helmet_violation' in request.form
        params.detect_wrongway_violation = 'wrong_way' in request.form
        params.detect_running_redlight_violation = 'running_red_light' in request.form

        # Initialize pipeline
        pipeline = TrafficViolationDetectionPipelines(params)

        # Process video
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['STATIC_FOLDER'], output_filename)
        # Use 'mp4v' codec for broader compatibility
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = pipeline.update(frame, frame_idx)
            out.write(processed_frame)
            frame_idx += 1

        cap.release()
        out.release()

        return redirect(url_for('results', filename=output_filename))

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)