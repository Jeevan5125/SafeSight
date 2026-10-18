from flask import Flask, render_template, Response, jsonify, request, session, send_file

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
from pathlib import Path
import warnings
import csv
import io


# Load the locally trained YOLO11 PPE model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
from safety_monitor import get_monitor, list_events

# Hide the known PyTorch checkpoint-loading FutureWarning.
warnings.filterwarnings('ignore', category=FutureWarning, module=r'ultralytics\.nn\.tasks')
app = Flask(__name__)

app.config['SECRET_KEY'] = 'vinayak'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv'}


#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')
# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/safety-monitor')
def safety_monitor_page():
    return render_template('safety_monitor.html')

@app.route('/dashboard')
def safety_dashboard():
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def safety_dashboard_data():
    monitor = get_monitor()
    return jsonify({
        'snapshot': monitor.snapshot(),
        'events': list_events(limit=40),
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def site_settings():
    monitor = get_monitor()
    if request.method == 'GET':
        return jsonify(monitor.settings())
    try:
        return jsonify(monitor.update_settings(request.get_json(silent=True) or {}))
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/monitor/video')
def safety_monitor_video():
    try:
        return Response(
            get_monitor().stream(0),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except (FileNotFoundError, RuntimeError) as exc:
        return str(exc), 503

@app.route('/api/zones', methods=['GET', 'POST'])
def safety_zones():
    monitor = get_monitor()
    if request.method == 'GET':
        return jsonify(monitor.zones())
    try:
        zone = monitor.add_zone(request.get_json(silent=True) or {})
        return jsonify(zone), 201
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({'error': str(exc)}), 400

@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def delete_safety_zone(zone_id):
    get_monitor().remove_zone(zone_id)
    return jsonify({'ok': True})

@app.route('/api/events')
def safety_events():
    return jsonify(list_events())

@app.route('/api/events.csv')
def safety_events_csv():
    rows = list_events(limit=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Zone', 'Track ID', 'Event', 'PPE status', 'Confidence', 'Evidence file'])
    for event in rows:
        writer.writerow([
            event['occurred_at'], event['zone_name'], event['track_id'],
            event['violation'], event.get('ppe_status', ''),
            f"{event['confidence']:.3f}", event['evidence_file'],
        ])
    buffer = io.BytesIO(output.getvalue().encode('utf-8-sig'))
    return send_file(buffer, mimetype='text/csv', as_attachment=True, download_name='safety_events.csv')

@app.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    error_message = None
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        filename = secure_filename(file.filename)
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error_message = 'Choose a JPG, PNG, MP4, AVI, MOV, or MKV file.'
        else:
            video_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            session['video_path'] = video_path
            session['is_image'] = Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            session['output_filename'] = f'{Path(filename).stem}_output.jpg'
    return render_template('videoprojectnew.html', form=form, error_message=error_message)
@app.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    video_path = session.get('video_path')
    if not video_path:
        return 'Choose an image or video and click Run to generate output.', 400
    return Response(generate_frames(path_x=video_path),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

