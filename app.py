from flask import Flask, render_template, Response
import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
from cascade import *
from pathlib import Path
from data.kaggle.cnn import CNN
from PIL import Image


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = CNN()
checkpoint = Path(__file__).parent/"model"/"bi_noise"/"model-cnn-epoch-17.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
trans = transforms.Compose([
    transforms.Resize((48, 48)),   
    transforms.ToTensor(),
])

# camera = cv2.VideoCapture(0)  # use 0 for web camera
# for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    # while True:
    #     # Capture frame-by-frame
    #     success, frame = camera.read()  # read the camera frame
    #     if not success:
    #         break
    #     else:
    #         frame = detectAndDisplay(frame)
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    cap = cv.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        frame, faceROI = detectAndDisplay(frame)
        if faceROI is not None:
            faceROI = cv2.resize(faceROI, dsize=(48,48), interpolation=cv2.INTER_CUBIC)
            # print(f"({len(faceROI)}, {len(faceROI[0])})") # (48, 48)
            # faceROI = np.array(faceROI).reshape((1, 48, 48))
            
            faceROI = Image.fromarray(faceROI)
            faceROI = trans(faceROI).float()
            faceROI  = faceROI.unsqueeze(1)
            
            outputs = model(faceROI)
            print(outputs)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        # if cv.waitKey(10) == 27:
        #     break

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
