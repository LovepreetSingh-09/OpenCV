import cv2
from flask import Flask, request, make_response
import numpy as np
import urllib.request

app = Flask(__name__)

@app.route('/canny')
def cannying():
    with urllib.request.urlopen(request.args.get('url')) as img:
        print('Recieved Image :',img)
        print('byte array Image :',bytearray(img))
        image=np.asarray(bytearray(img), dtype=np.uint8)
    cv_img = cv2.imdecode(image, -1)
    gray=cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray, 100, 200)
    ret, buffer = cv2.imencode('.jpeg', edges)
    response = make_response(buffer.tobytes())
    response.headers['Content-type']= 'image/jpeg'
    return response

if __name__ == "__main__":
    app.run()

