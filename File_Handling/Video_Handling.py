import cv2
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("video_path", help="path of the video file to read from")
args = parser.parse_args()

capture = cv2.VideoCapture(args.video_path)

def decode_fourcc(fourcc):
    fourcc_int = int(fourcc)
    print("int value of fourcc: '{}'".format(fourcc_int))
    fourcc_decode =''
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        print("int_value: '{}'".format(int_value))
        fourcc_decode += chr(int_value)
    return fourcc_decode

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
print("RAW  CAP_PROP_FOURCC  : '{}'".format(capture.get(cv2.CAP_PROP_FOURCC)))
print("CAP_PROP_FOURCC  : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
print("CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
print("CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
print("CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))

frame_index = 0

if capture.isOpened()is False:
    print("Error opening the video file")

while capture.isOpened() is True:
    ret, frame = capture.read()
    if ret is True:
        print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
        print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        cv2.imshow('Input frame from the camera', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale input camera', gray_frame)
        if cv2.waitKey(20) & 0xFF ==ord('q'):
            break
        elif cv2.waitKey(20) & 0xFF ==ord('c'):
            frame_name = 'Camera_Frame_{}.png'.format(frame_index)
            gray_frame_name = 'Gray_Camera_Frame_{}.png'.format(frame_index)
            cv2.imwrite(frame_name, frame)
            print('Saving Frame Image')
            cv2.imwrite(gray_frame_name, gray_frame)
            print('Saving Gray Frame Image')
            frame_index +=1

    else:
        break

capture.release()
cv2.destroyAllWindows()