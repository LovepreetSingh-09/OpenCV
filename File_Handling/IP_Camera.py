import cv2
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("ip_url", help="url of the IP camera to read from")
args = parser.parse_args()

capture = cv2.VideoCapture(args.ip_url)

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

frame_index = 0

if capture.isOpened()is False:
    print("Error opening the camera")

while capture.isOpened() is True:
    ret, frame = capture.read()
    if ret is True:
        processing_start = time.time()
        cv2.imshow('Club Nàutic Port de la Selva (Girona - Spain)', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale Club Nàutic Port de la Selva (Girona - Spain)', gray_frame)
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
        processing_end = time.time()
        processing_time_per_frame = processing_end - processing_start
        print('Frames Per Second (fps) : {:.2f}'.format(1/processing_time_per_frame))
    else:
        break

capture.release()
cv2.destroyAllWindows()