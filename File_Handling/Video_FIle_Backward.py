import argparse
import cv2

def fourcc_decode(fourcc):
    fourcc_int = int(fourcc)
    print('Fourcc : ', fourcc_int)
    decoded_fourcc = ''
    for i in range(4):
        int_fourcc = fourcc_int >> 8 * i & 0xFF
        print('Int Fourcc : ', int_fourcc)
        decoded_fourcc += chr(int_fourcc)
    return decoded_fourcc

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='video_path required')
parser.add_argument('output_video_path', help='Output Video Path Required')

args = parser.parse_args()

capture = cv2.VideoCapture(args.video_path)

fps = capture.get(cv2.CAP_PROP_FPS)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
codec = fourcc_decode(capture.get(cv2.CAP_PROP_FOURCC))

fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)), True)

total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
print('Total_Frames : {}'.format(total_frames))

frame_index = total_frames - 1

if not capture.isOpened():
    print('Error Opening Video File')

for i in range(int(frame_index), -1, -1):
    print("Current Frame Index : {}".format(i))
    capture.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = capture.read()
    print('Ret : ', ret)
    if ret:
        cv2.imshow('Original frame', frame)
        out.write(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale frame', gray_frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

out.release()
capture.release()
cv2.destroyAllWindows()



