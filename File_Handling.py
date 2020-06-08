import cv2
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('first_argument', help='First number is required', type=int)
parser.add_argument('second_argument', help='Second number is required', type=int)
parser.add_argument('img_path', help='img_path Required')


args = parser.parse_args()

img=cv2.imread(args.img_path)
cv2.imshow('loaded_image', img)

# Display image for 10 seconds
cv2.waitKey(10000)

print('Args : ',args)
print('Sum : {}'.format(args.first_argument + args.second_argument))

args_dict = vars(args)

print('Args Dict : ',args_dict)

print('First_argument from dict : ', args_dict['first_argument'])

l = len(sys.argv)
print('sys.argv length : ', l)

for i in range(l):
    print('Argv {} : {} '.format(i, sys.argv[i]))
