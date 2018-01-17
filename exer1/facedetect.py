#!/usr/bin/env python
# Q1: within ipython do
#     run facedetect.py group_photo_large.jpg
#     run facedetect.py group_photo_small.jpg
#     how are the results different?

# Q2: what happens if you adjust the value of the scaleFactor parameter passed to faceCascade.detectMultiScale?
# Q3: modify find_faces to check if the image is already grey-scale
# Q4: after you run find_faces, use cv2.imshow() to view the contents of 'image'.... did the function add green squares? Modify find_faces so that it does not change the original image.
# Q5: What are the current dimensions of the image? Which dimension is the height and which dimension is the width? Add a line to __main__ that prints the image dimensions.
# Q6: modify find_faces so that the image shown doesn't have a dimension larger than 800 pixels. Do you want to resize the image before or after the face detection routine runs?


import cv2
import os
import sys


def find_faces(image, faceCascade, output_path, resize=False):
    # the haar cascade algorithm requires greyscale images as input
    # convert image to greyscale
    if image.shape[2] >= 1:
        # make greyscale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize so never bigger than 800 px
    if resize:
        x_size = image.shape[1]
        y_size = image.shape[0]
        if x_size > y_size:
            new_x = 800
            ratio = new_x / x_size
            new_y = int(y_size * ratio)
        else:
            new_y = 800
            ratio = new_y / y_size
            new_x = int(x_size * ratio)

        print('old size', x_size, y_size, 'new size', new_x, new_y, 'ratio', ratio)

        resized = cv2.resize(image, (new_x, new_y))
    else:
        resized = image

    cv2.imwrite(output_path, resized)


def getCasc():
    # Get user supplied values
    cascPath = 'haarcascade_frontalface_default.xml'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    return faceCascade


# run facedetect.py group_photo_small.jpg
# commenting out the the 'main' statement allows you to work with the functions in ipython
if __name__ == "__main__":
    imagePath = sys.argv[1]

    image = cv2.imread(imagePath)
    print('Image dimensions:', image.shape[1], 'x', image.shape[0])
    faceCascade = getCasc()
    output_path = '.out'.join(os.path.splitext(imagePath))

    find_faces(image, faceCascade, output_path, resize=False)

    # max_x, max_y = 800, 800
    # yscale = min([max_y/image.shape[0], 1.0])
    # xscale = min([max_x/image.shape[1], 1.0])
    # scale = min([xscale, yscale])
    # image2 = cv2.resize(image, (0,0), fx=scale, fy=scale)
