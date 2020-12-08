import os
import cv2
import numpy as np
import glob

left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')
ear_cascade35_set1 = cv2.CascadeClassifier('cascade35_set1.xml')
ear_cascade30_set2 = cv2.CascadeClassifier('cascade30_set2.xml')

ear_cascade20_set1 = cv2.CascadeClassifier('cascade20_set1.xml')
ear_cascade20_set2 = cv2.CascadeClassifier('cascade20_set2.xml')
ear_cascade30_set1 = cv2.CascadeClassifier('cascade30_set1.xml')

if left_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if ear_cascade20_set1.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if ear_cascade20_set2.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if ear_cascade30_set1.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if ear_cascade30_set2.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

if ear_cascade35_set1.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')


def calculate_multiple_intersection_over_union(image, imageBB, maskBB):

    iou = []

    for i in imageBB:
        img = [np.array([[i[0], i[1]], [i[0]+i[2], i[1]], [i[0]+i[2], i[1]+i[3]], [i[0], i[1]+i[3]]])]
        for j in maskBB:
            stencil_mask = np.zeros(image.shape).astype(image.dtype)
            cv2.fillPoly(stencil_mask, [j], [255, 255, 255])
            mask_image = cv2.bitwise_and(image, stencil_mask)
            #mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

            stencil_image = np.zeros(image.shape).astype(image.dtype)
            cv2.fillPoly(stencil_image, img, [255, 255, 255])
            detected_image = cv2.bitwise_and(image, stencil_image)
            #detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

            intersection = np.logical_and(mask_image, detected_image)
            union = np.logical_or(mask_image, detected_image)
            iou_score = np.sum(intersection) / np.sum(union)
            iou.append(iou_score)

    return max(iou)


all_iou = []


def detector(image, mask):

    msk = cv2.imread(mask)
    imgray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(msk, contours, -1, (255, 0, 0), 3)
    # cv2.imshow('Ear Detector', msk)
    # cv2.waitKey()

    img = cv2.imread(image)
    img_copy = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, 1.05, 6)  # green
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.05, 6)  # blue
    ear_35_set1 = ear_cascade35_set1.detectMultiScale(gray, 1.3, 5)  # red
    ear_30_set2 = ear_cascade30_set2.detectMultiScale(gray, 1.3, 5)  # purple

    ear_20_set1 = ear_cascade20_set1.detectMultiScale(gray, 1.3, 5)  # cyan - not good
    ear_20_set2 = ear_cascade20_set2.detectMultiScale(gray, 1.3, 5)  # yellow - not good
    ear_30_set1 = ear_cascade30_set1.detectMultiScale(gray, 1.3, 5)  # black

    if type(left_ear) is tuple and type(right_ear) is tuple:
        # go with purple
        if type(ear_30_set2) is tuple:
            # go with black
            if type(ear_30_set1) is tuple:
                # go with red
                if type(ear_35_set1) is tuple:
                    # go with yellow and cyan
                    for (x, y, w, h) in ear_20_set2:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)  # yellow - not good
                    if not (type(ear_20_set2) is tuple):
                        all_iou.append(calculate_multiple_intersection_over_union(img_copy, ear_20_set2, contours))
                    for (x, y, w, h) in ear_20_set1:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)  # cyan - not good
                    if not (type(ear_20_set1) is tuple):
                        all_iou.append(calculate_multiple_intersection_over_union(img_copy, ear_20_set1, contours))
                else:
                    for (x, y, w, h) in ear_35_set1:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # red
                    all_iou.append(calculate_multiple_intersection_over_union(img_copy, ear_35_set1, contours))
            else:
                for (x, y, w, h) in ear_30_set1:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)  # black
                all_iou.append(calculate_multiple_intersection_over_union(img_copy, ear_30_set1, contours))
        else:
            for (x, y, w, h) in ear_30_set2:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)  # purple
            all_iou.append(calculate_multiple_intersection_over_union(img_copy, ear_30_set2, contours))
    else:
        for (x, y, w, h) in left_ear:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # green
        if not (type(left_ear) is tuple):
            all_iou.append(calculate_multiple_intersection_over_union(img_copy, left_ear, contours))

        for (x, y, w, h) in right_ear:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # blue
        if not(type(right_ear) is tuple):
            all_iou.append(calculate_multiple_intersection_over_union(img_copy, right_ear, contours))

    # uncomment to look at the detection live
    cv2.imshow('Ear Detector', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


for image in glob.glob("data/AWEForSegmentation/AWEForSegmentation/test/*.png"):
    mask = image.replace('test', 'testannot_rect')
    detector(image, mask)
    print(image)


print('overall: ' + str(sum(all_iou) / len(all_iou)))

counter = 0
sum = 0.0

for i in all_iou:
    if i > 0:
        sum = sum + i
        counter = counter + 1

on_image_detection = counter / len(all_iou)

print('ear found on image: ' + str(on_image_detection))

avg = sum/counter

print('detected: ' + str(counter))
print('average: ' + str(avg))

