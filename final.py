import numpy as np
import cv2
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10
dcount = 1
scount = 1
hcount = 1
count = 0

def draw(img1, img2, kp1, kp2):
    img1=cv2.drawKeypoints(img1, kp1, None)
    img2=cv2.drawKeypoints(img2, kp2, None)
    return img1,img2

def siftAlone(img1, img2, kp1, kp2, matches):
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,matchColor = (255,255,255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3

def siftHomo(img1, img2, kp1, kp2, matches):
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found ")
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img4

def newValues(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)

    return kp1, kp2, des1, des2, matches

arr =[['image01a.jpg', 'image01b.jpg'],
      ['image02a.jpg', 'image02b.jpg'],
      ['image03a.jpg', 'image03b.jpg'],
      ['image04a.jpg', 'image04b.jpg'],
      ['image05a.jpg', 'image05b.jpg']]

img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1,des2,k=2)

while True:
    if not img1 is None:
        cv2.imshow('img1',img1)
    cv2.imshow('img2',img2)
    key = cv2.waitKey(1);
    if key == 27:
        break
    elif key == ord('d'):
        if(dcount == 1):
            img1, img2 = draw(img1,img2,kp1,kp2)
            dcount = 0
        else:
            img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)
            dcount = 1
    elif key == ord('s'):
        if(scount == 1):
            cv2.destroyAllWindows()
            img3 = siftAlone(img1, img2, kp1, kp2,matches)
            img2 = img3
            img1 = None
            scount = 0
        else:
            cv2.destroyAllWindows()
            img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)
            scount = 1
    elif key == ord('h'):
        if hcount == 1:
            cv2.destroyAllWindows()
            img4 = siftHomo(img1, img2, kp1, kp2, matches)
            img2 = img4
            img1 = None
            hcount = 0
        else:
            img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)
            hcount = 1
    elif key == ord('n'):
        count += 1
        img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)
        kp1, kp2, des1, des2, matches = newValues(img1, img2);
    elif key == ord('p'):
        count -= 1
        img1 = cv2.imread(arr[count][0],cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(arr[count][1],cv2.IMREAD_GRAYSCALE)
        kp1, kp2, des1, des2, matches = newValues(img1, img2);
