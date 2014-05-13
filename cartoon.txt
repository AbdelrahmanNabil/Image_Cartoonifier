import cv2
import numpy as np
from matplotlib import pyplot as plt
#................readfile..................
testFile = open('input.txt','r')
testSize = int(testFile.readline())
for  x in range (0,testSize):
	imageName = testFile.readline()
	imageName =str(imageName.split()[0])
	#...............indexofdot..............
	dotIndex=imageName.index('.')
	#...............readimage.................
	img = cv2.imread(imageName)
	#.................grayscale.................
	gryimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
	cv2.imwrite(imageName[:dotIndex] + "Gray" + imageName[dotIndex:],gryimg)
	#......................medianfilter..........................
	median = cv2.medianBlur(gryimg, 7)
	cv2.imwrite(imageName[:dotIndex] + "Median" + imageName[dotIndex:],median)
	#.........................laplacian edge detection..........
	laplacian=cv2.Laplacian(median,cv2.CV_8U,ksize=5)
	#cv2.imwrite(imageName[:dotIndex] + "" + imageName[dotIndex:],laplacian)
	#.............threshold binary image...............................
	thresh= cv2.threshold(laplacian,125,255,cv2.THRESH_BINARY_INV)[1]
	cv2.imwrite(imageName[:dotIndex] + "Binary" + imageName[dotIndex:],thresh)
	#........................bilateralFilter......................
	bilateral=cv2.bilateralFilter(img,9,9,7)#1
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#2
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#3
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#4
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#5
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#6
	bilateral=cv2.bilateralFilter(bilateral,9,9,7)#7
	cv2.imwrite(imageName[:dotIndex] + "Bilateral" + imageName[dotIndex:],bilateral)
	#...........................final result...................................
	isedge=thresh==0
	finalimg=bilateral
	finalimg[isedge]=0
	cv2.imwrite(imageName[:dotIndex] + "Final" + imageName[dotIndex:],finalimg)