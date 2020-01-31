# import SimpleITK as sitk
# def registration(fixed, moving, moving_mask):
#     '''
#     :param fixed: SimpleITK image, which can get spacing ,origin, direction
#     :param moving: SimpleITK image, which can get spac  ing ,origin, direction
#     :param moving_mask: SimpleITK image, which can get spacing ,origin, direction
#     :return: registered moving image and registered mask image
#     '''
#     R = sitk.ImageRegistrationMethod()
#     R.SetMetricAsCorrelation()
#     R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
#     # R.SetOptimizerAsGradientDescent(learningRate=1.0,
#     #                                 numberOfIterations=300)
#     R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))  # get InitialTransform
#     R.SetInterpolator(sitk.sitkLinear)  # Interpolator params
#     outTx = R.Execute(fixed, moving)  # get transform params
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0)
#     resampler.SetTransform(outTx)
#     out_movingImage = resampler.Execute(moving)
#     moving_mask.SetOrigin(moving.GetOrigin())
#     moving_mask.SetOffset(moving.GetOffset())
#     moving_mask.SetDirection(moving.GetDirection())
#     moving_mask.SetSpacing(moving.GetSpacing())
#     out_mask = resampler.Execute(moving_mask)
#     return out_movingImage, out_mask
# import SimpleITK as sitk
# import numpy as np
# image = sitk.ReadImage('E:/zty/U-net/datasets/1/label/00C1240579_CT1_image00000.DCM')
# image=(image+1024)/7560
# print(np.max(image))
# print(np.min(image))
from __future__ import division
import math
import pprint
import skimage 
import numpy as np
import copy
import imageio 
from imageio import imread as _imread
import SimpleITK as sitk
import cv2
import skimage.transform
import matplotlib.pyplot as plt
import os
import shutil
path = 'E:/paper/new/datasets/trainB/'
file_list=os.listdir(path)
count = 0
for file in file_list:
    count = count+1 
    image = sitk.ReadImage(path+file)
    image = np.squeeze(sitk.GetArrayFromImage(image))
    image = np.float64(image)
    threshold,image_mask = cv2.threshold(image,-700,1,cv2.THRESH_BINARY)
    kernel = np.ones((10,10))
    image_mask = cv2.morphologyEx(image_mask,cv2.MORPH_CLOSE,kernel)
    image_mask = np.array(image_mask,np.uint8)
    cnt,hierarchy = cv2.findContours(image_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    delta = 0
    n=[]
    for i in range(len(cnt)):
        if cv2.arcLength(cnt[i],True) > delta:
            delte = cv2.arcLength(cnt[i],True)
            n = cnt[i]
    try:
        print(n.dtype)
        print(type(n))
        print(count)
    except:
        plt.imshow(image)
        plt.show()
        shutil.move(path+file,'E:/paper/new/baddata/')
        print("The %f image,name:"%count)
        print(file)
    
    
    # x1,y1,w1,h1 = cv2.boundingRect(np.array(n))
    # image = cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
    # plt.subplot(141)
    # plt.imshow(image)
    # image = image[y1:y1+h1,x1:x1+w1]
    # plt.subplot(142)
    # plt.imshow(image)
    # plt.subplot(143)
    # plt.imshow(image_mask)
    # plt.subplot(144)
    # image = image+700
    # image = cv2.threshold(image,0,0,cv2.THRESH_TOZERO)[1]
    # image = cv2.threshold(image,1200,1200,cv2.THRESH_TRUNC)[1]
    # image = cv2.resize(image,(512,512))
    # plt.imshow(image)
    # plt.show()
# import os
# import shutil
# path = 'E:/paper/new'
# patien_list = os.listdir(path+'/datasets')
# for patien in patien_list:
#     file_list = os.listdir(path+'/datasets/'+patien+'/label')
#     for file in file_list:
#         shutil.copy2(path+'/datasets/'+patien+'/label/'+file,'E:/paper/new/trainB')