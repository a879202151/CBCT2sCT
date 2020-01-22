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
import SimpleITK as sitk
import numpy as np
image = sitk.ReadImage('E:/zty/U-net/datasets/1/label/00C1240579_CT1_image00000.DCM')
image=(image+1024)/7560
print(np.max(image))
print(np.min(image))