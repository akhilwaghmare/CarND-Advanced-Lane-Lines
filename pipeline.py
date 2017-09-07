from helpers import *
import cv2
import matplotlib.pyplot as plt

def undistort_test():
    img = cv2.imread('./camera_cal/calibration1.jpg')
    plt.imshow(bgr2rgb(img))
    plt.figure()
    plt.imshow(undistort(img))
    plt.show()

def pipeline(img, debug=False):
    undistorted_img = undistort(img)
    threshold_img = thresholding(undistorted_img)
    masked_threshold_img = lane_mask(threshold_img)
    M, Minv = perspective_transform_matrices()
    perspective_img = transform(masked_threshold_img, M)

    leftx, lefty, rightx, righty, out_img = get_lane_pixels(perspective_img)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_poly_x, left_coeffs = lane_poly(leftx, lefty, perspective_img)
    right_poly_x, right_coeffs = lane_poly(rightx, righty, perspective_img)

    left_curvature = lane_curvature(left_poly_x, ploty, perspective_img)
    right_curvature = lane_curvature(right_poly_x, ploty, perspective_img)

    vehicle_offset = center_offset(left_poly_x, right_poly_x, perspective_img)

    lane_overlay_img = lane_overlay(left_coeffs, right_coeffs, Minv)
    img_with_overlay = cv2.add(lane_overlay_img, img)

    add_info(img_with_overlay, left_curvature + right_curvature / 2, vehicle_offset)

    if debug:
        plt.imshow(bgr2rgb(img))
        plt.figure()
        plt.imshow(bgr2rgb(undistorted_img))
        plt.figure()
        plt.imshow(threshold_img)
        plt.figure()
        plt.imshow(masked_threshold_img)
        plt.figure()
        plt.imshow(perspective_img)
        plt.figure()
        # plt.imshow(out_img)
        plt.imshow(perspective_img)
        plt.plot(left_poly_x, ploty, color='green')
        plt.plot(right_poly_x, ploty, color='green')
        plt.figure()
        plt.imshow(bgr2rgb(img_with_overlay))
        plt.show()

    return img_with_overlay

def pipeline_test():
    img = cv2.imread('./test_images/test6.jpg')
    pipeline(img, debug=True)
