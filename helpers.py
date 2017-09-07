import pickle
import cv2
import numpy as np

def undistort(img):
    # Read in calibration coefficients
    dist_pickle = pickle.load(open('calibration.pickle', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted

def thresholding(img, s_thresh=(170, 255), x_thresh=(20, 100), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    # Sobel thresholding
    # Gradient magnitude thresholding
    # Gradient direction thresholding
    # Color thresholding

    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Sobel magnitude
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    # Sobel direction
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # Threshold x gradient
    xbinary = np.zeros_like(scaled_sobel)
    xbinary[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1

    # Threshold gradient magnitude
    magbinary = np.zeros_like(scaled_sobelxy)
    magbinary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1

    # Threshold gradient direction
    dirbinary = np.zeros_like(dir_sobel)
    dirbinary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1

    # Threshold color channel
    sbinary = np.zeros_like(s_channel)
    sbinary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the binary thresholds
    combined_binary = np.zeros_like(xbinary)
    combined_binary[((sbinary == 1) | (xbinary == 1)) & (magbinary == 1) & (dirbinary == 1)] = 1

    return combined_binary

def lane_mask(img):
    vertices = np.array([[(0,img.shape[0]),(480, 450), (800, 450), (img.shape[1],img.shape[0])]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 1)
    masked_image = np.zeros_like(img)
    masked_image[(img == 1) & (mask == 1)] = 1

    return masked_image

def perspective_transform_matrices():
    src = np.float32(
        [[190, 720],
        [600, 450],
        [700, 450],
        [1090, 720]])

    dst = np.float32(
        [[200,720],
         [200,0],
         [1080,0],
         [1080,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def transform(img, M):
    image_shape = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, image_shape, flags=cv2.INTER_LINEAR)

    return warped

def get_lane_pixels(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    windows = 9
    window_height = np.int(img.shape[0]/windows)

    nonzero_img = img.nonzero()
    nonzeroy = np.array(nonzero_img[0])
    nonzerox = np.array(nonzero_img[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    window_margin = 100
    minpix = 20

    left_lane_inds = []
    right_lane_inds = []

    for window in range(windows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = win_y_low + window_height
        win_xleft_low = leftx_current - window_margin
        win_xleft_high = leftx_current + window_margin
        win_xright_low = rightx_current - window_margin
        win_xright_high = rightx_current + window_margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def lane_poly_fit_coeffs(x, y):
    coeffs = np.polyfit(y, x, 2)
    return coeffs

def lane_poly(x, y, img):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    coeffs = lane_poly_fit_coeffs(x, y)
    poly = coeffs[0]*ploty**2 + coeffs[1]*ploty + coeffs[2]
    return poly, coeffs

def eval_poly(coeffs, input):
    return coeffs[0]*input**2 + coeffs[1]*input + coeffs[2]

def draw_poly(img, coeffs, steps=30, color=[255, 0, 0], thickness=10):
    pixel_step = img.shape[0] // steps

    for step in range(steps):
        start = step * pixel_step
        end = start + pixel_step

        p1 = (int(eval_poly(coeffs, start)), start)
        p2 = (int(eval_poly(coeffs, end)), end)

        img = cv2.line(img, p2, p1, color, thickness)

    return img

def fill_lane(img, left_coeffs, right_coeffs, color=[0,255,0]):
    y_start = 0
    y_end = img.shape[0]

    for y in range(y_start, y_end):
        left_edge = int(eval_poly(left_coeffs, y))
        right_edge = int(eval_poly(right_coeffs, y))
        img[y][left_edge:right_edge] = color

    return img

def lane_overlay(left_coeffs, right_coeffs, M):
    blank_canvas = np.zeros((720, 1280))
    canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    canvas = draw_poly(canvas, left_coeffs)
    canvas = draw_poly(canvas, right_coeffs)
    canvas = fill_lane(canvas, left_coeffs, right_coeffs)

    canvas = transform(canvas, M)

    return canvas

def add_info(img, curvature, vehicle_position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %d meters' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    dir_label = "left" if vehicle_position < 0 else "right"
    cv2.putText(img, 'Vehicle is %.2f meters %s of center' % (np.abs(vehicle_position), dir_label), (50, 100), font, 1, (255, 255, 255), 2)

y_m_per_pix = 30/720
x_m_per_pix = 3.7/880

def lane_curvature(lane_x, lane_y, img):
    # Recalculate polyfit coefficients to adjust for unit conversion
    coeffs = np.polyfit(lane_y*y_m_per_pix, lane_x*x_m_per_pix, 2)

    y_eval = img.shape[0]*y_m_per_pix
    rad_of_curvature = ((1 + (2*coeffs[0]*y_eval + coeffs[1])**2)**1.5) / np.absolute(2*coeffs[0])
    return rad_of_curvature

def center_offset(left_lane_x, right_lane_x, img):
    lane_center = (left_lane_x[-1] + right_lane_x[-1]) / 2
    car_center = img.shape[1]/2
    return (car_center - lane_center) * x_m_per_pix

def bgr2rgb(img):
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img
