import os
import cv2
import numpy as np


# compute the camera calibration matrix to undistort images
def calibrate_camera(image_dir, image_names):
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    r = 6
    c = 9

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    grayscale_img_size = ()
    for fname in image_names:
        img = cv2.imread('camera_cal/' + fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_img_size = gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (c, r), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayscale_img_size, None, None)
    return ret, mtx, dist, rvecs, tvecs


# isolate lane line pixels using HSV thresholds
def isolate_lane_lines(undistorted_image):
    hsv_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HSV)
    white_lane_lines = cv2.inRange(hsv_image, np.array([0, 0, 225]), np.array([180, 255, 255]))
    yellow_lane_lines = cv2.inRange(hsv_image, np.array([20, 100, 100]), np.array([80, 255, 255]))
    lane_lines = cv2.bitwise_or(white_lane_lines, yellow_lane_lines)
    return lane_lines


# find the lane lines x/y locations using 8 image slices/segments
def get_lane_indexes_using_image_segments(warped_lane_lines, nonzero_x, nonzero_y):
    left_lane_indexes = []
    right_lane_indexes = []

    # compute the column level sum of the bottom half of the image to identify the location where lane lines are present
    midpoint = np.int(warped_lane_lines.shape[0] / 2)
    column_histogram = np.sum(warped_lane_lines[midpoint:, :], axis=0)
    left_lane_x = np.argmax(column_histogram[:midpoint])
    right_lane_x = np.argmax(column_histogram[midpoint:]) + midpoint

    global start_left_x
    global start_right_x

    start_left_x = left_lane_x
    start_right_x = right_lane_x

    search_margin = 50
    search_region_update_thresh = 50
    for i in range(8):
        # The region bounds
        bottom_y = warped_lane_lines.shape[0] - np.int(i * warped_lane_lines.shape[0] / 8)
        top_y = warped_lane_lines.shape[0] - np.int((i + 1) * warped_lane_lines.shape[0] / 8)

        left_top_x = left_lane_x - search_margin
        left_bottom_x = left_lane_x + search_margin
        right_top_x = right_lane_x - search_margin
        right_bottom_x = right_lane_x + search_margin

        # The x/y indexes for left lane
        segment_left_lane_indexes = ((nonzero_y >= top_y) & (nonzero_y < bottom_y) &
                                     (nonzero_x >= left_top_x) & (nonzero_x < left_bottom_x)).nonzero()[0]
        # the x/y indexes for right lane
        segment_right_lane_indexes = ((nonzero_y >= top_y) & (nonzero_y < bottom_y) &
                                      (nonzero_x >= right_top_x) & (nonzero_x < right_bottom_x)).nonzero()[0]
        # print(len(segment_left_lane_indexes), len(segment_right_lane_indexes))

        # Update the search region location based on found lanes points
        if len(segment_left_lane_indexes) > search_region_update_thresh:
            left_lane_x = np.int(np.mean(nonzero_x[segment_left_lane_indexes]))
        if len(segment_right_lane_indexes) > search_region_update_thresh:
            right_lane_x = np.int(np.mean(nonzero_x[segment_right_lane_indexes]))

        # Store the non-zero point indexes
        left_lane_indexes.extend(segment_left_lane_indexes)
        right_lane_indexes.extend(segment_right_lane_indexes)

    return left_lane_indexes, right_lane_indexes


# find the lane lines x/y locations using the previous computed fit equation
def get_lane_indexes_using_fit_equation(warped_lane_lines, nonzero_x, nonzero_y):
    search_margin = 50
    left_indexes = ((nonzero_x > (left_lane_fit_coeff[0] * (nonzero_y ** 2) + left_lane_fit_coeff[1] * nonzero_y + left_lane_fit_coeff[2] - search_margin)) &
                         (nonzero_x < (left_lane_fit_coeff[0] * (nonzero_y ** 2) + left_lane_fit_coeff[1] * nonzero_y + left_lane_fit_coeff[2] + search_margin)))
    right_indexes = ((nonzero_x > (right_lane_fit_coeff[0] * (nonzero_y ** 2) + right_lane_fit_coeff[1] * nonzero_y + right_lane_fit_coeff[2] - search_margin)) &
                          (nonzero_x < (right_lane_fit_coeff[0] * (nonzero_y ** 2) + right_lane_fit_coeff[1] * nonzero_y + right_lane_fit_coeff[2] + search_margin)))

    left_lane_indexes = left_indexes.nonzero()[0]
    right_lane_indexes = right_indexes.nonzero()[0]

    return left_lane_indexes, right_lane_indexes


# update the fit equation
def update_fit_equation(nonzero_x, nonzero_y, left_lane_indexes, right_lane_indexes):
    if (len(left_lane_indexes) > 150) and (len(right_lane_indexes) > 150):
        left_lane_pts_x = nonzero_x[left_lane_indexes]
        left_lane_pts_y = nonzero_y[left_lane_indexes]
        right_lane_pts_x = nonzero_x[right_lane_indexes]
        right_lane_pts_y = nonzero_y[right_lane_indexes]

        global left_lane_fit_coeff
        global right_lane_fit_coeff

        # Fit a second order polynomial to each
        left_lane_fit_coeff = np.polyfit(left_lane_pts_y, left_lane_pts_x, 2)
        right_lane_fit_coeff = np.polyfit(right_lane_pts_y, right_lane_pts_x, 2)


# generate the output image
def generate_output_image(input_image, y_values):
    img_size = input_image.shape[:2][::-1]
    left_x_values = left_lane_fit_coeff[0] * y_values ** 2 + left_lane_fit_coeff[1] * y_values + left_lane_fit_coeff[2]
    right_x_values = right_lane_fit_coeff[0] * y_values ** 2 + right_lane_fit_coeff[1] * y_values + right_lane_fit_coeff[2]

    # Calculate the lane curvature
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(y_values * ym_per_pix, left_x_values * xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_values * ym_per_pix, right_x_values * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(y_values) * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(y_values) * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # Calculate the position of the vehicle
    left_x_intercept = left_lane_fit_coeff[0] * img_size[1] ** 2 + left_lane_fit_coeff[1] * img_size[1] + left_lane_fit_coeff[2]
    right_x_intercept = right_lane_fit_coeff[0] * img_size[1] ** 2 + right_lane_fit_coeff[1] * img_size[1] + right_lane_fit_coeff[2]

    left_pts = np.array([np.transpose(np.vstack([left_x_values, y_values]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x_values, y_values])))])
    pts = np.hstack((left_pts, right_pts))

    lane_lines_filled = np.zeros_like(input_image).astype(np.uint8)
    cv2.fillPoly(lane_lines_filled, np.int_([pts]), (0,255, 0))

    # Warp the filled lane line back to the original image
    lane_lines_warped = cv2.warpPerspective(lane_lines_filled, Minv, (img_size[0], img_size[1]))

    # Merge the result
    result = cv2.addWeighted(input_image, 1, lane_lines_warped, 0.5, 0)

    lane_curvature = (left_curverad + right_curverad)/2
    vehicle_offset = (img_size[0] / 2 - ((left_x_intercept + right_x_intercept) / 2)) * xm_per_pix

    cv2.putText(result, 'Curvature: ' + str(np.round(lane_curvature,2)) +'m', (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, 'Car off by ' + str(np.round(vehicle_offset,2)) + 'm', (30, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    return result


# image processing pipeline
def image_processing_pipeline(image):
    # undistort the image
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    # isolate the lane lines
    lane_lines = isolate_lane_lines(undistorted_image)

    # apply perspective transform to get the "birds-eye" view
    img_size = lane_lines.shape[:2][::-1]
    warped_lane_lines = cv2.warpPerspective(lane_lines, M, img_size, flags=cv2.INTER_LINEAR)

    # get the lane lines x/y locations to fit a quadratic polynomial
    nonzero_pts = warped_lane_lines.nonzero()
    nonzero_x = nonzero_pts[1]
    nonzero_y = nonzero_pts[0]

    left_lane_indexes = []
    right_lane_indexes = []

    global frame_count
    global is_first_frame

    # if its not the first frame
    # find the lane lines x/y locations using the previous computed fit equation
    if is_first_frame == False:
        left_lane_indexes, right_lane_indexes = get_lane_indexes_using_fit_equation(warped_lane_lines, nonzero_x, nonzero_y)

    # if its first frame or if not points are found using the previous fit equation
    # find the lane lines x/y locations using 9 image slices/segments
    if (is_first_frame == True) or (len(left_lane_indexes) < 3) or (len(right_lane_indexes) < 3):
        left_lane_indexes, right_lane_indexes = get_lane_indexes_using_image_segments(warped_lane_lines, nonzero_x, nonzero_y)

    frame_count += 1
    if frame_count > 0:
        is_first_frame = False

    # update the fit equation
    update_fit_equation(nonzero_x, nonzero_y, left_lane_indexes, right_lane_indexes)

    y_values = np.linspace(0, warped_lane_lines.shape[0]-1, warped_lane_lines.shape[0])

    # generate the output image
    output = generate_output_image(image, y_values)

    return output

# compute the camera calibration matrix to undistort the images
calibration_image_dir = 'camera_cal/'
calibration_image_names = [f for f in os.listdir('camera_cal') if f.endswith('.jpg')]

ret, mtx, dist, rvecs, tvecs = calibrate_camera(calibration_image_dir, calibration_image_names)
#print (ret, mtx, dist, rvecs, tvecs)

# compute the warp perspective transform matrix to get the birds-eye view
# use a test image with straight lane lines (straight_lines1.jpg) and manually selected points
image = cv2.cvtColor(cv2.imread('test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)
src_pts = np.float32([[270,674],
                         [587,455],
                         [695,455],
                         [1035,674]])
dst_pts = np.float32([[270,674],
                         [270,0],
                         [1035,0],
                         [1035,674]])

# undistort the images using the camera calibration matrix
undistorted = cv2.undistort(image, mtx, dist, None, mtx)

# compute perspective transform matrix to get the "birds-eye" view
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# compute inverse perspective transform matrix to re-project back to the original image
Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
#print (M, Minv)

start_left_x = 0
start_right_x = 0

frame_count = 0
is_first_frame = True

left_lane_fit_coeff = []
right_lane_fit_coeff = []

video_path ='project_video.mp4'
output_video_path = 'project_video_solution.mp4'

# cap = cv2.VideoCapture(video_path)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         out = image_processing_pipeline(frame)
#         cv2.imshow('frame',out)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else: break
# cap.release()
# cv2.destroyAllWindows()

from moviepy.editor import VideoFileClip
clip_source = VideoFileClip(video_path)
vid_clip = clip_source.fl_image(image_processing_pipeline)
vid_clip.write_videofile(output_video_path, audio=False)