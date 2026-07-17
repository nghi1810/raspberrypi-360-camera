import numpy as np
import cv2
import glob
import os

# Chessboard pattern
CHECKERBOARD = (6, 10)
RESIZE_SHAPE = (720, 480)  # (width, height)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                     cv2.fisheye.CALIB_CHECK_COND +
                     cv2.fisheye.CALIB_FIX_SKEW)

# Prepare object points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# Dataset path
dataset_path = r"/Users/nghiphan/Desktop/front/left dat"
images = glob.glob(os.path.join(dataset_path, "*.jpg"))

if not images:
    print("❌ Không tìm thấy ảnh trong thư mục datasets")
    exit()

img_shape = None
for path in images:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Không thể đọc ảnh: {path}")
        continue

    # Resize ảnh input
    img = cv2.resize(img, RESIZE_SHAPE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = gray.shape[::-1]  # Lưu kích thước ảnh (720x480)

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
    print(f"Đã xử lý: {path}")

if not objpoints:
    print("❌ Không tìm thấy bàn cờ trong bất kỳ ảnh nào!")
    exit()

# Calibration
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(images))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(images))]

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, img_shape, K, D, rvecs, tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print(f"🎯 Calibration RMS error: {rms}")
print("K =", K)
print("D =", D)

# Undistort example
img_path = os.path.join(dataset_path, "1.jpg")
img = cv2.imread(img_path)
if img is None:
    print(f"❌ Không thể đọc ảnh: {img_path}")
    exit()

# Resize ảnh test trước khi undistort
img = cv2.resize(img, RESIZE_SHAPE)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, RESIZE_SHAPE, cv2.CV_16SC2
)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
