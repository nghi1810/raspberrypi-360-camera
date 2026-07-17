import cv2
import numpy as np

K = np.array([[307.4471188,    0.0,         364.80351666],
              [0.0,           274.96429101, 238.23251995],
              [0.0,             0.0,           1.0]], dtype=np.float32)

D = np.array([[0.06536886],
              [-0.24891032],
              [0.13488489],
              [-0.0074151]], dtype=np.float32)

# Đọc ảnh
img = cv2.imread(r"/Users/nghiphan/Desktop/front/datasets_front_new/capture_2.jpg")

# Resize ảnh gốc về 720x480
img = cv2.resize(img, (720, 480))

# Tạo ma trận camera mới (giảm tiêu cự xuống 1/2)
nk = K.copy()
nk[0, 0] = K[0, 0] / 2
nk[1, 1] = K[1, 1] / 2

# Khởi tạo bản đồ undistort với kích thước 800x600
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), nk, (720, 480), cv2.CV_16SC2
)

# Áp dụng remap để tạo ảnh undistorted
nemImg = cv2.remap(
    img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)
nemImg = cv2.resize(nemImg, (720, 480))


(h, w) = nemImg.shape[:2]
center = (w // 2, h // 2)
angle = -3.2  # Xoay phải 5 độ
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
nemImg = cv2.warpAffine(nemImg, M, (w, h))



# Các điểm
tl = (150, 150)
bl = (-360, 430)
tr = (700, 90)
br = (1000, 250)

# Vẽ điểm
for pt in [tl, bl, tr, br]:
    cv2.circle(nemImg, pt, 2, (0, 0, 255), -1)

# Vẽ các đường nối
cv2.line(nemImg, tl, bl, (0, 255, 0), 1)
cv2.line(nemImg, tl, tr, (0, 255, 0), 1)
cv2.line(nemImg, tr, br, (0, 255, 0), 1)
cv2.line(nemImg, bl, br, (0, 255, 0), 1)

# Warp để xem bird's-eye
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([
    [0, 0],
    [0, 160],
    [340, 0],
    [340, 160]
])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
transformed_frame = cv2.warpPerspective(nemImg, matrix, (340, 160))

connect_point1 = (70, 0)
connect_point2 = (70, 160)
cv2.circle(transformed_frame, connect_point1, 1, (0, 0, 255), -1)
cv2.circle(transformed_frame, connect_point2, 1, (0, 0, 255), -1)
#cv2.line(transformed_frame, connect_point1, connect_point2, (0, 255, 0), 1)
# Hiển thị
cv2.imshow("Original Image with Lines", nemImg)
cv2.imshow("Bird's Eye View", transformed_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
