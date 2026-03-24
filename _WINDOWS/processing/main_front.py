import cv2
import numpy as np
import os

K = np.array([[347.46273936,   0.        , 331.40305406],
              [  0.        , 411.0551961 , 246.72370283],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

D = np.array([[-0.20629088],
              [ 0.05675375],
              [-0.04382338],
              [ 0.01682283]], dtype=np.float32)



# Đọc ảnh
img = cv2.imread(r"/Users/nghiphan/Desktop/front/datasets_front_new/capture_2.jpg")

# Resize ảnh gốc về 720x480
img = cv2.resize(img, (720, 480))

# ===== XOAY ẢNH 3 ĐỘ =====
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

# Ma trận xoay 3 độ
M = cv2.getRotationMatrix2D(center, 0, 1.0)

# Thực hiện xoay
img = cv2.warpAffine(img, M, (w, h))

# Tạo ma trận camera mới (giảm tiêu cự xuống 1/2)
nk = K.copy()
nk[0, 0] = K[0, 0] / 2
nk[1, 1] = K[1, 1] / 2

# Khởi tạo bản đồ undistort
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), nk, (720, 480), cv2.CV_16SC2
)

# Áp dụng remap để tạo ảnh undistorted
nemImg = cv2.remap(
    img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)
nemImg = cv2.resize(nemImg, (720, 480))

# Các điểm
tl = (175, 145)
bl = (-624, 250)
tr = (490, 153)
br = (1232, 294)
# ✅ Saved points (REAL): [[175, 145], [-624, 250], [490, 153], [1232, 294]]
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
    [400, 0],
    [400, 160]
])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
transformed_frame = cv2.warpPerspective(nemImg, matrix, (400, 160))


# === TẠO CANVAS 1020x600, GHÉP 2 ẢNH ===
canvas = np.zeros((600, 1020, 3), dtype=np.uint8)

# Đặt transformed_frame vào (0,0)
canvas[0:160, 0:400] = transformed_frame

# Resize nemImg để vừa phần còn lại (1020x600)
img_resized = cv2.resize(img, (620, 600))
# Vẽ điểm
for pt in [tl, bl, tr, br]:
    cv2.circle(img_resized, pt, 2, (0, 0, 255), -1)
canvas[0:600, 400:1020] = img_resized

# Hiển thị
cv2.imshow("Combined View", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
