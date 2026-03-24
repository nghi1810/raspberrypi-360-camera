import cv2
import numpy as np


# ================= FRONT CAMERA =================
K_front = np.array([[347.46273936,   0.        , 331.40305406],
                     [  0.        , 411.0551961 , 246.72370283],
                     [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

D_front = np.array([[-0.20629088],
                     [ 0.05675375],
                     [-0.04382338],
                     [ 0.01682283]], dtype=np.float32)

img_front = cv2.imread("/Users/nghiphan/Desktop/front/datasets_front_new/capture_267.jpg")
img_front = cv2.resize(img_front, (720, 480))

# ===== MASK ELLIPSE FRONT =====
center = (350, 200)
axes = (320, 289)
mask = np.zeros(img_front.shape[:2], dtype=np.uint8)
cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

result = img_front.copy()
result[mask == 0] = (0, 0, 0)

# ===== UNDISTORT FRONT =====
nk_front = K_front.copy()
nk_front[0, 0] /= 2
nk_front[1, 1] /= 2

map1_front, map2_front = cv2.fisheye.initUndistortRectifyMap(
    K_front, D_front, np.eye(3), nk_front, (720, 480), cv2.CV_16SC2
)

nemImg_front = cv2.remap(
    img_front, map1_front, map2_front, interpolation=cv2.INTER_LINEAR
)

# ===== BIRD VIEW FRONT =====
points = [
    [171, 177],
    [-680, 382],
    [467, 181],
    [1319, 411],
]

pts1_front = np.float32(points)
pts2_front = np.float32([[0, 0], [0, 160], [400, 0], [400, 160]])

matrix_front = cv2.getPerspectiveTransform(pts1_front, pts2_front)
front_view = cv2.warpPerspective(nemImg_front, matrix_front, (400, 160))

# ================= CANVAS =================
canvas = np.zeros((600, 1020, 3), dtype=np.uint8)

canvas[0:160, 0:400] = front_view
canvas[:, 400:1020] = cv2.resize(result, (620, 600))

# kẻ vạch
for i in range(5):
    x = i * 80
    cv2.line(canvas, (x, 520), (x, 600), (255, 255, 255), 1)
cv2.line(canvas, (0, 520), (400, 520), (255, 255, 255), 1)





# ================= LEFT CAMERA =================
K_left = np.array([
    [344.62849104,   0.        , 369.37570941],
    [  0.        , 408.80124794, 243.12268812],
    [  0.        ,   0.        ,   1.        ]
], dtype=np.float32)

D_left = np.array([
    [-0.19154693],
    [ 0.03453566],
    [-0.02116621],
    [ 0.00779806]
], dtype=np.float32)
img_left = cv2.imread("/Users/nghiphan/Desktop/front/left data/capture_289.jpg")
img_left = cv2.resize(img_left, (720, 480))

nk_left = K_left.copy()
nk_left[0, 0] /= 2
nk_left[1, 1] /= 2

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, np.eye(3), nk_left, (720, 480), cv2.CV_16SC2
)

undistorted_left = cv2.remap(
    img_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR
)

points = [
    [200, 161],
    [-168, 326],
    [573, 136],
    [1266, 330],
]
# ===== BIRD VIEW LEFT =====
pts1_left = np.float32(points)

pts2_left = np.float32([[0, 0], [0, 160], [340, 0], [340, 160]])

matrix_left = cv2.getPerspectiveTransform(pts1_left, pts2_left)
left_view = cv2.warpPerspective(undistorted_left, matrix_left, (340, 160))

left_view = cv2.rotate(left_view, cv2.ROTATE_90_CLOCKWISE)
left_view = cv2.rotate(left_view, cv2.ROTATE_180)

# ===== MASK LEFT =====
# ===== MASK LEFT (SỬA LẠI ĐÚNG BIẾN) =====
mask_left = np.zeros(left_view.shape[:2], dtype=np.uint8)

roi_pts_left = np.array([
    [160, 270],
    [0, 340],
    [0, 0],
    [160, 70]
], np.int32)

cv2.fillPoly(mask_left, [roi_pts_left], 255)

left_view = cv2.bitwise_and(left_view, left_view, mask=mask_left)

# ===== GHÉP LEFT VÀO CANVAS (SỬA display → canvas) =====
y_offset_left = 90
x_offset_left = 0

roi_left = canvas[
    y_offset_left:y_offset_left + left_view.shape[0],
    x_offset_left:x_offset_left + left_view.shape[1]
]

mask_3ch_left = cv2.merge([mask_left, mask_left, mask_left])
roi_left[:] = np.where(mask_3ch_left == 255, left_view, roi_left)


# ================== LOAD ICON CAR ==================
car = cv2.imread("/Users/nghiphan/Desktop/front/open/car.png", cv2.IMREAD_UNCHANGED)

# Nếu ảnh không có alpha thì tạo mask
if car.shape[2] == 4:
    car_rgb = car[:, :, :3]
    alpha_src = car[:, :, 3]
else:
    car_rgb = car
    gray = cv2.cvtColor(car_rgb, cv2.COLOR_BGR2GRAY)
    _, alpha_src = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# ===== CẮT GỌN ICON XE =====
x, y, w, h = cv2.boundingRect(alpha_src)
car_crop = car_rgb[y:y+h, x:x+w]
mask_crop = alpha_src[y:y+h, x:x+w]

# ===== RESIZE ICON XE =====
car_target_w, car_target_h = 100, 200
car_resized = cv2.resize(car_crop, (car_target_w, car_target_h), interpolation=cv2.INTER_AREA)
mask_resized = cv2.resize(mask_crop, (car_target_w, car_target_h), interpolation=cv2.INTER_NEAREST)

alpha = mask_resized.astype(float) / 255.0

# ================== GHÉP ICON CAR VÀO CANVAS ==================
car_x = 150
car_y = 160


roi_car = canvas[
    car_y:car_y + car_target_h,
    car_x:car_x + car_target_w
]

# ===== BLEND ICON XE =====
for c in range(3):
    roi_car[:, :, c] = (
        alpha * car_resized[:, :, c] +
        (1 - alpha) * roi_car[:, :, c]
    ).astype(np.uint8)

# ================= HIỂN THỊ =================
cv2.imshow("Combined View", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
