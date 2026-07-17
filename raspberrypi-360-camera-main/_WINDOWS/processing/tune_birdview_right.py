import cv2
import numpy as np

# ================== THÔNG SỐ CALIB ==================
K_front = np.array([
    [343.16492064,   0.        , 369.41221672],
    [  0.        , 406.42212621, 241.37799682],
    [  0.        ,   0.        ,   1.        ]
], dtype=np.float32)

D_front = np.array([
    [-0.19201916],
    [ 0.0211984 ],
    [ 0.00091234],
    [-0.00160403]
], dtype=np.float32)



# ================== LOAD + RESIZE ==================
img_front = cv2.imread(r"/Users/nghiphan/Desktop/front/left data/capture_289.jpg")
img_front = cv2.resize(img_front, (720, 480))

# ================== FISHEYE ==================
nk_front = K_front.copy()
nk_front[0, 0] /= 2
nk_front[1, 1] /= 2

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_front, D_front, np.eye(3), nk_front, (720, 480), cv2.CV_16SC2
)

nemImg_front = cv2.remap(
    img_front, map1, map2,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT
)

# ================== CANVAS 2000x2000 ==================
CANVAS_W = 3000
CANVAS_H = 3000
OFFSET_X = (CANVAS_W - 720) // 2
OFFSET_Y = (CANVAS_H - 480) // 2


# ================== 4 ĐIỂM ==================
points = [
    [178, 171],
    [-231, 329],
    [559, 151],
    [1490, 416],
]

selected = None
point_names = ["TL", "BL", "TR", "BR"]

# ================== CHUỘT KÉO ==================
def mouse(event, x, y, flags, param):
    global points, selected

    real_x = x - OFFSET_X
    real_y = y - OFFSET_Y

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(points):
            if abs(real_x - p[0]) < 100 and abs(real_y - p[1]) < 100:
                selected = i

    elif event == cv2.EVENT_MOUSEMOVE and selected is not None:
        points[selected] = [real_x, real_y]
        print(f"{point_names[selected]} = {points[selected]}")

    elif event == cv2.EVENT_LBUTTONUP:
        print("===== TOẠ ĐỘ MỚI =====")
        for i, p in enumerate(points):
            print(f"{point_names[i]} = {p}")
        print("======================\n")
        selected = None

# ✅ GẮN CALLBACK ĐÚNG WINDOW
cv2.namedWindow("FRONT 2000x2000 CANVAS", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("FRONT 2000x2000 CANVAS", mouse)

# ================== LOOP ==================
while True:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Gắn ảnh vào canvas
    canvas[OFFSET_Y:OFFSET_Y+480, OFFSET_X:OFFSET_X+720] = nemImg_front
    view = canvas.copy()

    # Vẽ điểm + khung
    draw_pts = []
    for p in points:
        draw_p = (int(p[0] + OFFSET_X), int(p[1] + OFFSET_Y))
        draw_pts.append(draw_p)
        cv2.circle(view, draw_p, 10, (0, 0, 255), -1)

    cv2.line(view, draw_pts[0], draw_pts[1], (255, 0, 0), 2)
    cv2.line(view, draw_pts[1], draw_pts[3], (255, 0, 0), 2)
    cv2.line(view, draw_pts[3], draw_pts[2], (255, 0, 0), 2)
    cv2.line(view, draw_pts[2], draw_pts[0], (255, 0, 0), 2)

    # ================== BIRD VIEW 400x160 ==================
    pts1 = np.float32(points)
    pts2 = np.float32([
        [0, 0],
        [160, 0],
        [0, 400],
        [160, 400]
    ])


    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    bird = cv2.warpPerspective(nemImg_front, matrix, (160, 400))
    bird = cv2.flip(bird, 0)   # đảo trên ↔ dưới


    # ================== HIỂN THỊ ==================
    cv2.imshow("FRONT 2000x2000 CANVAS", view)
    cv2.imshow("BIRD 400x160", bird)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("===== SAVE POINTS =====")
        print("points = [")
        for p in points:
            print(f"    {p},")
        print("]\n")

    if key == 27:
        break

cv2.destroyAllWindows()
