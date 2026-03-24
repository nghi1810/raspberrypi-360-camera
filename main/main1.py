import cv2
import threading
import queue
from multiprocessing import Process, Queue
import numpy as np
import math

VIDEO_SOURCE  = 1
VIDEO_SOURCE1 = 1
VIDEO_SOURCE2 = 1
VIDEO_SOURCE3 = 0


wide_front = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/wide_front.jpg"), (80, 80))
front = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/front.jpg"), (80, 80))
sides = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/sides.jpg"), (80, 80))
back = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/back.jpg"), (80, 80))
wide_back = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/wide_back.jpg"), (80, 80))
parking = cv2.resize(cv2.imread("/Users/nghiphan/Desktop/front/open/icon/parking.png"), (80, 80))

# thêm "around" vào danh sách chế độ
display_mode = ["wide_front", "front", "sides", "back", "wide_back", "parking"]



show_parking_icon = False

def on_mouse(event, x, y, flags, param):
    global show_parking_icon, display_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= x < 80 and 520 <= y < 600:
            display_mode[0] = "wide_front"
            show_parking_icon = False  # ẩn parking khi chọn icon khác
        elif 80 <= x < 160 and 520 <= y < 600:
            display_mode[0] = "front"
            show_parking_icon = False
        elif 160 <= x < 240 and 520 <= y < 600:
            display_mode[0] = "sides"
            show_parking_icon = False
        elif 240 <= x < 320 and 520 <= y < 600:
            display_mode[0] = "back"
            show_parking_icon = False
        elif 320 <= x < 400 and 520 <= y < 600:
            display_mode[0] = "wide_back"
            show_parking_icon = False
        elif 400 <= x < 480 and 520 <= y < 600:
            # toggle 2 lần: lần 1 hiện, lần 2 chuyển chế độ
            if not show_parking_icon:
                show_parking_icon = True   # lần 1: bật hiển thị icon
            else:
                display_mode[0] = "parking"  # lần 2: chuyển chế độ

# =============================
# Thread đọc camera
# =============================
def video_reader(cam_index, thread_queue):
    cap = cv2.VideoCapture(cam_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720,1280,3), dtype=np.uint8)
        if thread_queue.full():
            thread_queue.get()
        thread_queue.put(frame)
    cap.release()

# =============================
# Process xử lý
# =============================
def process_front(frame_front):
    K_front = np.array([[347.46273936,   0.        , 331.40305406],
                        [  0.        , 411.0551961 , 246.72370283],
                        [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

    D_front = np.array([[-0.20629088],
                        [ 0.05675375],
                        [-0.04382338],
                        [ 0.01682283]], dtype=np.float32)
    img_front = cv2.resize(frame_front, (720, 480))

    nk_front = K_front.copy()
    nk_front[0, 0] = K_front[0, 0] / 2
    nk_front[1, 1] = K_front[1, 1] / 2

    # Undistort
    map1_front, map2_front = cv2.fisheye.initUndistortRectifyMap(
        K_front, D_front, np.eye(3), nk_front, (720, 480), cv2.CV_16SC2
    )
    undistorted_front = cv2.remap(
        img_front, map1_front, map2_front, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Perspective transform
    points = [
        [144, 186],
        [-480, 348],
        [522, 173],
        [1615, 366],
    ]
    pts1_front = np.float32(points)
    pts2_front = np.float32([
        [0, 0],
        [0, 160],
        [400, 0],
        [400, 160]
    ])
    matrix_front = cv2.getPerspectiveTransform(pts1_front, pts2_front)
    transformed_frame_front = cv2.warpPerspective(undistorted_front, matrix_front, (400, 160))
    transformed_frame_front = cv2.resize(transformed_frame_front,(400, 160))

    return transformed_frame_front


def process_front2(frame_front):
    K_front = np.array([[347.46273936,   0.        , 331.40305406],
                        [  0.        , 411.0551961 , 246.72370283],
                        [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

    D_front = np.array([[-0.20629088],
                        [ 0.05675375],
                        [-0.04382338],
                        [ 0.01682283]], dtype=np.float32)

    img_front = cv2.resize(frame_front, (720, 480))

    nk_front = K_front.copy()
    nk_front[0, 0] = K_front[0, 0] / 2
    nk_front[1, 1] = K_front[1, 1] / 2

    # Undistort
    map1_front, map2_front = cv2.fisheye.initUndistortRectifyMap(
        K_front, D_front, np.eye(3), nk_front, (720, 480), cv2.CV_16SC2
    )
    undistorted_front = cv2.remap(
        img_front, map1_front, map2_front, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )



    return undistorted_front

def process_right(frame_right):
    K_right = np.array([
        [344.62849104,   0.        , 369.37570941],
        [  0.        , 408.80124794, 243.12268812],
        [  0.        ,   0.        ,   1.        ]
    ], dtype=np.float32)

    D_right = np.array([
        [-0.19154693],
        [ 0.03453566],
        [-0.02116621],
        [ 0.00779806]
    ], dtype=np.float32)

    frame_right = cv2.resize(frame_right, (720, 480))

    nk_right= K_right.copy()
    nk_right[0, 0] = K_right[0, 0] / 2
    nk_right[1, 1] = K_right[1, 1] / 2

    # Undistort
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, np.eye(3), nk_right, (720, 480), cv2.CV_16SC2
    )
    undistorted_right = cv2.remap(
        frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Perspective transform
    points = [
        [200, 161],
        [-209, 340],
        [573, 136],
        [1376, 366],
    ]  
    pts1_right = np.float32(points)
    pts2_right = np.float32([
        [0, 0],
        [0, 160],
        [400, 0],
        [400, 160]
    ])
    matrix_right = cv2.getPerspectiveTransform(pts1_right, pts2_right)
    transformed_frame_right = cv2.warpPerspective(undistorted_right, matrix_right, (340, 160))
    transformed_frame_right = cv2.rotate(transformed_frame_right, cv2.ROTATE_90_CLOCKWISE)

    

    return transformed_frame_right


def process_right2(frame_right):
    K_right = np.array([
        [344.62849104,   0.        , 369.37570941],
        [  0.        , 408.80124794, 243.12268812],
        [  0.        ,   0.        ,   1.        ]
    ], dtype=np.float32)

    D_right = np.array([
        [-0.19154693],
        [ 0.03453566],
        [-0.02116621],
        [ 0.00779806]
    ], dtype=np.float32)

    frame_right = cv2.resize(frame_right, (720, 480))

    nk_right= K_right.copy()
    nk_right[0, 0] = K_right[0, 0] / 2
    nk_right[1, 1] = K_right[1, 1] / 2

    # Undistort
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, np.eye(3), nk_right, (720, 480), cv2.CV_16SC2
    )
    undistorted_right = cv2.remap(
        frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    
    return undistorted_right


def process_left(frame_left):
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

    frame_left = cv2.resize(frame_left, (720, 480))

    nk_left= K_left.copy()
    nk_left[0, 0] = K_left[0, 0] / 2
    nk_left[1, 1] = K_left[1, 1] / 2

    # Undistort
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, np.eye(3), nk_left, (720, 480), cv2.CV_16SC2
    )
    undistorted_left = cv2.remap(
        frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Perspective transform
    points = [
        [200, 161],
        [-209, 340],
        [573, 136],
        [1376, 366],
    ]  
    pts1_left = np.float32(points)
    pts2_left = np.float32([
        [0, 0],
        [0, 160],
        [340, 0],
        [340, 160]
    ])
    matrix_left = cv2.getPerspectiveTransform(pts1_left, pts2_left)
    transformed_frame_left = cv2.warpPerspective(undistorted_left, matrix_left, (340, 160))
    transformed_frame_left = cv2.rotate(transformed_frame_left, cv2.ROTATE_90_CLOCKWISE)
    transformed_frame_left = cv2.rotate(transformed_frame_left, cv2.ROTATE_180)
    

    return transformed_frame_left


def process_left2(frame_left):
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

    frame_left = cv2.resize(frame_left, (720, 480))

    nk_left= K_left.copy()
    nk_left[0, 0] = K_left[0, 0] / 2
    nk_left[1, 1] = K_left[1, 1] / 2

    # Undistort
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, np.eye(3), nk_left, (720, 480), cv2.CV_16SC2
    )
    undistorted_left = cv2.remap(
        frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted_left


def process_back(frame_back):
    K_back = np.array([[347.46273936,   0.        , 331.40305406],
                        [  0.        , 411.0551961 , 246.72370283],
                        [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

    D_back = np.array([[-0.20629088],
                        [ 0.05675375],
                        [-0.04382338],
                        [ 0.01682283]], dtype=np.float32)

    img_back = cv2.resize(frame_back, (720, 480))

    nk_back = K_back.copy()
    nk_back[0, 0] = K_back[0, 0] / 2
    nk_back[1, 1] = K_back[1, 1] / 2

    # Undistort
    map1_back, map2_back = cv2.fisheye.initUndistortRectifyMap(
        K_back, D_back, np.eye(3), nk_back, (720, 480), cv2.CV_16SC2
    )
    undistorted_back = cv2.remap(
        img_back, map1_back, map2_back, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    points = [
        [144, 186],
        [-480, 348],
        [522, 173],
        [1615, 366],
    ]
    pts1_back = np.float32(points)
    pts2_back = np.float32([
        [0, 0],
        [0, 160],
        [400, 0],
        [400, 160]
    ])
    matrix_back = cv2.getPerspectiveTransform(pts1_back, pts2_back)
    transformed_frame_back = cv2.warpPerspective(undistorted_back, matrix_back, (400, 160))
    transformed_frame_back = cv2.rotate(transformed_frame_back, cv2.ROTATE_180)
    transformed_frame_back = cv2.flip(transformed_frame_back, 1)
    transformed_frame_back = cv2.flip(transformed_frame_back, 1)



    return transformed_frame_back


def process_back2(frame_back):
    K_back = np.array([[347.46273936,   0.        , 331.40305406],
                        [  0.        , 411.0551961 , 246.72370283],
                        [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

    D_back = np.array([[-0.20629088],
                        [ 0.05675375],
                        [-0.04382338],
                        [ 0.01682283]], dtype=np.float32)

    img_back = cv2.resize(frame_back, (720, 480))

    nk_back = K_back.copy()
    nk_back[0, 0] = K_back[0, 0] / 2
    nk_back[1, 1] = K_back[1, 1] / 2

    # Undistort
    map1_back, map2_back = cv2.fisheye.initUndistortRectifyMap(
        K_back, D_back, np.eye(3), nk_back, (720, 480), cv2.CV_16SC2
    )
    undistorted_back = cv2.remap(
        img_back, map1_back, map2_back, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


    return undistorted_back

def process_wide_front(frame_front):
    frame_front = cv2.resize(frame_front, (1080, 1080))
    wide_frame_front = cv2.resize(frame_front, (1020, 520))
    return wide_frame_front

def process_wide_back(frame_back):
    frame_back = cv2.resize(frame_back, (1080, 1080))
    wide_frame_back = cv2.resize(frame_back, (1020, 520))
    return wide_frame_back

def draw_around_view():
    W, H = 460, 598
    img = np.ones((H, W, 3), dtype=np.uint8) * 0

    cx, cy = W // 2, H // 2
    rect_w, rect_h = int(80*1.15), int(200*1.15)
    x1, y1 = cx - rect_w // 2, cy - rect_h // 2
    x2, y2 = cx + rect_w // 2, cy + rect_h // 2

    #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ====================
    # Tạo 5 điểm fan trên
    # ====================
    y_list = [200, 184, 184, 184, 200]
    x_start = int(160*1.15)
    x_end = 276
    step = (x_end - x_start) / 4
    x_list = [int(x_start + i * step) for i in range(5)]
    angles_deg = [165, 127.5, 90, 52.5, 15]
    length = 184

    p1, p1_end, p2, p2_end = None, None, None, None

    for idx, ((x, y), angle) in enumerate(zip(zip(x_list, y_list), angles_deg)):
        angle_rad = math.radians(angle)
        end_point = (x + int(math.cos(angle_rad) * length),
                     y - int(math.sin(angle_rad) * length))

        cv2.line(img, (x, y), end_point, (255, 255, 255), 1)

        if idx == 0:
            p1, p1_end = (x, y), end_point
        elif idx == 4:
            p2, p2_end = (x, y), end_point

    def divide_line(p_start, p_end, n=19):
        return [(int(p_start[0] + (p_end[0] - p_start[0]) * i / n),
                 int(p_start[1] + (p_end[1] - p_start[1]) * i / n)) for i in range(n + 1)]

    points1 = divide_line(p1, p1_end, 9)
    points2 = divide_line(p2, p2_end, 9)

    # ====================
    # Vẽ các cung trên
    # ====================
    for i in range(1, len(points1)):
        ptA, ptB = points1[i], points2[i]
        center = ((ptA[0] + ptB[0]) // 2, (ptA[1] + ptB[1]) // 2 + 100)

        radius = int(math.hypot(ptA[0] - center[0], ptA[1] - center[1]))
        angle_start = math.degrees(math.atan2(ptA[1] - center[1], ptA[0] - center[0]))
        angle_end = math.degrees(math.atan2(ptB[1] - center[1], ptB[0] - center[0]))

        cv2.ellipse(img, center, (radius, radius), 0, angle_start, angle_end,
                    (255, 255, 255), 1)

        # 4 điểm chia cung
        n_parts = 4
        coords = []
        for k in range(n_parts):
            ang = math.radians(angle_start + (k + 0.5) * (angle_end - angle_start) / n_parts)
            px = int(center[0] + radius * math.cos(ang))
            py = int(center[1] + radius * math.sin(ang))
            coords.append((px, py))
            cv2.circle(img, (px, py), 1, (0, 0, 255), -1)

        print(f"Cung TRÊN {i}: {coords}")

    # ====================
    # FAN DƯỚI
    # ====================
    y_list_bot = [y2, y2 + 16, y2 + 16, y2 + 16, y2]
    angles_deg_bot = [195, 232.5, 270, 307.5, 345]
    x_list_bot = x_list[:]

    p1b, p1b_end, p2b, p2b_end = None, None, None, None

    for idx, ((x, y), angle) in enumerate(zip(zip(x_list_bot, y_list_bot), angles_deg_bot)):
        angle_rad = math.radians(angle)
        end_point = (x + int(math.cos(angle_rad) * length),
                     y - int(math.sin(angle_rad) * length))

        cv2.line(img, (x, y), end_point, (255, 255, 255), 1)

        if idx == 0:
            p1b, p1b_end = (x, y), end_point
        elif idx == 4:
            p2b, p2b_end = (x, y), end_point

    points1b = divide_line(p1b, p1b_end, 9)
    points2b = divide_line(p2b, p2b_end, 9)

    # ====================
    # **Sửa lỗi: vẽ cung dưới**
    # ====================
    for i in range(1, len(points1b)):
        ptA, ptB = points1b[i], points2b[i]
        center = ((ptA[0] + ptB[0]) // 2, (ptA[1] + ptB[1]) // 2 - 100)

        radius = int(math.hypot(ptA[0] - center[0], ptA[1] - center[1]))
        angle_start = math.degrees(math.atan2(ptA[1] - center[1], ptA[0] - center[0]))
        angle_end = math.degrees(math.atan2(ptB[1] - center[1], ptB[0] - center[0]))

        cv2.ellipse(img, center, (radius, radius), 0, angle_start, angle_end,
                    (255, 255, 255), 1)

        n_parts = 4
        coords = []
        for k in range(n_parts):
            ang = math.radians(angle_start + (k + 0.5) * (angle_end - angle_start) / n_parts)
            px = int(center[0] + radius * math.cos(ang))
            py = int(center[1] + radius * math.sin(ang))
            coords.append((px, py))
            cv2.circle(img, (px, py), 1, (0, 0, 255), -1)

        print(f"Cung DƯỚI {i}: {coords}")

    return img


######### CAR
# Load car only once (before while True)
car = cv2.imread("/Users/nghiphan/Desktop/front/open/car.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(mask)
car_crop = car[y:y+h, x:x+w]
mask_crop = mask[y:y+h, x:x+w]

# Chuẩn bị kích thước mong muốn
car_target_w, car_target_h = 100, 200  # điều chỉnh tuỳ ý
car_resized = cv2.resize(car_crop, (car_target_w, car_target_h), interpolation=cv2.INTER_AREA)
mask_resized = cv2.resize(mask_crop, (car_target_w, car_target_h), interpolation=cv2.INTER_NEAREST)

def process_worker(proc_queue_in, proc_queue_out, func):
    while True:
        frame = proc_queue_in.get()
        result = func(frame)
        proc_queue_out.put(result)

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    # Thread queues
    thread_queue_front  = queue.Queue(maxsize=1)
    thread_queue_right = queue.Queue(maxsize=1)
    thread_queue_left = queue.Queue(maxsize=1)
    thread_queue_back = queue.Queue(maxsize=1)

    # Process queues
    proc_in  = Queue(maxsize=1)
    proc_out = Queue(maxsize=1)

    proc_in1  = Queue(maxsize=1)
    proc_out1 = Queue(maxsize=1)

    proc_in2  = Queue(maxsize=1)
    proc_out2 = Queue(maxsize=1)

    proc_in3  = Queue(maxsize=1)
    proc_out3 = Queue(maxsize=1)

    # Start threads
    threading.Thread(target=video_reader, args=(VIDEO_SOURCE, thread_queue_front), daemon=True).start()
    threading.Thread(target=video_reader, args=(VIDEO_SOURCE1, thread_queue_right), daemon=True).start()
    threading.Thread(target=video_reader, args=(VIDEO_SOURCE2, thread_queue_left), daemon=True).start()
    threading.Thread(target=video_reader, args=(VIDEO_SOURCE3, thread_queue_back), daemon=True).start()

    # Start processes
    p  = Process(target=process_worker, args=(proc_in,  proc_out, process_front))
    p1 = Process(target=process_worker, args=(proc_in1, proc_out1, process_right))
    p2  = Process(target=process_worker, args=(proc_in2,  proc_out2, process_left))
    p3 = Process(target=process_worker, args=(proc_in3, proc_out3, process_back))
    p.start()
    p1.start()
    p2.start()
    p3.start()

    # =========================
    # Main loop hiển thị
    # =========================
    # Main
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_mouse)
    while True:
        if not thread_queue_front.empty() and not thread_queue_right.empty() and not thread_queue_left.empty() and not thread_queue_back.empty():
            frame  = thread_queue_front.get()
            frame1 = thread_queue_right.get()
            frame2 = thread_queue_left.get()
            frame3 = thread_queue_back.get()

            # gửi vào process nếu queue rỗng
            if not proc_in.full():
                proc_in.put(frame)
            if not proc_in1.full():
                proc_in1.put(frame1)
            if not proc_in2.full():
                proc_in2.put(frame2)
            if not proc_in3.full():
                proc_in3.put(frame3)

        # Lấy kết quả xử lý từ process
        if not proc_out.empty() and not proc_out1.empty() and not proc_out2.empty() and not proc_out3.empty():
            transformed_frame_front  = proc_out.get()
            transformed_frame_right  = proc_out1.get()
            transformed_frame_left  = proc_out2.get()
            transformed_frame_back  = proc_out3.get()
            display = np.zeros((600, 1020, 3), dtype=np.uint8)
            if display_mode[0] == "wide_front":
                wide_frame = process_wide_front(frame)
                display[0:520, 0:1020] = wide_frame

            elif display_mode[0] == "front":
                display[0:160, 0:400] = transformed_frame_front
                transformed_frame_back = cv2.resize(transformed_frame_back, (400, 160))
                display[360:520, 0:400] = transformed_frame_back
                #right
                mask = np.zeros(transformed_frame_right.shape[:2], dtype=np.uint8)
                roi_pts = np.array([
                    [0, 70],    # top-left
                    [160, 0],   # top-right
                    [160, 340], # bottom-right
                    [0, 270]    # bottom-left
                ], np.int32)

                cv2.fillPoly(mask, [roi_pts], 255)

                transformed_frame_right = cv2.bitwise_and(transformed_frame_right, transformed_frame_right, mask=mask)

                y_offset = 90
                x_offset = 400-160
                roi = display[y_offset:y_offset+transformed_frame_right.shape[0],
                x_offset:x_offset+transformed_frame_right.shape[1]]
                mask_3ch = cv2.merge([mask, mask, mask])
                roi[:] = np.where(mask_3ch==255, transformed_frame_right, roi)
                #left
                mask_left = np.zeros(transformed_frame_left.shape[:2], dtype=np.uint8)
                roi_pts_left = np.array([
                    [160, 270],  # new top-left
                    [0, 340],    # new top-right
                    [0, 0],      # new bottom-right
                    [160, 70]    # new bottom-left
                ], np.int32)

                cv2.fillPoly(mask_left, [roi_pts_left], 255)
                transformed_frame_left = cv2.bitwise_and(transformed_frame_left, transformed_frame_left, mask=mask_left)

                y_offset_left = 90
                x_offset_left = 0
                roi_left = display[y_offset_left:y_offset_left+transformed_frame_left.shape[0],
                                x_offset_left:x_offset_left+transformed_frame_left.shape[1]]
                mask_3ch_left = cv2.merge([mask_left, mask_left, mask_left])
                roi_left[:] = np.where(mask_3ch_left == 255, transformed_frame_left, roi_left)



                #car
                front_left = process_front2(frame)
                car_h, car_w = car_resized.shape[:2]
                y_offset1 = 160   
                x_offset1 = 150
                front_frame = cv2.resize(front_left, (620, 600))
                display[0:600, 400:1020] = front_frame

                roi = display[y_offset1:y_offset1+car_h, x_offset1:x_offset1+car_w]
                cv2.copyTo(car_resized, mask_resized, roi)




            elif display_mode[0] == "sides":
                display[0:160, 0:400] = transformed_frame_front
                transformed_frame_back = cv2.resize(transformed_frame_back, (400, 160))
                display[360:520, 0:400] = transformed_frame_back
                #right
                mask = np.zeros(transformed_frame_right.shape[:2], dtype=np.uint8)
                roi_pts = np.array([
                    [0, 70],    # top-left
                    [160, 0],   # top-right
                    [160, 340], # bottom-right
                    [0, 270]    # bottom-left
                ], np.int32)

                cv2.fillPoly(mask, [roi_pts], 255)

                transformed_frame_right = cv2.bitwise_and(transformed_frame_right, transformed_frame_right, mask=mask)

                y_offset = 90
                x_offset = 400-160
                roi = display[y_offset:y_offset+transformed_frame_right.shape[0],
                x_offset:x_offset+transformed_frame_right.shape[1]]
                mask_3ch = cv2.merge([mask, mask, mask])
                roi[:] = np.where(mask_3ch==255, transformed_frame_right, roi)
                #left
                mask_left = np.zeros(transformed_frame_left.shape[:2], dtype=np.uint8)
                roi_pts_left = np.array([
                    [160, 270],  # new top-left
                    [0, 340],    # new top-right
                    [0, 0],      # new bottom-right
                    [160, 70]    # new bottom-left
                ], np.int32)

                cv2.fillPoly(mask_left, [roi_pts_left], 255)
                transformed_frame_left = cv2.bitwise_and(transformed_frame_left, transformed_frame_left, mask=mask_left)

                y_offset_left = 90
                x_offset_left = 0
                roi_left = display[y_offset_left:y_offset_left+transformed_frame_left.shape[0],
                                x_offset_left:x_offset_left+transformed_frame_left.shape[1]]
                mask_3ch_left = cv2.merge([mask_left, mask_left, mask_left])
                roi_left[:] = np.where(mask_3ch_left == 255, transformed_frame_left, roi_left)

                #car
                car_h, car_w = car_resized.shape[:2]
                y_offset1 = 160   # thử giá trị này, tuỳ theo layout của bạn
                x_offset1 = 150
                right_side = process_right2(frame1)
                right_side = cv2.resize(right_side, (272, 578))
                left_side = process_left2(frame2)
                left_side = cv2.resize(left_side, (272, 578))

                display[11:589, 710:982] = right_side
                display[11:589, 438:710] = left_side
                roi = display[y_offset1:y_offset1+car_h, x_offset1:x_offset1+car_w]
                cv2.copyTo(car_resized, mask_resized, roi)


    
            elif display_mode[0] == "back":
                display[0:160, 0:400] = transformed_frame_front
                transformed_frame_back = cv2.resize(transformed_frame_back, (400, 160))
                display[360:520, 0:400] = transformed_frame_back
                #right
                mask = np.zeros(transformed_frame_right.shape[:2], dtype=np.uint8)
                roi_pts = np.array([
                    [0, 70],    # top-left
                    [160, 0],   # top-right
                    [160, 340], # bottom-right
                    [0, 270]    # bottom-left
                ], np.int32)

                cv2.fillPoly(mask, [roi_pts], 255)

                transformed_frame_right = cv2.bitwise_and(transformed_frame_right, transformed_frame_right, mask=mask)

                y_offset = 90
                x_offset = 400-160
                roi = display[y_offset:y_offset+transformed_frame_right.shape[0],
                x_offset:x_offset+transformed_frame_right.shape[1]]
                mask_3ch = cv2.merge([mask, mask, mask])
                roi[:] = np.where(mask_3ch==255, transformed_frame_right, roi)
                #left
                mask_left = np.zeros(transformed_frame_left.shape[:2], dtype=np.uint8)
                roi_pts_left = np.array([
                    [160, 270],  # new top-left
                    [0, 340],    # new top-right
                    [0, 0],      # new bottom-right
                    [160, 70]    # new bottom-left
                ], np.int32)

                cv2.fillPoly(mask_left, [roi_pts_left], 255)
                transformed_frame_left = cv2.bitwise_and(transformed_frame_left, transformed_frame_left, mask=mask_left)

                y_offset_left = 90
                x_offset_left = 0
                roi_left = display[y_offset_left:y_offset_left+transformed_frame_left.shape[0],
                                x_offset_left:x_offset_left+transformed_frame_left.shape[1]]
                mask_3ch_left = cv2.merge([mask_left, mask_left, mask_left])
                roi_left[:] = np.where(mask_3ch_left == 255, transformed_frame_left, roi_left)



                #car
                back_left = process_back2(frame3)
                car_h, car_w = car_resized.shape[:2]
                y_offset1 = 160   
                x_offset1 = 150
                back_frame = cv2.resize(back_left, (620, 600))
                display[0:600, 400:1020] = back_frame

                roi = display[y_offset1:y_offset1+car_h, x_offset1:x_offset1+car_w]
                cv2.copyTo(car_resized, mask_resized, roi)

            elif display_mode[0] == "wide_back":
                wide_frame = process_wide_back(frame3)
                display[0:520, 0:1020] = wide_frame



            elif display_mode[0] == "parking":
                around_view = draw_around_view()
                ah, aw = around_view.shape[:2]
                display[1:599, 480:940] = around_view
                display[0:160, 0:400] = transformed_frame_front
                transformed_frame_back = cv2.resize(transformed_frame_back, (400, 160))
                display[360:520, 0:400] = transformed_frame_back
                #right
                mask = np.zeros(transformed_frame_right.shape[:2], dtype=np.uint8)
                roi_pts = np.array([
                    [0, 70],    # top-left
                    [160, 0],   # top-right
                    [160, 340], # bottom-right
                    [0, 270]    # bottom-left
                ], np.int32)

                cv2.fillPoly(mask, [roi_pts], 255)

                transformed_frame_right = cv2.bitwise_and(transformed_frame_right, transformed_frame_right, mask=mask)

                y_offset = 90
                x_offset = 400-160
                roi = display[y_offset:y_offset+transformed_frame_right.shape[0],
                x_offset:x_offset+transformed_frame_right.shape[1]]
                mask_3ch = cv2.merge([mask, mask, mask])
                roi[:] = np.where(mask_3ch==255, transformed_frame_right, roi)
                #left
                mask_left = np.zeros(transformed_frame_left.shape[:2], dtype=np.uint8)
                roi_pts_left = np.array([
                    [160, 270],  # new top-left
                    [0, 340],    # new top-right
                    [0, 0],      # new bottom-right
                    [160, 70]    # new bottom-left
                ], np.int32)

                cv2.fillPoly(mask_left, [roi_pts_left], 255)
                transformed_frame_left = cv2.bitwise_and(transformed_frame_left, transformed_frame_left, mask=mask_left)

                y_offset_left = 90
                x_offset_left = 0
                roi_left = display[y_offset_left:y_offset_left+transformed_frame_left.shape[0],
                                x_offset_left:x_offset_left+transformed_frame_left.shape[1]]
                mask_3ch_left = cv2.merge([mask_left, mask_left, mask_left])
                roi_left[:] = np.where(mask_3ch_left == 255, transformed_frame_left, roi_left)
                #car
                car_h, car_w = car_resized.shape[:2]
                y_offset1 = 160   # thử giá trị này, tuỳ theo layout của bạn
                x_offset1 = 150
                roi = display[y_offset1:y_offset1+car_h, x_offset1:x_offset1+car_w]
                cv2.copyTo(car_resized, mask_resized, roi)



            display[520:600, 0:80] = wide_front
            display[520:600, 80:160] = front
            display[520:600, 160:240] = sides
            display[520:600, 240:320] = back
            display[520:600, 320:400] = wide_back
            if show_parking_icon:
                display[520:600, 400:480] = parking

            cv2.imshow("Camera", display)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    p.terminate()
    p1.terminate()
    p2.terminate()
    p3.terminate()
    cv2.destroyAllWindows()
