import cv2

cap = cv2.VideoCapture(1)
img_count = 0

# ================== HÀM VẼ LƯỚI ==================
def draw_grid(img, rows=10, cols=10, color=(0, 255, 0), thickness=1):
    h, w = img.shape[:2]

    # Vẽ các đường ngang
    for i in range(1, rows):
        y = int(i * h / rows)
        cv2.line(img, (0, y), (w, y), color, thickness)

    # Vẽ các đường dọc
    for j in range(1, cols):
        x = int(j * w / cols)
        cv2.line(img, (x, 0), (x, h), color, thickness)

    return img

# ================== LOOP CAMERA ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vẽ lưới lên frame
    frame_grid = draw_grid(frame.copy(), rows=3, cols=3)

    cv2.imshow("Camera + Grid", frame_grid)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"capture_{img_count}.jpg"
        cv2.imwrite(filename, frame)   # Lưu ảnh gốc KHÔNG có lưới
        print(f"Saved: {filename}")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
