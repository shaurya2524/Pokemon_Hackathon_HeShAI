from ultralytics import YOLO
m = YOLO("/home/shaurya/HACKATHON_2/runs/yolov8n_custom4/weights/weights_16_09/best.pt")
print("Model names:", getattr(m, "names", None))
res = m.predict(source="./dataset/images/img_00000.png", conf=0.2, imgsz=640, device='cpu', verbose=True)

print("len(res):", len(res))
r0 = res[0]
print("r0.boxes:", r0.boxes)
try:
    print("xyxy:", r0.boxes.xyxy)
    print("conf:", r0.boxes.conf)
    print("cls:", r0.boxes.cls)
except Exception as e:
    print("Can't access tensors:", e)
