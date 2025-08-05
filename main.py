MÃ NGUỒN HỆ THỐNG
import cv2
import time
import RPi.GPIO as GPIO
import threading
import numpy as np
from tflite_runtime.interpreter import Interpreter
from RPLCD.i2c import CharLCD
# Cấu hình màn hình LCD I2C
lcd = CharLCD('PCF8574', 0x27)  # Địa chỉ I2C thường là 0x27
# Cấu hình GPIO cho các nút bấm và LED
BUTTON_PINS = [17, 22, 5, 6, 16]  # Chân GPIO cho các nút bấm
LED_PINS = [18, 23, 24, 25]         # Chân GPIO cho các đèn LED
GPIO.setmode(GPIO.BCM)
for led_pin in LED_PINS:
GPIO.setup(led_pin, GPIO.OUT)
GPIO.output(led_pin, GPIO.LOW)  # Khởi động tắt các LED
for button_pin in BUTTON_PINS:
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
led_states = [False] * len(LED_PINS)
mode = "detection"  # Chế độ khởi động là nhận diện
# Đường dẫn tới file model và label
model_path = '/home/tkien/Desktop/custom_model_lite/detect.tflite'
label_path = '/home/tkien/Desktop/custom_model_lite/labelmap.txt'
# Load mô hình TFLite
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
with open(label_path, 'r') as f:
classlabels = f.read().strip().split('\n')
# Hàm tắt tất cả các LED
def clear_all():
for led_pin in LED_PINS:
GPIO.output(led_pin, GPIO.LOW)
# Định nghĩa hàm xử lý khung hình
def process_frame(frame):
global led_states
locations = [0, 0, 0, 0]  # Bốn vị trí: [trái trên, phải trên, trái dưới, phải dưới]
# Chuẩn bị dữ liệu đầu vào
input_data = cv2.resize(frame, (width, height))
input_data = np.expand_dims(input_data, axis=0)
input_data = input_data.astype(np.float32)
input_data = (input_data - 127.5) / 127.5
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
boxes = interpreter.get_tensor(output_details[1]['index'])[0]
classes = interpreter.get_tensor(output_details[3]['index'])[0]
scores = interpreter.get_tensor(output_details[0]['index'])[0]
num_detections = len(scores) if isinstance(scores, np.ndarray) else scores.shape[0]
for i in range(num_detections):
if scores[i] > 0.65:
class_id = int(classes[i])
if class_id == 0:  # Chỉ xử lý người
box = boxes[i]
ymin, xmin, ymax, xmax = box
xmin = int(xmin * frame.shape[1])
xmax = int(xmax * frame.shape[1])
ymin = int(ymin * frame.shape[0])
ymax = int(ymax * frame.shape[0])
center_x = (xmin + xmax) // 2
center_y = (ymin + ymax) // 2  # Tính center_y
# Vẽ bounding box
cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
# Hiển thị score
cv2.putText(frame, f'Score: {scores[i]:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# Cập nhật vị trí LED dựa trên vị trí người
if (center_y > frame.shape[0] // 2) and (center_x > frame.shape[1] // 2):
locations[1] += 1  # Bên phải dưới
elif (center_y > frame.shape[0] // 2) and (center_x < frame.shape[1] // 2):
locations[0] += 1  # Bên trái dưới
elif (center_y < frame.shape[0] // 2) and (center_x > frame.shape[1] // 2):
locations[3] += 1  # Bên phải trên
elif (center_y < frame.shape[0] // 2) and (center_x < frame.shape[1] // 2):
locations[2] += 1  # Bên trái trên
# Cập nhật trạng thái LED
for i in range(len(LED_PINS)):
if locations[i] > 0:
GPIO.output(LED_PINS[i], GPIO.HIGH)
led_states[i] = True
else:
GPIO.output(LED_PINS[i], GPIO.LOW)
led_states[i] = False
# Hiển thị khung hình
cv2.imshow('Object Detection', frame)
# Hàm xử lý nút bấm vật lý
def button_handler():
global led_states, mode
while True:
for i in range(len(BUTTON_PINS)):
button_state = GPIO.input(BUTTON_PINS[i])
if button_state == GPIO.LOW:  # Nút được nhấn
if i == 4:  # Nút ở chân GPIO 16 để chuyển đổi chế độ
if mode == "detection":
mode = "manual"
lcd.clear()
lcd.write_string("Mode: Manual")  # Cập nhật lên LCD
cv2.destroyWindow('Object Detection')  # Đóng cửa sổ hiển thị
print(f"Chuyển sang chế độ: {mode}")
else:
mode = "detection"
lcd.clear()
lcd.write_string("Mode: Detection")  # Cập nhật lên LCD
print(f"Chuyển sang chế độ: {mode}")
elif mode == "manual":
led_states[i] = not led_states[i]
GPIO.output(LED_PINS[i], led_states[i])
print(f"LED{i} is now {'ON' if led_states[i] else 'OFF'}")
time.sleep(0.3)  # Thời gian chống nhấp nháy
time.sleep(0.1)
# Chạy chương trình chính
def main():
threading.Thread(target=button_handler, daemon=True).start()
camera = cv2.VideoCapture(0)  # Sử dụng camera USB
if not camera.isOpened():
print("Không thể mở camera.")
return
try:
while True:
ret, frame = camera.read()
if not ret:
print("Không thể nhận dữ liệu từ camera.")
break
if mode == "detection":
# Xử lý từng khung hình trong chế độ nhận diện
process_frame(frame)
# Thoát nếu nhấn phím 'q'
if cv2.waitKey(1) & 0xFF == ord('q'):
break
except KeyboardInterrupt:
pass
finally:
camera.release()
cv2.destroyAllWindows()
clear_all()
GPIO.cleanup()
# Gọi hàm main
if __name__ == "__main__":
main()