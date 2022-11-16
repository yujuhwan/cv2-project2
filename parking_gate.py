# 주차장 차단기
import cv2  # 웹캠으로 비디오 읽기
import numpy as np  # 연산
from dynamikontrol import Module  # 모터 제어 모듈

CONFIDENCE = 0.9  # 차인가 아닌가 판별 0~1사이로 설정 90%이상 차로 인식
THRESHOLD = 0.3   # Non-Maximum Suppression의 쓰레쉬홀드
LABELS = ['Car', 'Plate']  # 차량, 번호판
CAR_WIDTH_TRESHOLD = 500  # 자동차의 크기가 500pixel 이상일 때 모터 on (수정)

cap = cv2.VideoCapture(0)  # 웹캠 on

net = cv2.dnn.readNetFromDarknet('cfg/yolov4-ANPR.cfg', 'yolov4-ANPR.weights')  # dnn의 모델 열기

module = Module()  # dynamikontrol initialize

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    H, W, _ = img.shape  # 웹캠의 세로 가로 사이즈

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)  # 전처리,
    net.setInput(blob)  # 전처리한 이미지 설정
    output = net.forward()

    boxes, confidences, class_ids = [], [], []  # 네모칸, 컨피던스, 클래스아이디 변수 설정

    for det in output:
        box = det[:4]  # darknet 박스의 정보 x,y,w,h
        scores = det[5:]  # 라벨의 score
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:  # 0.9 보다 크면
            cx, cy, w, h = box * np.array([W, H, W, H])  # 0~1사이이므로 원래 픽셀크기로 설정
            x = cx - (w / 2)  # 좌상단의 x
            y = cy - (h / 2)  # 좌상단의 y

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # 압축된 index

    if len(idxs) > 0:
        for i in idxs.flatten():  # 1차원으로 펴줌
            x, y, w, h = boxes[i]

            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)  # 사각형 그리기
            cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

            if class_ids[i] == 0:  # 모터 제어, 0 == car,
                if w > CAR_WIDTH_TRESHOLD:
                    module.motor.angle(80)  # 모터 on
                else:
                    module.motor.angle(0)  # 모터 off
    else:
        module.motor.angle(0)  # 인식 못했을 경우 모터 off

    cv2.imshow('result', img)  # 결과 확인
    if cv2.waitKey(1) == ord('q'):
        break
