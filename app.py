import cv2

thres = 0.5  # Threshold to detect object

cap = cv2.VideoCapture(0)  # Change camera index to 0 for default webcam
cap.set(3, 640)
cap.set(4, 480)

classnames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)

configPath = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId <= len(classnames):  # Check if classId is within the range of classnames
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classnames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Unknown Class', (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()