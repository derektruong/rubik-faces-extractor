import cv2

# global variables
detected_state = 1
face_index = 0
current_face = ['Front', 'Back', 'Right', 'Left', 'Up', 'Down']

# init programs
with open('./detect_data/label.names', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('./detect_data/yolov4-tiny-custom.cfg', './detect_data/yolov4-tiny-custom_best.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale = 1/255, size = (416, 416), swapRB=True)

def frame_detect(frame):    
    global detected_state
    global face_index
    global current_face

    try:
        classIds, scores, boxes = model.detect(
            frame,
            confThreshold = 0.6,
            nmsThreshold = 0.4
        )
        
        if (len(classIds) == 0 or len(scores) == 0 or len(boxes) == 0):
            return     

        classId, score, box = classIds[0], scores[0], boxes[0]
        
        # crop image
        expand_size = 15
        x, y, w, h = box[0]-expand_size, box[1], box[2]+expand_size*2, box[3]
        
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color=(0, 255, 0),
            thickness = 2
        )
        print(box)
        
        text = f"{classes[classId[0]]} - {score * 100}%"
        print(x, y, w, h)
        
        cv2.putText(
            frame, text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color = (0, 255, 0),
            thickness = 2
        )
        
        crop_img = frame[y : y + h, x : x + w]
        cv2.imwrite(f"./output/cam/Frame_{current_face[face_index]}.jpg", crop_img)
        
        face_index += 1
        detected_state = 1
    except Exception as err:
        print(err)
        print("hehe")
    
def detector():
    global detected_state
    global face_index
    global current_face

    vid = cv2.VideoCapture(0)
    vid. set(cv2.CAP_PROP_FPS, 60)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output/output.avi',fourcc, 30, (1280,720))
    
    while True:
        ret, frame = vid.read()
        
        if ret == True:
            b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(b)
            
            if (detected_state == 0):
                frame_detect(frame)
            
            if face_index < 6:
                text = f"Show {current_face[face_index]} and press 's' to save the face"
                cv2.putText(
                    frame, text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color = (0, 255, 0),
                    thickness = 2
                )

            cv2.imshow('frame', frame)
            
            if face_index > 5:
                break
            
            if face_index < 6 and cv2.waitKey(1) & 0xFF == ord('s'):
                detected_state = 0
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
        
    vid.release()
    cv2.destroyAllWindows()