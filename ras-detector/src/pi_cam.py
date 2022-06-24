import cv2
import serial
import time
from colorizer import colorizer
from two_phase import solver
import twophase.solver  as sv

class PiCam:
    # class variables
    ser = serial.Serial("/dev/ttyUSB0",9600)
    isSent = False
    detected_state = 1
    face_index = 0
    current_face = ["Down", "Front", "Up", "Back", "Right", "Left"]
    
    # Yolov4 tiny
    net = cv2.dnn.readNetFromDarknet(
        "./detect_data/yolov4-tiny-custom.cfg",
        "./detect_data/yolov4-tiny-custom_final.weights",
    )
    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    
    # init classes
    with open("detect_data/label.names", "r") as f:
        classes = f.read().splitlines()

    @staticmethod
    def frame_detect(frame):
        try:
            classIds, scores, boxes = PiCam.model.detect(
                frame,
                confThreshold = 0.6,
                nmsThreshold = 0.4
            )
            
            if (len(classIds) == 0 or len(scores) == 0 or len(boxes) == 0):
                return {}     

            classId, score, box = classIds[0], scores[0], boxes[0]
            
            # crop image
            expand_size = 30
            x, y, w, h = box[0], box[1], box[2], box[3]+expand_size*2
            
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color=(0, 255, 0),
                thickness = 2
            )
            
            text = f"{PiCam.classes[classId]} - {score * 100}%"
            
            cv2.putText(
                frame, text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color = (0, 255, 0),
                thickness = 2
            )
            
            crop_img = frame[y : y + h, x : x + w]
            
            res = colorizer.colorize(crop_img, PiCam.current_face[PiCam.face_index])
            
            PiCam.face_index += 1 if res['is_success'] else 0
            # PiCam.detected_state = 1
            return res
        except Exception as err:
            print(err)
            return {}
            
    @staticmethod
    def takeOneFrame():
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FPS, 60)
        
        while True:
            ret, frame = video.read()
            try:
               frame = cv2.resize(frame,(800,600))
            except:
                pass
            if ret == True:
                video.release()
                return frame
        
            
    @staticmethod
    def detector():
        final_result = {}
        for index, face in enumerate(PiCam.current_face):
            if index != 0:
                data = 'D' + str(PiCam.current_face[PiCam.face_index][0])
                PiCam.ser.write(str.encode(data))
                time.sleep(7)
            
            isDetected = False
#             frame = PiCam.takeOneFrame()
            while not isDetected:
                frame = PiCam.takeOneFrame()
                res = PiCam.frame_detect(frame)
                isDetected = res['is_success']
                if isDetected:
                    final_result[face] = res['colors']
        
        resSolver = solver.transformationColorFaces(final_result['Up'], final_result['Right'],
                                                    final_result['Front'], final_result['Down'],
                                                    final_result['Left'], final_result['Back'])
        print(resSolver[:-5])
        PiCam.ser.write(str.encode(resSolver[:-5]))
            
#             cv2.imshow(f'frame{index}', frame)
#             cv2.waitKey()
            
#         
#         ind = 0
#         isSent = False
#         PiCam.detected_state = 0
#         while ind < 6:
#             frame = PiCam.takeOneFrame(vid)
#             cv2.imshow('frame', frame)
#             isDetected = False 
#             if ind == 0:
#                 isDetected = PiCam.frame_detect(frame)
#             elif ind > 0:
#                 if not isSent:
#                     data = 'D' + str(PiCam.current_face[PiCam.face_index][0])
#                     PiCam.ser.write(str.encode(data))
#                     time.sleep(10)
#                 isSent = True
#                 isDetected = PiCam.frame_detect(frame)
#             
#             if isDetected:
#                 isSent = False
#                 ind += 1
#                 PiCam.detected_state = 0
                
            
        
#         while True:
# #             time.sleep(5)
#             ret, frame = vid.read()
#             frame = cv2.resize(frame,(800,600))
#             if ret == True:
#                 isDetected = False
#                 if (PiCam.detected_state == 0):
#                     isDetected = PiCam.frame_detect(frame)
#                     if not isDetected: continue
#                 
#                 PiCam.detected_state = 1
#                 
#                 if PiCam.face_index < 6:
#                     if PiCam.face_index != 0:
#                         data = 'D' + str(PiCam.current_face[PiCam.face_index][0])
#                         PiCam.ser.write(str.encode(data))
#                         time.sleep(5)
# #                     data = 'D' + str(PiCam.current_face[PiCam.face_index][0])
# #                     PiCam.ser.write(str.encode(data))
#                     #print(data)
#                     text = f"Show {PiCam.current_face[PiCam.face_index]} and press 's' to save the face"
#                     cv2.putText(
#                         frame, text,
#                         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         color = (0, 255, 0),
#                         thickness = 2
#                     )
#                     
#                     PiCam.detected_state = 0
#                     
#                 
#                 cv2.imshow('frame', frame)
#                 
# #                 if PiCam.face_index < 6 and cv2.waitKey(1) & 0xFF == ord('s'):
# #                     PiCam.detected_state = 0
# #                     
#                 if PiCam.face_index >= 6: break    
#                 
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             
#             else:
#                 break
            
        cv2.destroyAllWindows()
        
pi_cam = PiCam()
