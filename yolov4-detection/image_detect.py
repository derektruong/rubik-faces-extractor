import cv2

def rubik_face_detector(img_path, order = 1):
	img = cv2.imread(img_path)

	with open('./detect_data/label.names', 'r') as f:
		classes = f.read().splitlines()
		
	# net = cv2.dnn.readNetFromDarknet('./detect_data/yolov4-custom.cfg', './detect_data/yolov4-custom_last.weights')
	net = cv2.dnn.readNetFromDarknet('./detect_data/yolov4-tiny-custom.cfg', './detect_data/yolov4-tiny-custom_best.weights');

	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

	def detector():
		classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
		# print(classIds, scores, boxes)
		if (len(classIds) == 0 or len(scores) == 0 or len(boxes) == 0): return

		classId, score, box = classIds[0], scores[0], boxes[0]
        # print(box)
		cv2.rectangle(
            img,
            (box[0], box[1]),
            (box[0] + box[2], box[1] + box[3]),
            color=(0, 255, 0),
            thickness = 2
        )
        
		# text = f"{classes[classId]} - {score * 100}%"
        
		cv2.putText(
            img, 'rubik',
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color = (0, 255, 0),
            thickness = 2
        )
        
        # crop image
		# crop_img = img[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
		
		# cv2.imshow('Image', img)
		cv2.imwrite(f"./output/images/res_{order}.jpg", img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	detector()
  
def main():
	for i in range(1, 101):
		rubik_face_detector(f"./dataset/ras/image{i}.jpg", i)
