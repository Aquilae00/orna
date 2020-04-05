from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()

model_path = "./models/detection_model-ex-056--loss-0018.541.h5"
input_path = "./input/test1.png"
output_path = "./output/newimage.jpg"
json_path = './cusd/json/detection_config.json'
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.setJsonPath(json_path)
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
