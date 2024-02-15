from ultralytics import YOLO
from bentoml.io import JSON
import bentoml 
import typing as t
import cv2
import base64 
import numpy as np
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

class yolov8(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.model = YOLO("yolov8m.pt")

    @bentoml.Runnable.method(batchable=False)
    def predict(self, input):
        results=self.model(input)
        return results[0].boxes.data.cpu().numpy()
    
yolo_runner  = t.cast(
    "RunnerImpl", bentoml.Runner(yolov8, name="yolov8")
)
svc = bentoml.Service("yolov8", runners=[yolo_runner])

@svc.api(input=JSON(), output=JSON())
def predict(input):
    image = np.frombuffer(base64.b64decode(input["instances"][0]["image"]) , dtype=np.uint8)
    image = cv2.imdecode(image,1)
    print(image.shape)
    return {"predictions": yolo_runner.predict.run(image)}
        