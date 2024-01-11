import bentoml
from bentoml.io import JSON
import numpy as np
import torch
import cv2
from torchvision.models import vit_b_16 , ViT_B_16_Weights

weights_vitb16 = ViT_B_16_Weights.DEFAULT
preprocess_vit = weights_vitb16.transforms()
vitb16 = vit_b_16(weights=weights_vitb16)

bentoml.pytorch.save_model(
    "vit16",   # Model name
    vitb16,  # objeto que contiene al modelo
)
vit_runner = bentoml.pytorch.get("vit16:latest").to_runner()
svc = bentoml.Service("vit16", runners=[vit_runner])
example= np.zeros((1,3,224,224))

@svc.api(input=JSON(), output=JSON())
def classify(input_series):
    result = vit_runner.run(example).argmax().numpy().tolist()
    return {"predictions": result}
