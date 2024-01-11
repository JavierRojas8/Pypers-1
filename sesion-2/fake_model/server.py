from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

class Item(BaseModel):
    instances: list

app = FastAPI()

@app.get("/salud")
async def salud():
    return {"status": 200}


@app.post("/predict")
async def create_item(item: Item):
    fake_class = np.random.randint(10)
    return {"predictions": [int(fake_class)]}
