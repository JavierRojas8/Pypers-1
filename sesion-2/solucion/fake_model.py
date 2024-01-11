from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel


class Item(BaseModel):
    instances: list

app = FastAPI()

@app.get("/healthcheck")
async def salud():
    return {"status":200}

@app.post("/predict")
async def create_item(item: Item):
    return {"predictions": [int(np.random.randint(10)) ] } 

