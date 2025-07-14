from fastapi import FastAPI,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()

MODEL = tf.keras.models.load_model("../models/1.keras")

# endpoint = "http://localhost:8501/models/:predict"
# MODEL =
CLASS_NAMES= ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
@app.get("/ping")
async def ping():
    return {"message": "Hello, World!"}
def read_file_as_imag(data) ->np.ndarray:
    """Convert bytes data to numpy array image."""
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    bytes = await file.read()
    image = read_file_as_imag(bytes)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    # response = requests.post(
    #     endpoint,
    #     json={
    #         "signature_name": "serving_default",
    #         "instances": img_batch.tolist()
    #     }
    # )
    # response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # return {"filename": file.filename}

    return {
        'class': predicted_class,
        'confidence': float(confidence),  
    }
    
   
    
    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)