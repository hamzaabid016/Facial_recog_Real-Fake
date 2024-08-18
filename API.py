from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uuid
import cv2
import numpy as np
import base64
import os
import database as db
from train_v2 import get_encoding  # Importing the function from train.py
from main import detect_live_stream, data_setting


app = FastAPI()

@app.post("/update_train/")
async def update_train(
    name: str = Form(...),
    ref: str = Form(...),
    summary: str = Form(...),
    image: UploadFile = File(...)
):
    uu_id = str(uuid.uuid4())
    ref_no_exists = db.check_ref_no_exists(ref)

    if ref_no_exists:
        return {"status": "error", "message": "ref_no already exists in the Database."}
    else:
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        image_encodes = get_encoding(img)
        if image_encodes is None:
            return {"status": "error", "message": "Failed to encode images."}
        
        # Convert image to base64 for storing in the database
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        status = db.insert_data(uu_id, name, ref, summary, image_base64, image_encodes)
        return {"status": "success", "message": status}






@app.get("/live_stream/")
async def live_stream():
    try:
        datas_s = db.get_all_data()
        datas = data_setting(datas_s)
        detect_live_stream(datas)
        return {"status": "Live Stream Finished"}
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")





# Running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
