from fastapi import FastAPI,Form,Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
import joblib 
import pandas as pd 
import os
app=FastAPI() 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_smote_pipeline.joblib")
pipeline=joblib.load(MODEL_PATH)
templates=Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
#app.mount("/static",StaticFiles(directory="static"),name="static")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict",response_class=HTMLResponse)
async def predict(
    request:Request,
    Power:float=Form(...),
    OSF:float=Form(...),
    PWF:float=Form(...),
    HDF:float=Form(...),
    TWF:float=Form(...),
    Torque_Nm:float=Form(...),
    Rotational_speed_rpm:float=Form(...),
    Temp_difference:float=Form(...)

):
    try:
        input_data=pd.DataFrame([{
            "Power":Power,
            "OSF":OSF,
            "PWF":PWF,
            "HDF":HDF,
            "TWF":TWF,
            "Torque [Nm]":Torque_Nm,
            "Rotational speed [rpm]":Rotational_speed_rpm,
            "Temp_Difference":Temp_difference
        }])

        prediction=pipeline.predict(input_data)[0]
        if prediction==1:
            prediction_text="Machine Failure"
        else:
            prediction_text="No Failure"
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction_text})
    except Exception as e:
        return JSONResponse(content={"error":str(e)})
        
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)