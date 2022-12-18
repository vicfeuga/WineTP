from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
import csv
from model.functions import *
from model.model import * 

app = FastAPI()

#uvicorn main:app --reload

@app.get("/")
async def root():
    """message d'accueil

    Returns:
        message: hello world
    """
    return {"message": "Hello World"}

@app.get("/api/predict")
async def perfect_wine():
    """renvoie le vin parfait

    Raises:
        HTTPException: impossible de trouver le vin

    Returns:
        perfect_wine: le meilleur vin
    """
    perfect_wine = get_perfect_wine()
    if perfect_wine == -1:
       raise HTTPException(status_code=500, detail="Server failed to find perfect wine") 
    return perfect_wine;

@app.post("/api/predict")
async def predict_grade(wine: Wine):
    """predit le quality d'un vin

    Args:
        wine (Wine): le vin a predir

    Raises:
        HTTPException: impossible de predire la note

    Returns:
        grade: la prédiciton du model
    """
    data = {
        "fixed_acidity": wine.fixed_acidity,
        "volatile_acidity": wine.volatile_acidity,
        "citric_acidity": wine.citric_acidity,
        "residual_sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free_sulfur_dioxide": wine.free_sulfur_dioxide,
        "total_sulfur_dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "ph": wine.ph,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol 
    }
    result = int((predict(data))[0])
    if result != 1 and result != 0 :
        raise HTTPException(status_code=500, detail="Server failed to predict grade")
    return {"grade": result}

@app.get("/api/model")
async def get_model():
    """recuper le model de prediciton

    Raises:
        HTTPException: impossible de trouver le model serialized

    Returns:
        le model serialized format .z
    """
    serialize_model()
    serialized_file_exists = exists('./resources/serialized_model.z')
    if serialized_file_exists:
        return FileResponse('./resources/serialized_model.z')
    raise HTTPException(status_code=404, detail="Serialized model not found")

@app.get("/api/model/description")
async def get_model_info():
    """recuper les infos du model

    Raises:
        HTTPException: unale to find model infos

    Returns:
        les informations du model
    """
    metrics = get_model_metrics()
    if metrics == -1 :
        raise HTTPException(status_code=404, detail="Unable to find model infos you should considere regenerating the model")
    parameter = Parameters(C=metrics["C"], gamma=metrics["gamma"], kernel=metrics["kernel"])
    metric = Metrics(break_ties=metrics["break_ties"], cache_size=metrics["cache_size"], class_weight=metrics["class_weight"], coef0=metrics["coef0"], decision_function_shape=metrics["decision_function_shape"], degree=metrics["degree"], max_iter=metrics["max_iter"], probability=metrics["probability"], random_state=metrics["random_state"], shrinking=metrics["shrinking"], tol=metrics["tol"], verbose=metrics["verbose"])
    model = Model(parameters=parameter, metrics=metric)
    return model
    

@app.put("/api/model", status_code=status.HTTP_201_CREATED)
async def insert_wine(wine: Wine):
    """ajouter un vin aux données

    Args:
        wine (Wine): nouveau vin

    Raises:
        HTTPException: erreur d'insertion
    """
    data = {
        "fixed_acidity": wine.fixed_acidity,
        "volatile_acidity": wine.volatile_acidity,
        "citric_acidity": wine.citric_acidity,
        "residual_sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free_sulfur_dioxide": wine.free_sulfur_dioxide,
        "total_sulfur_dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "ph": wine.ph,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol 
    }
    result = int((predict(data))[0])
    resultI = insertWine(wine, result)
    if resultI == -1:
         raise HTTPException(status_code=500, detail="Server failed to insert wine")


@app.post("/api/model/retrain")
async def retrain_model():
    """reentraine le model de prediction

    Raises:
        HTTPException: impossible de reentrainer le model
    """
    save_model(train_model())
    model_file_exists = exists('./resources/model.sav')
    if model_file_exists == False:
        raise HTTPException(status_code=500, detail="Server failed to retrain model")
    
