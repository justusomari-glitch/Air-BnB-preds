import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app=FastAPI()
model1=joblib.load("models/city_prediction.pkl")
model2=joblib.load("models/price_prediction.pkl")

print(model1.feature_names_in_)
print(model2.feature_names_in_)

class ListingInput(BaseModel):
    price_per_night: float
    review_score: float
    number_of_reviews: int
    availability_365: int
    room_type: str
    city: str
    neighbourhood: str
    property_type: str
    min_amount_per_property: float

@app.get("/")
def home():
    return {"message": "Welcome To Airbnb Prediction"}


@app.post("/predict")
def predict(data: ListingInput):
        input_dict= data.dict()
        df=pd.DataFrame([input_dict])
        city_features=model1.feature_names_in_
        price_features=model2.feature_names_in_
        city_input=df[city_features]
        price_input=df[price_features]
        predicted_city=model1.predict(city_input)[0]
        predicted_price=model2.predict(price_input)[0]
        return {
            "predicted_city": predicted_city,
            "predicted_price": predicted_price
        }
        