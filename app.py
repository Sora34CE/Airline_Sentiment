import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from model import AirlineModel, Airline

# Create the app and model objects
app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})
model = AirlineModel()

@app.get('/')
def index():
    return {'message': 'Hello, traveler!'}
# Expose prediction functionality, make predictions from inputted
# Airline name and return the predicted sentiment

@app.get('/predict/{name}')
def predict_sentiment(name):
    airline = Airline()
    data = airline.dict()
    if name.upper() not in data.keys():
        return "This airline does not exist in the data."
    data.update({name.upper() : 1})
    pred = model.predict_sentiment(data)
    if pred == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return {
        'prediction': sentiment
    } 

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Airline Sentiment Analysis",
        version="1.0.0",
        description="Find sentiments for airlines",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Run the API with uvicorn on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

app.openapi = custom_openapi