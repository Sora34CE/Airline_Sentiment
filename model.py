import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from pydantic import BaseModel
import joblib


# Class that details the airlines
class Airline(BaseModel):
    UNITED: int = 0
    USAIRWAYS: int = 0
    JETBLUE: int = 0
    AMERICANAIR: int = 0


# Trains the model and makes predictions
class AirlineModel:
    # Loads the dataset and loads the model
    # if exists. If not, calls the _train_model method and 
    # saves the model
    def __init__(self):
        self.df = pd.read_csv('airline_sentiment_analysis.csv')
        self.df.columns=['id', 'sentiment', 'comment']
        self.model_fname_ = 'airline_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model(self.df)
            joblib.dump(self.model, self.model_fname_)

    # Perform model training using the LogisticRegression classifier
    def _train_model(self, df):
        # Pre-processing
        df['comment'] = df.comment.str.upper()
        df.sentiment = df.sentiment.map({'positive': 1, 'negative': 0}) # turn sentiment to numeric
        # Generate the airline column
        airlines = ["UNITED", "USAIRWAYS", "JETBLUE", "AMERICANAIR"]
        df4 = pd.DataFrame()
        for a in airlines:    
            df2 = df.loc[df['comment'].str.contains(a)].copy()  # important to use .copy(), else next line throws error
            df2['airline'] = a
            # concat       
            df2.reset_index(drop=True, inplace=True) # important to reset index, else cancat won't work
            df4 = pd.concat([df4, df2], axis=0)
        # Preparing for X and y datasets
        df5 = df4[['airline', 'sentiment']]
        dum_airlines = pd.get_dummies(df5.airline)

        X = dum_airlines
        y = df5['sentiment']

        # building model
        logreg = LogisticRegression(random_state=0, class_weight='balanced')
        model = logreg.fit(X, y)
        return model

    # Make a prediction based on the user-entered data
    # Returns the predicted sentiment
    def predict_sentiment(self, airline_dict):
        df9 = pd.DataFrame(airline_dict.values()).T
        prediction = self.model.predict(df9)
        return prediction[0]

model = AirlineModel()