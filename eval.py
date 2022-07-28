import pandas as pd 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from pydantic import BaseModel

# Build and evaluate airline-sentiment models using multiple classifiers 
class AirlineModelEval:
    # Loads the dataset and loads the model
    # if exists. If not, calls the _train_model method and 
    # saves the model
    def __init__(self):
        self.df = pd.read_csv('airline_sentiment_analysis.csv')
        self.df.columns=['id', 'sentiment', 'comment']

    # Build models using multiple classifiers and generate evaluation metrics
    def _eval_models(self, df):
        # Pre-processing
        df['comment'] = df.comment.str.upper()
        df.sentiment = df.sentiment.map({'positive': 1, 'negative': -1}) # turn sentiment to numeric
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

        # split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
        
        # Evaluating different classifiers
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="rbf", C=0.025, probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            MLPClassifier(hidden_layer_sizes=(64,64,64), activation='relu', solver='adam', max_iter=500),
            LogisticRegression(random_state=0, class_weight='balanced')    
        ]
        # Logging metrics for comparison
        log_cols=["Classifier", "Accuracy", "Log Loss"]
        log = pd.DataFrame(columns=log_cols)
        for clf in classifiers:
            model = clf.fit(X_train, y_train)
            name = clf.__class__.__name__
            
            print("="*30)
            print(name)           
            print('****Results****')
            train_predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, train_predictions)
            print(accuracy)
            print(confusion_matrix(y_test, train_predictions))
            print("classification report\n")
            print(classification_report(y_test,train_predictions))
            
            train_probas = clf.predict_proba(X_test)
            ll = log_loss(y_test, train_probas)
            print("Log Loss: {}".format(ll))
            
            log_entry = pd.DataFrame([[name, accuracy*100, ll]], columns=log_cols)
            log = pd.concat([log, log_entry], axis=0)
          
        print("="*30)
        print(log)

# Train and evaluate model
models = AirlineModelEval()
models._eval_models(models.df)




