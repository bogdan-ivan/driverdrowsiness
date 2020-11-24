import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


class KNeighborsWrapper:
    """This class brings nothing extra to the KNeighborsClassifier.
    the main purpose of this class is to fit
    the data colected in the drowsiness_data_set.csv and predict in real time
    what the data from a given frame can tell about an individuals awareness"""
    """def __init__(self):
        self.classifier=RandomForestClassifier(n_estimators=100)
        self.scaler=MinMaxScaler()
        data_set=pd.read_csv("merged_data.csv")
        #data_set.drop(data_set.columns[13],axis=1,inplace=True)
        data = data_set.values
        np.random.shuffle(data)
        x=data[:,:-1]
        self.scaler.fit(x)
        x=self.scaler.transform(x)
        y=data[:,-1]
        self.classifier.fit(x,y)
        joblib.dump(self.classifier,"RandomForest.sav")
        joblib.dump(self.scaler, "ScalerRF.sav")
        """
    def __init__(self):
        self.classifier=joblib.load("RandomForest.sav")
        self.scaler=joblib.load("ScalerRF.sav")

    def predict(self,listOfFeatures):
        listOfFeatures=self.scaler.transform(listOfFeatures)
        return self.classifier.predict(listOfFeatures)

    def predict_proba(self,listOfFeatures):
        listOfFeatures=self.scaler.transform(listOfFeatures)
        return self.classifier.predict(listOfFeatures)