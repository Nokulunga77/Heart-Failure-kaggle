"""
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
from enum import unique
from inspect import Parameter
from typing import Sized
from PIL.Image import alpha_composite
from google.protobuf import message
from google.protobuf.message import Message
import nltk
from pandas.core.frame import DataFrame
import streamlit as st
import joblib,os
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
# Data dependencies
import pandas as pd
import re
import numpy as np
import requests
import io
import altair as alt
from collections import Counter
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import boxcox, zscore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures


# Load your raw data
url = "https://github.com/Nokulunga77/Heart-Failure-kaggle/blob/main/heart.csv?raw=true" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe
df = pd.read_csv(io.StringIO(download.decode('utf-8')))


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Main", "Understand your data", "Predictions", "About"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Buidling out the "Main" page
	if selection == "Main":
		st.title("Heart Failure Prediction")
		st.image('https://images.ctfassets.net/yixw23k2v6vo/6BezXYKnMqcG4LSEcWyXlt/b490656e99f34bc18999f3563470eae6/iStock-1156928054.jpg?fm=webp&fit=thumb&q=65&w=864&h=576', use_column_width=False)
		st.markdown("""
		Created by:
		* **Nokulunga Twala**
		* **Kwanda Mazibuko**
		""")
		
	
	
	# Building out the "Information" page
	if selection == "Understand your data":
		#st.title("(EDA), understanding the data!")
		#st.image('Images/eda.jpeg', use_column_width=False)
		#st.dataframe(raw)
		#st.subheader("Using graphs we can understand the data better, so from here we will look at the type of hashtags being used as well as what the data implies:")
		st.info("""Defining the data:   
		* **Age:** age of the patient [years]   
		* **Sex:** sex of the patient [M: Male, F: Female]    
		* **ChestPainType:** chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]  
		* **RestingBP:** resting blood pressure [mm Hg]
		* **Cholesterol:** serum cholesterol [mm/dl]
		* **FastingBS:** fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
		* **RestingECG:** resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality , LVH: showing probable or definite left ventricular hypertrophy]
		* **MaxHR:** maximum heart rate achieved [Numeric value between 60 and 202]
		* **ExerciseAngina:** exercise-induced angina [Y: Yes, N: No]
		* **Oldpeak:** oldpeak = ST [Numeric value measured in depression]
		* **ST_Slope:** the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
		* **HeartDisease:** output class [1: heart disease, 0: Normal]
		""")

		
		
	
	# Buidling out the "prediction" page
	if selection == "Predictions":
		st.title("Predictions:")
		st.subheader("In this section the user will input their information:")

		def user_input_features():
			st.write("""**1. Select Age :**""") 
			age = st.slider('', 0, 100, 25)

			st.write("""**2. Select Gender :**""")
			sex = st.selectbox("(1=Male, 0=Female)",["1","0"])

			st.write("""**3. Select Chest Pain Type :**""")
			cp = st.selectbox("(0 = Typical Angina, 1 = Atypical Angina, 2 = Non—anginal Pain, 3 = Asymptotic) : ",["0","1","2","3"])

			st.write("""**4. Select Resting Blood Pressure :**""")
			restbps = st.slider('In mm/Hg unit', 0, 200, 110)

			st.write("""**5. Select Serum Cholesterol :**""")
			chol = st.slider('In mg/dl unit', 0, 600, 115)

			st.write("""**6. Select Fasting Blood Sugar :**""")
			fastingBS = st.selectbox("(0 =lower than 120mg/ml, 1 = Greater than 120mg/ml)",["0", "1"])
		
			st.write("""**7. Selecting RestingECG :**""")
			restingecg= st.selectbox("(3 = Normal, 1.5 = ST, 1 = LVH])",["3","1.5", "1"])

			st.write("""**8. Maximum Heart Rate Achieved:**""")
			maxhr = st.slider('', 0, 202, 60)

			st.write("""**9. Exercise Induced Angina (Pain in chest while exersice) :**""")
			exang = st.selectbox("(1=Yes, 0=No)",["1","0"])
			
			st.write("""**10. Oldpeak (ST depression induced by exercise relative to rest) :**""")
			oldpeak = float(st.slider('', 0.0, 10.0, 2.0))

			st.write("""**11. Slope (The slope of the peak exercise ST segment) :**""")
			slope = st.selectbox(" 1 = up, 2 = flat, 0 = down )",["1","2","0"])
			


			data = {'Age': age, 'Sex': sex, 'ChestPainType': cp,'RestingBP': restbps, 'Cholesterol': chol, 
			'fastingBS': fastingBS, 'RestingECG': restingecg, 'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak, 'ST_slope': slope}
			
			features = pd.DataFrame(data, index=[0])

			return features

		df = user_input_features()
		st.subheader('Given Inputs : ')
		st.write(df)
		
		st.write("""**If you're satisfied with the inputs, click Classify to get your results**""")
		

		if st.button("Classify"):
			# Transforming user input with vectorizer
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			result = "rfc_model.pkl"
			predictor = joblib.load(open(os.path.join(result),"rb"))
			prediction = predictor.predict(df)
			
			if prediction == 1:
				st.success("You are likely to suffer from heart disease")
			else:
			        st.success("You are less likely to suffer from heart disease")
    

    

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
