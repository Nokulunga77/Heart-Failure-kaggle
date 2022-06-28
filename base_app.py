"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------
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
	options = ["Main","Understand your data", "Modelling", "Predictions", "About"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Buidling out the "Main" page
	if selection == "Main":
		st.title("Heart Failure:")
		st.image('heart.jpg', use_column_width=False)
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

		
	# Building out the predication page
	if selection == "Modelling":
		st.title("Modelling:")
		st.image('Images/perform.jpg', use_column_width=False)
		st.subheader("From here we can see the different models we used as well as how they performed:")
		
		st.header('Each model that we use, with their positives and negatives:')

		col1, col2, col3, col4 = st.columns(4)

		with col1:
			st.header('Logistic Regression')
			st.info('''
			Positives:
			* Is easier to impliment, interpret, and very efficient to train .
			* It makes no assumptionsof classes in feature space .
			* It can easily extend to multiple classes(multinomial regression) and a natural probabilistic view of class predictions. 
			
			Negatives:
			* If the number of observations is lesser than the number of features, Logistic Regression should not be used, otherwise, it may lead to overfitting.
			* It constructs linear boundaries.
			* The major limitation is the assumption of linearity between the dependent variable and the independent variables.
			''')

		with col2:
			st.header('Random Forest Classifier')
			st.info('''
			Positives:
			* Can be used to solve both classification as well as regression problems.
			* Works well with both categorical and continuous variables.
			* It creates as many trees on the subset of the data and combines the output of all the trees. In this way it reduces overfitting problem in decision trees and also reduces the variance and therefore improves the accuracy.
			
			Negatives:
			* Complexity is increase because it creates multiple trees and combines their outputs
			* Training period takes longer because it takes a majority of the votes from the trees
			''')

		with col3:
			st.header('Linear Support Vector Classification')
			st.info('''
			Positives:
			* Works relatively well when there is a clear margin of separation between classes.
			* Is more effective in high dimensional spaces.
			* Is effective in cases where the number of dimensions is greater than the number of samples. 
			
			Negatives:
			* Algorithm is not suitable for large data sets.
			* Does not perform very well when the data set has more noise i.e. target classes are overlapping.
			* In cases where the number of features for each data point exceeds the number of training data samples, the SVC will underperform.
			''')

		with col4:
			st.header('XGBClassifier')
			st.info('''
			Positives:
			* Less feature engineering required
			* Fast to interpret
			* Outliers have minimal impact.
			
			Negatives:
			* Harder to tune as there are too many hyperparameters.
			* Difficult interpretation , visualization tough
			* Overfitting possible if parameters not tuned properly.
			''')
		st.header('How each model performed on an f1-score:')
		st. latex(r'''F1 score = 2 \left(\frac{Precision * Recall}{Pecision + Recall}\right)''')
		st.info('F1 score is the measure of a tests accuracy or in this case our models accuracy. It is calculated as shown above, where the precision is the number of true positive results is devided by the number of all positive results.')
		st.image('Images/f1.jfif', use_column_width=False)

		tweet_text = st.text_area("Enter Text","Type here")
		Logistic = ("lr_model.pkl")
		Random = ("rfc_model.pkl")
		Linear = ('lsvc_model.pkl')
		XGBmodel = ('xgb_model.pkl')

		original_list = [Logistic, Random, Linear, XGBmodel]
		st.info('''
		Models:
		* **LogisticRegression Model** = lr_model.pkl
		* **RandomForestClassifier Model** = rfc_model.pkl
		* **LinearSupportVectorClassifier Model** = lsvc_model.pkl
		* **XGBoosterClassifier** = xgb_model.pkl
		''')
		result = st.selectbox('Select the model you want to use:', original_list)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(result),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		st.info(''' 
		Categorized as:
		* **Pro** = 1
		* **Anti** = -1
		* **News** = 2
		* **Neutral** = 0
		
		''')
	
	# Buidling out the "prediction" page
	if selection == "Predictions":
		st.title("Predictions:")
		st.subheader("In this section the user will input their information:")

		def user_input_features():
			st.write("""**1. Select Age :**""") 
			age = st.slider('', 0, 100, 25)
			st.write("""**You selected this option **""",age)

			st.write("""**2. Select Gender :**""")
			sex = st.selectbox("(1=Male, 0=Female)",["1","0"])
			st.write("""**You selected this option **""",sex)

			st.write("""**3. Select Chest Pain Type :**""")
			cp = st.selectbox("(0 = Typical Angina, 1 = Atypical Angina, 2 = Non—anginal Pain, 3 = Asymptotic) : ",["0","1","2","3"])
			st.write("""**You selected this option **""",cp)

			st.write("""**4. Select Resting Blood Pressure :**""")
			trestbps = st.slider('In mm/Hg unit', 0, 200, 110)
			st.write("""**You selected this option **""",trestbps)

			st.write("""**5. Select Serum Cholesterol :**""")
			chol = st.slider('In mg/dl unit', 0, 600, 115)
			st.write("""**You selected this option **""",chol)

			st.write("""**6. Select Fasting Blood Sugar :**""")
			fastingBS = st.selectbox("(0 =lower than 120mg/ml, 1 = Greater than 120mg/ml)",["0", "1"])
			st.write("""**You selected this option **""",fastingBS)

			st.write("""**7. Selecting RestingECG :**""")
			restingecg= st.selectbox("(3 = Normal, 1.5 = ST, 1 = LVH])",["3","1.5", "1"])
			st.write("""**You selected this option **""",restingecg)

			st.write("""**8. Maximum Heart Rate Achieved:**""")
			maxhr = st.slider('', 0, 202, 60)
			st.write("""**You selected this option **""",maxhr)

			st.write("""**9. Exercise Induced Angina (Pain in chest while exersice) :**""")
			exang = st.selectbox("(1=Yes, 0=No)",["1","0"])
			st.write("""**You selected this option **""",exang)
			
			st.write("""**10. Oldpeak (ST depression induced by exercise relative to rest) :**""")
			oldpeak = float(st.slider('', 0.0, 10.0, 2.0))
			st.write("""**You selected this option **""",oldpeak)

			st.write("""**11. Slope (The slope of the peak exercise ST segment) :**""")
			slope = st.selectbox(" 1 = up, 2 = flat, 0 = down )",["1","2","0"])
			st.write("""**You selected this option **""",slope)
			


			data = {'age': age, 'sex': sex, 'cp': cp,'trestbps': trestbps, 'chol': chol, 'fastingBS': fastingBS, 'restingecg': restingecg, 'maxhr': maxhr, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope}
			features = pd.DataFrame(data, index=[0])

			return features

		df = user_input_features()
		st.subheader('Given Inputs : ')
		st.write(df)

    

    

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()