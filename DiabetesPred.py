import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import base64



def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        base64_encoded = base64.b64encode(data).decode("utf-8")
    return base64_encoded



def set_background(png_file):
    bin_str = get_base64(png_file)  # You should define the get_base64 function
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Define your get_base64 function here (convert PNG file to base64 string)

# Call the set_background function with your PNG file
set_background("C:/Users/User/Desktop/Data Science/Streamlit/DiabetesPred/64t65j8r.png")

# Rest of your Streamlit app code
st.title("Diabetes Prediction")
# ... continue with the rest of your Streamlit app

# Load data
data = pd.read_csv("C:/Users/User/Desktop/Data Science/Streamlit/DiabetesPred/Diabetes.csv")

# Data exploration
st.write("Sample Data:")

D = st.sidebar.radio("Menu",["Home", "Prediction"])
if D == "Home":
    st.write(data.head())
    st.write("Data Shape:")
    st.write(data.shape)

    st.write("Data Summary:")
    st.write(data.describe())

    st.write("Correlation Matrix:")
    st.write(data.corr())


elif D == "Prediction":
    st.subheader("PREDICTION")
# Data preprocessing
    x = data.iloc[:, 0:8]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model training and evaluation
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    st.write("Predicted Values:")
    st.write(y_pred)

    st.write("Actual Values:")
    st.write(y_test)

    cf = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cf)

    st.write("Classification Report:")
    report = classification_report(y_test, y_pred)
    st.write(report) 

    h=open("classifier.pkl","wb")  
    pickle.dump(model,h)  
    h.close()  
    pregnancies = st.number_input("number of Pregnancies:",0,20,1)
    glucose=st.number_input("Glucose:",0,800,1)  
    bp=st.number_input("Blood Pressure:",0,200,1)  
    skin_thickness=st.number_input("Skin Thickness:",0,100,1)  
    insulin=st.number_input("Insulin:",30,200,30)  
    bmi=st.number_input("BMI:",20,80,20)  
    DBF=st.number_input("Diabetes Pedigree Function:",0.0,4.0,0.1)  
    AGE=st.number_input("Age:",0,120,1)  
    data=[np.array([pregnancies,glucose,bp,skin_thickness,insulin,bmi,DBF, AGE])] 

# = np.array(data).reshape(1,-1)
    prediction = model.predict(data)

    if st.button("Diabetes Prediction"):
        st.write(str(prediction))