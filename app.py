# Importing required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle

# Setting up page configuration and directory path
st.set_page_config(page_title="Titanic Survival App", page_icon="🛳️", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Setting background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('images/titanic_background.avif')

# Setting up logo
left1, left2, mid,right1, right2 = st.columns(5)
with mid:
    st.image("images/titanic_ship.jpeg", use_column_width=True)

# Setting up Sidebar
social_acc = ['Data Field Description', 'EDA', 'About App']
social_acc_nav = st.sidebar.radio('**INFORMATION SECTION**', social_acc)

if social_acc_nav == 'Data Field Description':
    st.sidebar.markdown("<h2 style='text-align: center;'> Data Field Description </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown("""
The table below gives a description on the variables required to make predictions.
| Variable      | Definition:       | Key   |
| :------------ |:--------------- |:-----|
| pclass        | Ticket Class     |1st / 2nd / 3rd|
| sex           | sex of passenger |male / female|
| Age           | Age of passenger |Enter age of passenger|
| Fare          | Passenger fare   |Enter Fare of passenger|
| Embarked      | Port of Embarkation|C=Cherbourg/ Q=Queenstown/ S=Southampton|
| IsAlone       | Whether passenger has relative(s) onboard or not|No = Passenger has relatives on board/ Yes = Passenger is Alone|
""")
    
elif social_acc_nav == 'EDA':
    st.sidebar.markdown("<h2 style='text-align: center;'> Exploratory Data Analysis </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("""
                        | About EDA|
                        | :------------ |
                        The exploratory data analysis of this project can be find in a Jupyter notebook from the linl below""" )
    st.sidebar.markdown("[Open Notebook](https://github.com/Kyei-frank/Titanic-Project---Machine-Learning-from-Disaster/blob/main/workflow.ipynb)")

elif social_acc_nav == 'About App':
    st.sidebar.markdown("<h2 style='text-align: center;'> Titanic Survival Prediction App </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("""
                        | Brief Introduction|
                        | :------------ |
                        On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This App uses classification model to predict whether a passenger survives or not based on the data of the passenger.""")
    st.sidebar.markdown("")
    st.sidebar.markdown("[ Visit Github Repository for more information](https://github.com/Kyei-frank/Titanic-Project---Machine-Learning-from-Disaster)")

# Loading Machine Learning Objects
@st.cache()
def load_saved_objects(file_path = 'ml_components'):
    # Function to load saved objects
    with open('ml_components', 'rb') as file:
        loaded_object = pickle.load(file)
        
    return loaded_object

# Instantiating ML_items
Loaded_object = load_saved_objects(file_path = 'ml_components')
pipeline_of_my_app = Loaded_object["pipeline"]


# Setting up variables for input data
@st.cache()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            Pclass=[],
            Sex=[],
            Age=[],
            Fare=[],
            Embarked=[],
            IsAlone=[],
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

# setting Title for forms
st.markdown("<h2 style='text-align: center;'> Titanic Survival Prediction </h2> ", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'> Fill in the details below and click on SUBMIT button to make a prediction on the Survival of a passenger. </h7> ", unsafe_allow_html=True)

# Creating columns for for input data(forms)
left_col, right_col = st.columns(2)

# Developing forms to collect input data
with st.form(key="information", clear_on_submit=True):
    
    # Setting up input data for 1st column
    left_col.markdown("**CATEGORICAL DATA**")
    Pclass = left_col.selectbox("Passenger Class:", options = ["1st", "2nd", "3rd"])
    Sex = left_col.selectbox("Gender of Passenger:", options= ["male", "female"])
    IsAlone = left_col.selectbox("Does passenger has relative onboard?", options= ["Yes", "No"])
    Embarked = left_col.radio("Port of Embarkation:", options= ["C", "Q", "S"])
    
    # Setting up input data for 2nd column
    right_col.markdown("**NUMERICAL DATA**")
    Age = right_col.number_input("Enter Age of Passenger")
    Fare = right_col.number_input("Enter Fare of passenger")
    
    submitted = st.form_submit_button(label="Submit")
    
# Setting up background operations after submitting forms
if submitted:
    # Saving input data as csv after submission
    pd.read_csv(tmp_df_file).append(
        dict(
            Pclass= Pclass,
            Sex= Sex,
            Age= Age,
            Fare= Fare,
            Embarked= Embarked,
            IsAlone= IsAlone,
            ),
            ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    st.balloons()

    # Converting input data to a dataframe for prediction
    df = pd.read_csv(tmp_df_file)
    df= df.copy()
    
    # Making Predictions
    # Passing data to pipeline to make prediction
    pred_output = pipeline_of_my_app.predict(df)
    prob_output = np.max(pipeline_of_my_app.predict_proba(df))
    
    # Interpleting prediction output for display
    X= pred_output[-1]
    if X == 1:
        explanation = 'Passenger Survived'
    else: 
        explanation = 'Passenger did not Survive'
    output = explanation
    
    # Displaying prediction results
    st.markdown('''---''')
    st.markdown("<h4 style='text-align: center;'> Prediction Results </h4> ", unsafe_allow_html=True)
    st.success(f"Prediction: {output}")
    st.success(f"Confidence Probability: {prob_output}")
    st.markdown('''---''')    

    # Making expander to view all records
    expander = st.expander("See all records")
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['Survival']= pred_output
        st.dataframe(df)

    
    