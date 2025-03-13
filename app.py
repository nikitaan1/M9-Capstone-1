import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


df1 = pd.read_csv('data/student_performance_data.csv')
df1.columns = df1.columns.str.lower() 
df1['gender'] = df1['gender'].replace({'Male': 0, 'Female': 1})
df1 = df1.replace({'Yes': 1, 'No': 0}) 


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_transform = ohe.fit_transform(df1[['parental_education_level']])
ohe_df = pd.DataFrame(ohe_transform, columns=ohe.get_feature_names_out(['parental_education_level']))
df1 = pd.concat([df1, ohe_df], axis=1).drop(columns=['parental_education_level'])

st.set_page_config(page_title='Student Exam Prediction', page_icon='📚', layout='wide')

## Sidebar
page = st.sidebar.radio('Select a Page', ['Home', 'Model Training and Evaluation Page'])

if page == 'Home':
    ## Title
    st.title('Student Exam Performance Analysis')

    ## Welcome Message
    st.write('''This app provides an interactive platform to explore the student performance dataset.
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.''')

    ## Import Dataset
    df = pd.read_csv('data/student_performance_data.csv')
    df.columns = df.columns.str.lower() 

    with st.expander('Data Preview'):
        st.dataframe(df)

    ## Visualizations
    fig1 = px.bar(df, x='gender', title='Gender Distribution')
    st.plotly_chart(fig1)

    fig2 = px.bar(df, x='internet_access_at_home', title='Internet Access at Home')
    st.plotly_chart(fig2)

    fig3 = px.bar(df, x='extracurricular_activities', title='Extracurricular Activities')
    st.plotly_chart(fig3) 

    st.subheader("Interactive Plot")

    def interactive_plot(dataframe):
        x_axis = st.selectbox('Select X-Axis Value', options=['gender', 'parental_education_level', 'internet_access_at_home', 'extracurricular_activities'])
        y_axis = st.selectbox('Select Y-Axis Value', options=['study_hours_per_week', 'attendance_rate', 'past_exam_score', 'final_exam_score'])

        if x_axis and y_axis:
            fig4 = px.box(dataframe, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig4)

    interactive_plot(df)


elif page == 'Model Training and Evaluation Page':
    st.title("Exam Performance Prediction")

    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    X = df1.drop(columns=['pass_fail', 'student_id', 'final_exam_score'])
    y = df1['pass_fail']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    sc = StandardScaler()

    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)    

    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    model.fit(X_train_sc, y_train)

    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_sc, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_sc, y_test):.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_sc, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

