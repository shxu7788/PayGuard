import pandas as pd
import streamlit as st


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.compose import ColumnTransformer
import codecs
from sklearn.model_selection import train_test_split
import joblib
import hashlib
import re
from PIL import Image
from sklearn.metrics import roc_curve, auc

from sklearn import set_config
import seaborn as sns
import streamlit.components.v1 as components
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from managed_db import *
import time
import pickle
import base64


from streamlit_option_menu import option_menu
st.set_page_config(layout='wide')
hide_st_style= """
<style>
#MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)
def generated_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verified_hashes(password,hashed_text):
    if generated_hashes(password)==hashed_text:
        return hashed_text
    return False
def home():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
    <h1><center>Bank Fraud Detection</center></h1>
    """, unsafe_allow_html=True)
    
    st.image("images.jpg", use_column_width=True)
    st.write("Instructions for using the Web-App:")
    st.write("* Navigate to the Signup page and complete the process to become eligible for login.")
    st.write("* Once you have signed up, go to the Login page and enter your credentials. If your credentials are incorrect, you will not be able to access the app.")
    st.write("* After logging in, you will see four main pages: Home, Login and Signup. Click on the subpages in the Login tab in the following order: File Uploading, EDA, Model Building, Deploying models before tuning, Deploying models after tuning, Comparison of Algorithm's accuracy, Comparison of Algorithm's recall-scores, Comparison of Algorithm's precision-scores, Comparison of Algorithm's F1-scores, Comparison of Algorithm's AUC-scores, Gift")
    st.write("* It is crucial to check all checkboxes and buttons across submenus to ensure accurate results. If any of them are not activated, the results may not appear as expected. Please note that one checkbox result is interlinked to another checkbox result, so do not miss any checkbox while working with the application.")
    st.write("* If you have checked a checkbox on a page and visit that page again later, the checkbox will not be checked, but your previous checked results will be stored in the sessions separately. Do not worry about this, but make sure to check all checkboxes at least once.")

def login():
    sub_selected='File Uploading'
    username=st.sidebar.text_input('Username')
    password=st.sidebar.text_input('Password',type='password')
   

   
    if 'c01' not in st.session_state:
        st.session_state.c01=False
    c01_state=st.session_state.c01
    c01_state=st.sidebar.checkbox('Login',value=c01_state)
    
    if c01_state:
        
        create_usertable()
        hashed_pwsd=generated_hashes(password)
        result=login_user(username,verified_hashes(password,hashed_pwsd))
        
        if result:
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            #st.session_state.c01=c01_state

            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing {i+1}%")
                time.sleep(0.01) # Add a delay to simulate processing time

            status_text.text("Processing completed!")
            
        else:
            st.warning('Incorrect usernmae/password')
    
    if st.sidebar.button('Logout'):
    # set login state to false and clear credentials
        c01_state = False
        #st.session_state.c01 = c01_state
        st.sidebar.empty()
        st.empty()
        
        #st.session_state.checkbox_login = False
        #st.sidebar.empty()
        
        st.warning('Successfully logged out')
        
    if c01_state:

        sub_selected = option_menu(
                menu_title='Machine Learning Menu',
                options=['File Uploading','EDA','Model Building','Deploying models before tuning',
                'Deploying models after tuning',"Comparison of Algorithm's Accuracy","Comparison of Algorithm's Recall-Scores",
                "Comparison of Algorithm's Precision-Scores","Comparison of Algorithm's F1-Scores",
                "Comparison of Algorithm's AUC-Scores",'Gift','Prediction'],
                orientation='vertical',
                default_index=0,
            )
        if sub_selected=='File Uploading':
            file_upload()
        elif sub_selected=='EDA':
            eda()
            
            

        elif sub_selected == 'Model Building':
            model_building()
        elif sub_selected=='Deploying models before tuning':
            deploying_models_without_parameters()
        elif sub_selected=='Deploying models after tuning':
            deploying_models_with_parameters()
        elif sub_selected=="Comparison of Algorithm's Accuracy":
            accuracy()
        elif sub_selected=="Comparison of Algorithm's Recall-Scores":
            recall()
        elif sub_selected=="Comparison of Algorithm's Precision-Scores":
            precision()
        elif sub_selected=="Comparison of Algorithm's F1-Scores":
            f1_score()
        elif sub_selected=="Comparison of Algorithm's AUC-Scores":
            auc()
        elif sub_selected=='Gift':
            gift()
        elif sub_selected=='Prediction':
            prediction()
            # Add a logout button
            
            
        




    
    
def signup():
    new_username=st.text_input('User name')
    new_password=st.text_input('Password',type='password')
    confirm_password=st.text_input('Confirm Password',type='password')
    
    if new_password==confirm_password and new_password!='':
        st.success('Password Confirmed')
    else:
        st.warning('Passwords not the same' )
        
    if st.button('Submit'):
        create_usertable()
        hashed_new_password=generated_hashes(new_password)
        add_userdata(new_username,hashed_new_password)
        st.success('You are successfully created a new account')
        
        st.info('Login to get started')

def file_upload():

    if 'df' not in st.session_state:
        st.session_state.df=0
    if 'c1' not in st.session_state:
        st.session_state.c1=False
    c1_state=st.session_state.c1
    c1_state=st.checkbox('Uploading the file from root directory of app',value=c1_state)
        
        

        
    
    if c1_state:
           
        df=pd.read_csv('bank_fraud_dataset.csv',index_col='Unnamed: 0',usecols=lambda column: column != 'pdate')
        st.success('Successfully file was uploaded!')
        st.session_state.c1=c1_state
    if 'c3' not in st.session_state:
        st.session_state.c3=False
    c3_state=st.session_state.c3
    c3_state=st.checkbox('Click to see the data frame',value=c3_state)
    if c3_state:
        st.write(df)
        st.session_state.df=df
        st.session_state.c3=c3_state
    
                
        

    
def eda():
    if 'df' not in st.session_state:
        st.session_state.df=0
    df=st.session_state.df
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if c3 and c1:
        st.subheader('Exploratory Data Analysis')
        with open('EDA.html', 'r') as f:
            report = f.read()
        
        components.html(report, width=2000, height=1200, scrolling=True)

    else:
        st.warning('You missed the checkboxes in previous pages!')
def model_building():
    st.warning('We are using the same csv for model building')
    if 'df' not in st.session_state:
        st.session_state.df=0
    df=st.session_state.df
    
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if "preprocessing" not in st.session_state:
        st.session_state.preprocessing = 0
    if "xtrain" not in st.session_state:
        st.session_state.xtrain = 0
    if "xtest" not in st.session_state:
        st.session_state.xtest = 0
    if "ytrain" not in st.session_state:
        st.session_state.ytrain = 0
    if "ytest" not in st.session_state:
        st.session_state.ytest = 0
    
    
    
    if c1 and c3:
        df1=df.copy()
        df2=df.copy()
        if 'c4' not in st.session_state:
            st.session_state.c4=False
        c4_state=st.session_state.c4
        c4_state=st.checkbox('Checking the column names of the data frame',value=c4_state)
        if c4_state:
            
            st.write(df.columns)
            st.session_state.c4=c4_state
        if 'c5' not in st.session_state:
            st.session_state.c5=False
        c5_state=st.session_state.c5
        c5_state=st.checkbox('Checking the null values of the data frame',value=c5_state)  
        if c5_state:
            
            null_values=df.isna().sum().sum()
            if null_values==0:
                st.write("Your data set doesn't contain any null values")
            else:
                st.write('Null values by column wise:')
                st.write(df.isna().sum())
            st.session_state.c5=c5_state
        
        
        
        
        catcols=[i for i in df.select_dtypes('object')]
        numcols=[i for i in df.select_dtypes('number')]
        numcols1=numcols[1:]
        if 'c7' not in st.session_state:
            st.session_state.c7=False
        c7_state=st.session_state.c7
        c7_state=st.checkbox('Checking the categorical columns',value=c7_state)
        if c7_state:
            st.write(catcols)
            st.session_state.c7=c7_state
        if 'c8' not in st.session_state:
            st.session_state.c8=False
        
        c8_state=st.session_state.c8
        c8_state=st.checkbox('Checking the numerical columns',value=c8_state)
        if c8_state:
            st.write(numcols)
            st.session_state.c8=c8_state
        if 'c9' not in st.session_state:
            st.session_state.c9=False
        c9_state=st.session_state.c9
        c9_state=st.checkbox('Seperating the independent and dependent features',value=c9_state)
        if c9_state:
            x=df.drop('label',axis=1)
            y=df['label']
            dependent_features=y
            independent_features=x
            st.write('Dependent Features are as follows:')
            st.write(dependent_features)
            st.write('Independent Features are as follows:')
            st.write(independent_features)
            st.session_state.c9=c9_state
        if 'c10' not in st.session_state:
            st.session_state.c10=False
        c10_state=st.session_state.c10
        c10_state=st.checkbox('Splitting the data into training and testing',value=c10_state)
        if c10_state:
            st.info('We are using 20 percent of data for testing')
                        
                                    
                                    
            xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=50)
            st.write('Shape of xtrain {}'.format(xtrain.shape))
            st.write('Shape of xtest {}'.format(xtest.shape))
            st.write('Shape of ytrain {}'.format(ytrain.shape))
            st.write('Shape of ytest {}'.format(ytest.shape))
            st.session_state.xtrain=xtrain
            st.session_state.xtest=xtest
            st.session_state.ytrain=ytrain
            st.session_state.ytest=ytest
            st.session_state.c10=c10_state
        if 'c11' not in st.session_state:
            st.session_state.c11=False
        c11_state=st.session_state.c11
        c11_state=st.checkbox('Pipeline building for doing standard scaler on numerical columns',value=c11_state)
        if c11_state:
            numerical_cols=Pipeline(
                                        steps=[
                                        
                                        ('Scaler',StandardScaler()),
                                        ]
                                    )
            st.success('Successfully numerical pipleline is built!')
            st.session_state.c11=c11_state
        if 'c12' not in st.session_state:
            st.session_state.c12=False
        c12_state=st.session_state.c12
        c12_state=st.checkbox('Pipeline building for doing onehot encoding on categorical columns',value=c12_state)
        if c12_state:
            categorical_cols=Pipeline(
                                        steps=[
                                        
                                        ('Encoding',OneHotEncoder(handle_unknown='ignore')),
                                        ]
                                    )
            st.success('Successfully categorical pipleline is built!')
            st.session_state.c12=c12_state
        if 'c13' not in st.session_state:
            st.session_state.c13=False
        c13_state=st.session_state.c13
        c13_state=st.checkbox('Combining both transformers into single transformer using column tranformers',value=c13_state)
        if c13_state:
            preprocessing=ColumnTransformer(
                                    [
                                        ('categorical columns',categorical_cols,catcols),
                                        ('numerical columns',numerical_cols,numcols1),

                                        ]


                                    )
            st.session_state.preprocessing=preprocessing
            st.success('Column tranformers are built')
            st.session_state.c13=c13_state
       


        
        
        
        
        
        

        
        

    else:
        st.warning('You missed the checkboxes in previous pages')

   
def deploying_models_without_parameters():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13]):
        st.info('All the models will be deployed using automl module of python named Pycaret')
        if "ac" not in st.session_state:
            st.session_state.ac = 0
        if "f1score" not in st.session_state:
            st.session_state.f1score = 0
        if "recall_score" not in st.session_state:
            st.session_state.recall_score = 0
        if "precision_score" not in st.session_state:
            st.session_state.precision_score = 0
        if 'auc_score' not in st.session_state:
            st.session_state.auc_score=0
        
        st.info('We are building models without any hyper parameters')
        
        
        if 'c14' not in st.session_state:
            st.session_state.c14=False
        c14_state=st.session_state.c14
        c14_state=st.checkbox('Deploying models without tuning',value=c14_state)
        if c14_state:
            
            df=joblib.load('originaldf.pkl')
            #st.write(df)

        
            def conversion_originaldf(df):
                ac={}
                
                f1score={}
                
                recall_score={}
                
                precision_score={}
            
                auc_score={}
            
                for i,j in df.iterrows():
                    model_name=j['Model']
                    accuracy=j['Accuracy']*100
                    f1=j['F1']
                    recall=j['Recall']
                    precision=j['Prec.']
                    auc=j['AUC']
                    ac[model_name]=accuracy
                    f1score[model_name]=f1
                    recall_score[model_name]=recall
                    precision_score[model_name]=precision
                    auc_score[model_name]=auc
                return ac,f1score,recall_score,precision_score,auc_score
            ac,f1score,recall_score,precision_score,auc_score=conversion_originaldf(df)
            def deploying_models(df):
                for i,j in df.iterrows():
                    model_name=j['Model']
                    accuracy=j['Accuracy']*100
                    f1_score=j['F1']
                    recall_score=j['Recall']
                    precision_score=j['Prec.']
                    auc_score=j['AUC']
                    st.write(f'Model : {model_name}')
                    st.write(f'Accuracy : {accuracy}')
                    st.write(f'F1 Score : {f1_score}')
                    st.write(f'Recall Score : {recall_score}')
                    st.write(f'Precision Score : {precision_score}')
                    #st.write(f'AUC Score : {auc_score}')
                    st.write('---'*20)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing {i+1}%")
                time.sleep(0.1) # Add a delay to simulate processing time
            deploying_models(df)
            st.balloons()
        
        
            
            
                
                
            
            
            st.session_state.ac=ac
            st.session_state.f1score=f1score
            st.session_state.recall_score=recall_score
            st.session_state.precision_score=precision_score
            st.session_state.auc_score=auc_score
            st.session_state.c14=c14_state
        
        
        

        
        
    else:
        st.warning('You missed the checkboxes in previous pages!')




def deploying_models_with_parameters():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14]):
        if "ac1" not in st.session_state:
            st.session_state.ac1 = 0
        if "f1scores" not in st.session_state:
            st.session_state.f1scores = 0
        if "recall_scores" not in st.session_state:
            st.session_state.recall_scores = 0
        if "precision_scores" not in st.session_state:
            st.session_state.precision_scores = 0
        if 'auc_scores' not in st.session_state:
            st.session_state.auc_scores=0

        
        
        if 'c15' not in st.session_state:
            st.session_state.c15=False
        c15_state=st.session_state.c15
        c15_state=st.checkbox('Deploying models after tuning',value=c15_state)
        if c15_state:
            
            df1=joblib.load('tuneddf1.pkl')
            #st.write(df)

        
            def conversion_tuneddf(df1):
                ac1={}
                
                f1scores={}
                
                recall_scores={}
                
                precision_scores={}
            
                auc_scores={}
            
                for i,j in df1.iterrows():
                    model_name=j['Model']
                    accuracy=j['Accuracy']*100
                    f1=j['F1']
                    recall=j['Recall']
                    precision=j['Prec.']
                    auc=j['AUC']
                    ac1[model_name]=accuracy
                    f1scores[model_name]=f1
                    recall_scores[model_name]=recall
                    precision_scores[model_name]=precision
                    auc_scores[model_name]=auc
                return ac1,f1scores,recall_scores,precision_scores,auc_scores
            ac1,f1scores,recall_scores,precision_scores,auc_scores=conversion_tuneddf(df1)
            def deploying_models_after_tuning(df1):
                for i,j in df1.iterrows():
                    model_name=j['Model']
                    accuracy=j['Accuracy']*100
                    f1_score=j['F1']
                    recall_score=j['Recall']
                    precision_score=j['Prec.']
                    auc_score=j['AUC']
                    st.write(f'Model : {model_name}')
                    st.write(f'Accuracy : {accuracy}')
                    st.write(f'F1 Score : {f1_score}')
                    st.write(f'Recall Score : {recall_score}')
                    st.write(f'Precision Score : {precision_score}')
                    #st.write(f'AUC Score : {auc_score}')
                    st.write('---'*20)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing {i+1}%")
                time.sleep(0.1) # Add a delay to simulate processing time
            deploying_models_after_tuning(df1)
            st.balloons()
        
        
            
            
                
                
            
            
            st.session_state.ac1=ac1
            st.session_state.f1scores=f1scores
            st.session_state.recall_scores=recall_scores
            st.session_state.precision_scores=precision_scores
            st.session_state.auc_scores=auc_scores
            st.session_state.c15=c15_state
        
        
        

        
        
    else:
        st.warning('You missed the checkboxes in previous pages!')


def accuracy():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15]):
        if "ac" not in st.session_state:
            st.session_state.ac = 0
        ac=st.session_state.ac
        if "ac1" not in st.session_state:
            st.session_state.ac1 = 0
        ac1=st.session_state.ac1
        
        if 'c16' not in st.session_state:
            st.session_state.c16=False
        c16_state=st.session_state.c16
        c16_state=st.checkbox("Comparison of Algorithm's accuarcy",value=c16_state)
        if c16_state:
           
           
            # Accuracy scores before tuning
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16,8))

            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')
            ac_scores_before_tuning = [j for i, j in enumerate(ac.values())]
            ac_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                'Random Forest Classifier', 'Light Gradient Boosting Machine',
                'Gradient Boosting Classifier', 'Ada Boost Classifier',
                'Extra Trees Classifier', 'Extreme Gradient Boosting',
                'K Neighbors Classifier', 'Dummy Classifier',
                'Logistic Regression', 'Decision Tree Classifier',
                'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                'Naive Bayes']
            # Accuracy scores after tuning
            ac_scores_after_tuning = [j for i, j in enumerate(ac1.values())]

            ac1_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                'Random Forest Classifier', 'Light Gradient Boosting Machine',
                'Gradient Boosting Classifier', 'Ada Boost Classifier',
                'Extra Trees Classifier', 'Extreme Gradient Boosting',
                'K Neighbors Classifier', 'Dummy Classifier',
                'Logistic Regression', 'Decision Tree Classifier',
                'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                'Naive Bayes']

            # Set the positions of the bars on the x-axis
            pos = np.arange(len(ac_labels))

            # Set the width of the bars
            width = 0.25

            # Create a figure and axis object
            fig, ax = plt.subplots(figsize=(20, 10))

            # Plot the before tuning bars
            rects1 = ax.bar(pos - width/2, ac_scores_before_tuning, width, label='Before Tuning',color='red')

            # Plot the after tuning bars
            rects2 = ax.bar(pos + width/2, ac_scores_after_tuning, width, label='After Tuning',color='blue')

            # Set the x-axis labels and tick marks
            ax.set_xticks(pos)
            ax.set_xticklabels(ac_labels, rotation=45, ha='right',size=17,color='white')

            # Set the y-axis label and limit
            ax.set_ylabel('Accuracy Score',size=20,color='white')
            ax.set_ylim([0, 100])
            ax.tick_params(axis='y',labelsize=15,color='white')
            ax.tick_params(axis='y', colors='white')
            ax.yaxis.label.set_color('white')
            

            # Set the plot title and legend
            ax.set_title("Comparison of all algorithm's accuracy",size=20,color='white')
            ax.legend(fontsize=15)
            ax.spines['left'].set_color('white')
            # Add labels for the bar heights
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{:.2f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',rotation=90,size=15)

            # Add the bar height labels
            autolabel(rects1)
            autolabel(rects2)

            st.pyplot(fig)
            st.session_state.c16=c16_state

        
    else:
        st.warning('You missed the checkboxes in previous pages!')



def recall():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16]):
        if "recall_scores" not in st.session_state:
            st.session_state.recall_scores = 0
        recall_scores=st.session_state.recall_scores
        if "recall_score" not in st.session_state:
            st.session_state.recall_score = 0
        recall_score=st.session_state.recall_score
        if 'c17' not in st.session_state:
            st.session_state.c17=False

        c17_state=st.session_state.c17
        c17_state=st.checkbox("Comparison of Algorithm's recall-score",value=c17_state)
        if c17_state:
            ac1_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                        'Random Forest Classifier', 'Light Gradient Boosting Machine',
                        'Gradient Boosting Classifier', 'Ada Boost Classifier',
                        'Extra Trees Classifier', 'Extreme Gradient Boosting',
                        'K Neighbors Classifier', 'Dummy Classifier',
                        'Logistic Regression', 'Decision Tree Classifier',
                        'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                        'Naive Bayes']
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16,8))

            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')
            recall_before_tuning = [j for i, j in enumerate(recall_score.values())]
            recall_after_tuning = [j for i, j in enumerate(recall_scores.values())]
            ax.plot(recall_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax.plot(recall_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax.set_title('Comparison of Recall-Score Before and After Tuning',fontsize=20)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)

            ax.legend(fontsize=20)
            ax.grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
            
            ax.set_xticks(range(len(ac1_labels)))
            ax.set_xticklabels(ac1_labels, rotation=45, ha='right')
            st.pyplot(fig)
            st.session_state.c17=c17_state

        
    else:
        st.warning('You missed the checkboxes in previous pages!')


    
def precision():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if 'c17' not in st.session_state:
        st.session_state.c17=0
    c17=st.session_state.c17
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16,c17]):
        if "precision_score" not in st.session_state:
            st.session_state.precision_score = 0
        precision_score=st.session_state.precision_score
        if "precision_scores" not in st.session_state:
            st.session_state.precision_scores = 0
        precision_scores=st.session_state.precision_scores
        if 'c18' not in st.session_state:
            st.session_state.c18=False
        c18_state=st.session_state.c18
        c18_state=st.checkbox("Comparison of Algorithm's precision-score",value=c18_state)
        if c18_state:
            ac1_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                        'Random Forest Classifier', 'Light Gradient Boosting Machine',
                        'Gradient Boosting Classifier', 'Ada Boost Classifier',
                        'Extra Trees Classifier', 'Extreme Gradient Boosting',
                        'K Neighbors Classifier', 'Dummy Classifier',
                        'Logistic Regression', 'Decision Tree Classifier',
                        'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                        'Naive Bayes']
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16,8))

            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')
            precision_before_tuning = [j for i, j in enumerate(precision_score.values())]
            precision_after_tuning = [j for i, j in enumerate(precision_scores.values())]
            ax.plot(precision_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax.plot(precision_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax.set_title('Comparison of Precision-Score Before and After Tuning',fontsize=20)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)

            ax.legend(fontsize=20)
            ax.grid(color='white', linestyle='--', linewidth=1, alpha=0.3)

            ax.set_xticks(range(len(ac1_labels)))
            ax.set_xticklabels(ac1_labels, rotation=45, ha='right')
            st.pyplot(fig)
            st.session_state.c18=c18_state




        
    else:
        st.warning('You missed the checkboxes in previous pages!')


def f1_score():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if 'c17' not in st.session_state:
        st.session_state.c17=0
    c17=st.session_state.c17
    if 'c18' not in st.session_state:
        st.session_state.c18=0
    c18=st.session_state.c18
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16,c17,c18]):
        if "f1score" not in st.session_state:
            st.session_state.f1score = 0
        f1score=st.session_state.f1score
        if "f1scores" not in st.session_state:
            st.session_state.f1scores = 0
        f1scores=st.session_state.f1scores
        if 'c19' not in st.session_state:
            st.session_state.c19=False
        c19_state=st.session_state.c19
        c19_state=st.checkbox("Comparison of Algorithm's F1-score",value=c19_state)
        if c19_state:
            ac1_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                        'Random Forest Classifier', 'Light Gradient Boosting Machine',
                        'Gradient Boosting Classifier', 'Ada Boost Classifier',
                        'Extra Trees Classifier', 'Extreme Gradient Boosting',
                        'K Neighbors Classifier', 'Dummy Classifier',
                        'Logistic Regression', 'Decision Tree Classifier',
                        'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                        'Naive Bayes']
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16,8))

            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')
            f1_scores_before_tuning = [j for i, j in enumerate(f1score.values())]
            f1_scores_after_tuning = [j for i, j in enumerate(f1scores.values())]

            

            ax.plot(f1_scores_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax.plot(f1_scores_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax.set_title('Comparison of F1 Scores Before and After Tuning',fontsize=20)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)

            ax.legend(fontsize=20)
            ax.grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
            ax.set_xticks(range(len(ac1_labels)))
            ax.set_xticklabels(ac1_labels, rotation=45, ha='right')

            st.pyplot(fig)
            st.session_state.c19=c19_state
    else:
        st.warning('You missed the checkboxes in previous pages!')



        


    
def auc():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if 'c17' not in st.session_state:
        st.session_state.c17=0
    c17=st.session_state.c17
    if 'c18' not in st.session_state:
        st.session_state.c18=0
    c18=st.session_state.c18
    if 'c19' not in st.session_state:
        st.session_state.c19=0
    c19=st.session_state.c19
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16,c17,c18,c19]):
        if 'auc_score' not in st.session_state:
            st.session_state.auc_score=0
        auc_score=st.session_state.auc_score
        if 'auc_scores' not in st.session_state:
            st.session_state.auc_scores=0
        auc_scores=st.session_state.auc_scores
        if 'c20' not in st.session_state:
            st.session_state.c20=False
        c20_state=st.session_state.c20
        c20_state=st.checkbox("Comparison of Algorithm's AUC-score",value=c20_state)
        if c20_state:
            ac1_labels = ['Ridge Classifier', 'Linear Discriminant Analysis',
                        'Random Forest Classifier', 'Light Gradient Boosting Machine',
                        'Gradient Boosting Classifier', 'Ada Boost Classifier',
                        'Extra Trees Classifier', 'Extreme Gradient Boosting',
                        'K Neighbors Classifier', 'Dummy Classifier',
                        'Logistic Regression', 'Decision Tree Classifier',
                        'SVM - Linear Kernel', 'Quadratic Discriminant Analysis',
                        'Naive Bayes']
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16,8))

            fig.subplots_adjust(hspace=1.0)
            plt.rcParams['grid.color'] = 'black'
            sns.set_style('darkgrid')
            auc_before_tuning = [j for i, j in enumerate(auc_score.values())]
            auc_after_tuning = [j for i, j in enumerate(auc_scores.values())]
            ax.plot(auc_before_tuning, label='Before Tuning',linewidth=5,c='red')
            ax.plot(auc_after_tuning, label='After Tuning',linewidth=5,c='blue')
            ax.set_title('Comparison of AUC Score Before and After Tuning',fontsize=20)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)

            ax.legend(fontsize=20)
            ax.grid(color='white', linestyle='--', linewidth=1, alpha=0.3)
            ax.set_xticks(range(len(ac1_labels)))
            ax.set_xticklabels(ac1_labels, rotation=45, ha='right')

            st.pyplot(fig)
            st.session_state.c20=c20_state


        
    else:
        st.warning('You missed the checkboxes in previous pages!')


def gift():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if 'c17' not in st.session_state:
        st.session_state.c17=0
    c17=st.session_state.c17
    if 'c18' not in st.session_state:
        st.session_state.c18=0
    c18=st.session_state.c18
    if 'c19' not in st.session_state:
        st.session_state.c19=0
    c19=st.session_state.c19
    if 'c20' not in st.session_state:
        st.session_state.c20=0
    c20=st.session_state.c20
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16,c17,c18,c19]):

        new_df=joblib.load('originaldf.pkl')
        high=new_df['Model'][0]
        with st.spinner('Analyzing data...'):
            time.sleep(5)
                        
        st.markdown(
                            f"""
                            <style>
                            .algorithm-name {{
                                color: #0072C6;
                                font-weight: bold;
                                font-size: 24px;
                                margin: 0 0 10px 0;
                            }}
                            
                            .bomb {{
                                position: absolute;
                                top: 50%;
                                left: 50%;
                                animation: explode 0.5s ease-in-out 4s forwards;
                            }}
                            @keyframes explode {{
                                0% {{
                                    transform: scale(1);
                                    opacity: 1;
                                }}
                                100% {{
                                    transform: scale(10);
                                    opacity: 0;
                                }}
                            }}
                            
                            .animate__animated {{
                                animation-duration: 1s;
                                animation-fill-mode: both;
                            }}
                            
                            .animate__zoomIn {{
                                animation-name: zoomIn;
                            }}
                            
                            @keyframes zoomIn {{
                                from {{
                                    opacity: 0;
                                    transform: scale3d(0.3, 0.3, 0.3);
                                }}
                            
                                50% {{
                                    opacity: 1;
                                }}
                            
                                to {{
                                    transform: scale3d(1, 1, 1);
                                }}
                            }}
                            </style>
                            
                            <div class="animate__animated animate__zoomIn">
                                <h2>Awesome! You found the best algorithm.</h2>
                                <p>The highest accuracy score was gained by <span class="algorithm-name">{high}</span> algorithm.</p>
                            </div>
                            <div class="bomb"></div>
                            """,
                            unsafe_allow_html=True
                        )
        st.balloons()
    else:
        st.warning('You missed the checkboxes in previous pages!')

def prediction():
    if 'c3' not in st.session_state:
        st.session_state.c3=0
    c3=st.session_state.c3
    if 'c1' not in st.session_state:
        st.session_state.c1=0
    c1=st.session_state.c1
    if 'c4' not in st.session_state:
        st.session_state.c4=0
    c4=st.session_state.c4
    if 'c5' not in st.session_state:
        st.session_state.c5=0
    c5=st.session_state.c5
    
    if 'c7' not in st.session_state:
        st.session_state.c7=0
    c7=st.session_state.c7
    if 'c8' not in st.session_state:
        st.session_state.c8=0
    c8=st.session_state.c8
    if 'c9' not in st.session_state:
        st.session_state.c9=0
    c9=st.session_state.c9
    if 'c10' not in st.session_state:
        st.session_state.c10=0
    c10=st.session_state.c10
    if 'c11' not in st.session_state:
        st.session_state.c11=0
    c11=st.session_state.c11
    if 'c12' not in st.session_state:
        st.session_state.c12=0
    c12=st.session_state.c12
    if 'c13' not in st.session_state:
        st.session_state.c13=0
    c13=st.session_state.c13
    if 'c14' not in st.session_state:
        st.session_state.c14=0
    c14=st.session_state.c14
    if 'c15' not in st.session_state:
        st.session_state.c15=0
    c15=st.session_state.c15
    if 'c16' not in st.session_state:
        st.session_state.c16=0
    c16=st.session_state.c16
    if 'c17' not in st.session_state:
        st.session_state.c17=0
    c17=st.session_state.c17
    if 'c18' not in st.session_state:
        st.session_state.c18=0
    c18=st.session_state.c18
    if 'c19' not in st.session_state:
        st.session_state.c19=0
    c19=st.session_state.c19
    if 'c20' not in st.session_state:
        st.session_state.c20=0
    c20=st.session_state.c20
    if all([c1, c3, c4, c5, c7, c8, c9, c10, c11, c12, c13,c14,c15,c16,c17,c18,c19]):
        st.info('Prediction is based on the algorithm which has highest accuracy score')
        df3=joblib.load('ytest.pkl')
        df4=joblib.load('ypred.pkl')
        ytest_arr = np.array(df3['ytest'])
        ypred_arr = np.array(df4['ypred'])
        MissClassified = np.sum(ytest_arr != ypred_arr)
        df3=joblib.load('ytest.pkl')
        df4=joblib.load('ypred.pkl')
        ytest_arr = np.array(df3['ytest'])
        ypred_arr = np.array(df4['ypred'])
        results_df = pd.DataFrame({'Actual': ytest_arr, 'Predicted': ypred_arr})
        missclassified=np.sum(ytest_arr != ypred_arr)
        classified=np.sum(ytest_arr == ypred_arr)
        MissClassified = (np.sum(ytest_arr != ypred_arr) / len(df3['ytest'])) * 100
        Classified=(np.sum(ytest_arr == ypred_arr) / len(df3['ytest'])) * 100
        if st.checkbox('Prediction'):
            st.balloons()
            
            st.write(results_df)
                     
            st.write('The number of Classified samples are {} and MissClassified samples are {}'.format(classified,missclassified))
            st.write('The percentage of Classified samples are {}% and MissClassified samples are {}%'.format(Classified,MissClassified))
                     
                     
        
    else:
        st.warning('You missed the checkboxes in previous pages!')
        
    
        
    


selected = option_menu(
    menu_title='Main Menu',
    options=['Home', 'Login', 'SignUp'],
    default_index=0,
    menu_icon='house',
    orientation='horizontal'
)

if selected == 'Home':
    home()
    
    
elif selected == 'Login':
    login()
    
    
   
        
elif selected == 'SignUp':
    signup()
    




