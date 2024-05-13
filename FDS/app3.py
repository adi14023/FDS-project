import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

model = pickle.load(open('svm.pkl', 'rb'))
encoder = pickle.load(open('oh_encoder.pkl', 'rb')) 



  
def main(): 
    st.title("Animal Condition Predictor Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Animal Condition Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    aname = st.selectbox("AnimalName",['Dog','Cat','Buffaloes','Rabbit','Cow','Chicken'])
    sym1  = st.selectbox("symptoms1",["Fever","loss in weight","loss of appetite","swollen","Dejection","Neck paralysis","Facial swelling","Weakness"])
    sym2  = st.selectbox("symptoms2",["Diarrhea","Difficulty in Breathing","Respiratory distress","High moratality","Anemia"])
    sym3  = st.selectbox("symptoms3",["Congestion","Vomiting","Respiratory distress","Seizuers","Dehydration","ruffled feathers"])
    sym4  = st.selectbox("symptoms4",["Weight loss","Depression","Vomiting","Lethargy","Stiffness","Labored breathing"])
    sym5  = st.selectbox("symptoms5",["Pain","loss of appetite","Vomiting","Dehydration","Weakness","Anorexia"])
    
    if st.button("Predict"): 
        
        features = [[aname,sym1,sym2,sym3,sym4,sym5]]
        data = {'AnimalName':aname,'symptoms1':sym1,'symptoms2':sym2,'symptoms3':sym3,'symptoms4':sym4,'symptoms5':sym5}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['AnimalName','symptoms1','symptoms2','symptoms3','symptoms4','symptoms5'])
        df['AnimalName']=df.AnimalName.str.capitalize()

        s = (df.dtypes == 'object')
        object_cols = list(s[s].index)
        print(object_cols)

        new_data = pd.DataFrame(encoder.transform(df[object_cols]))
        # One-hot encoding removed index; put it back
        new_data.index = df.index


        # Remove categorical columns (will replace with one-hot encoding)

        num_data = df.drop(object_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        new_data = pd.concat([new_data,num_data], axis=1)

        # Ensure all columns have string type
        new_data.columns = new_data.columns.astype(str)

        df=new_data

        X= df

        prediction = model.predict(X)
    
        output = int(prediction[0])
        if output == 1:
          text = "Dangerous"
        else:
            text = "Not Dangerous"

        st.success('Animal Condition  is {}'.format(text))
      
if __name__=='__main__': 
    main()