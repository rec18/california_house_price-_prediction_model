import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

# Title

col=[ 'MedInc' , 'HouseAge' , 'AveRooms' , 'AveBedrms' , 'Population' , 'AveOccup' ]

st.title("Calafornia Housing Price Prediction") #this give the title of our page

st.image("https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg") #this is our image show in main page

st.header("model of housing prices to predict median house values in California",divider=True) #its our header of the site, and divider is used to divide our header 

#st.subheader('''User Must Enter Values To Predict Price:
#[ 'MedInc' , 'HouseAge' , 'AveRooms' , 'AveBedrms' , 'Population' , 'AveOccup' ]''') #this is subheader

st.sidebar.title("Select House Feature 🏠") #this give the side bar title

st.sidebar.image("https://images.pexels.com/photos/259588/pexels-photo-259588.jpeg?cs=srgb&dl=landscape-sky-clouds-259588.jpg&fm=jpg") #this give an image in our side bar

# This give the min and max value of the house in our california.csv file
# and also read the data  
temp_df=pd.read_csv("California.csv")
all_Value=[]
random.seed(57) #this will the random value so, that it cannot be change 
for i in temp_df[col]:
    min_Value,max_value=temp_df[i].agg(['min','max'])
    
    var= st.sidebar.slider(f"Select {i} value", int(min_Value), int(max_value),random.randint(int(min_Value), int(max_value)))

    all_Value.append(var)

ss=StandardScaler()
ss.fit(temp_df[col])
final_value=ss.transform([all_Value])

with open("House_price_pred_ridge_model.pkl","rb")as f:
    chatgpt=pickle.load(f)

price=chatgpt.predict(final_value)[0]*1000000

st.write(pd.DataFrame(dict(zip(col,all_Value)),index=[1])) #zip is basically used for merge/add 2 data in a single data
progress_bar=st.progress(0)
placeholder=st.empty()
placeholder.subheader("Predicting Price !!")
place=st.empty()
place.image("https://i.gifer.com//7Fmb.gif",width=100)

if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body=f"Predicted Median House Price ${round(price,2)}"
    placeholder.empty()
    place.empty()
    st.success(body)
    
else:
    body="Incalid House feature"
    st.warning(body)