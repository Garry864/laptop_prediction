import pickle
import streamlit as st
import numpy as np


rf = pickle.load(open('rf1_28July.sav','rb'))
xgb = pickle.load(open('xgb1_28July.sav','rb'))


st.title('Laptop Price Prediction Web App')

st.header('Fill the details to generate Laptop price Prediction')


options = st.sidebar.selectbox('Select ML model',['RF_Reg','XGB_Reg'])


company = st.selectbox('Company',['Dell','Lenovo','HP','Asus','Acer','MSI',
                                  'Other','Toshiba','Apple'])
typename = st.selectbox('TypeName',['Notebook','Gaming','Ultrabook','2in1',
                                   'Workstation','Netbook'])
ram = st.selectbox('RAM',[2,4,6,8,12,16,24,32,64])
weight = st.slider('Weight',min_value=0.7,max_value=4.2,step=0.1)
touchscreen = st.selectbox('TouchScreen',['Yes','No'])
ips = st.selectbox('IPS',['Yes','No']) 
cpu = st.selectbox('CPU',['I7','I5','I3','AMD','Other'])
hdd = st.selectbox('HDD',[0,128,500,1000,2000])
ssd = st.selectbox('SSD',[0,8,16,32,64,128,180,256,512,768,1000])
gpu = st.selectbox('GPU',['Intel','Nvidia','AMD']) 
os = st.selectbox('OS',['Win','Linux/Other','MAC'])



if company == "Dell":
    company =3
elif company == "Lenovo":
    company = 5
elif company == "HP":
    company = 4
elif company == "ASUS":
    company = 2
elif company == "Acer":
    company = 0
elif company == "MSI":
    company = 6
elif company == "Other":
    company = 7
elif company == "Other":
    company = 8
else:  # Apple
     company = 1



typename_var = {"Notebook":3,'Gaming':1,'Ultrabook':4,'2in1':0,'Workstation':5,
            'Netbook':2}


    
