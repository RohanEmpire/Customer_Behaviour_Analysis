#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from numpy import outer


#m = open("‪‪C:/Users/ROHAN'S/Downloads/Rohan_Project/customer.pkl","rb")

#model = joblib.load(m)



#model = joblib.load("‪C:\Users\ROHAN'S\Downloads\customer.pkl")
model = joblib.load('customer.pkl')

def web_app():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapercave.com/dwp2x/wp9160803.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

     
    st.write("""
    # CUSTOMER BEHAVIOUR ANALYSIS USING MACHINE LEARNING
    ## 
   """)

    st.image("""https://cdn.wperp.com/uploads/2020/07/customer-behavior-analysis-A-guide-for-entrepreneurs-customer-behavior-analysis-%E2%80%93-1-1536x614.png""")
    st.header("THIS APP LETS YOU KNOW TO WHICH SEGEMENT YOU BELONGS TO")
    st.subheader("FILL THE DETAILS BELOW")

    INCOME = st.number_input("INCOME",1500,120000)
    AGE = st.number_input("AGE",19,80)
    #AGE = st.sidebar.slider('AGE', min_value=19, max_value=80, step=1)
    Month_Customer = st.number_input("MONTHS",12,50)
    TotalSpendings = st.number_input("SPENDINGS",5,3000)
    Children = st.radio("CHIDREN",("0","1","2","3"))
    
    if st.button("PRESS HERE TO KNOW THE CATEGORY"):
        
        result = model.predict([[INCOME,AGE,Month_Customer,TotalSpendings,
                                Children]])
        if result == 0:
            result = "ALPHA"
        elif result == 1: 
            result = "BETA"
        elif result == 2: 
            result = "GAMMA"
        else : 
            result = "OMEGA"
        
        
        st.text_area(label='CATEGORY CUSTOMER BELONGS TO:- ',value=result , height= 100)
         
    
    
run = web_app()


# In[ ]:




