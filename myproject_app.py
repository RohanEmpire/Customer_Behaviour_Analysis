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

    st.write("""
    # Customer Behaviour Analysis with Machine Learning
    ## This app predicts to which category a customer belongs too
   """)

    st.image("""https://cdn.wperp.com/uploads/2020/07/customer-behavior-analysis-A-guide-for-entrepreneurs-customer-behavior-analysis-%E2%80%93-1-1536x614.png""")
    st.header("User Details")
    st.subheader("Kindly Enter The following Details in order to make a prediction")

    INCOME = st.number_input("INCOME",1500,120000)
    AGE = st.number_input("AGE",19,80)
    Month_Customer = st.number_input("Month_Customer",12,50)
    TotalSpendings = st.number_input("TotalSpendings",5,3000)
    Children = st.number_input("Children",0,3)
    
    if st.button("Press here to make Prediction"):
        
        result = model.predict([[INCOME,AGE,Month_Customer,TotalSpendings,
                                Children]])
        if result == 0:
            result = "CATEGORY_D"
        elif result == 1: 
            result = "CATEGORY_A"
        elif result == 2: 
            result = "CATEGORY_B"
        else : 
            result = "CATEGORY_C"
        
        
        st.text_area(label='Category belongs to:- ',value=result , height= 100)
         
    
    
run = web_app()


# In[ ]:




