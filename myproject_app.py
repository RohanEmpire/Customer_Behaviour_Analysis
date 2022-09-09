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
    # Customer Behaviour Analysis Using Machine Learning
    ## 
   """)

    st.image("""https://cdn.wperp.com/uploads/2020/07/customer-behavior-analysis-A-guide-for-entrepreneurs-customer-behavior-analysis-%E2%80%93-1-1536x614.png""")
    st.header("This App Let You Know To Which Segement A Customer Belongs To")
    st.subheader("Fill The Details Below")

    INCOME = st.number_input("INCOME",1500,120000)
    #AGE = st.number_input("AGE",19,80)
    AGE = st.sidebar.slider('AGE', min_value=19, max_value=80, step=1)
    Month_Customer = st.number_input("Month_Customer",12,50)
    TotalSpendings = st.number_input("TotalSpendings",5,3000)
    Children = st.radio("Children",("0","1","2","3"))
    
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




