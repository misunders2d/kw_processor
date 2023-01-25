# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:36:08 2022

@author: Sergey
"""

import streamlit as st
from datetime import datetime
import pandas as pd
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

st.title('Keyword processing tool')
asins = st.text_area('Input ASINs').split('\n')
cerebro_file = st.file_uploader('Select Cerebro file')
if cerebro_file:
    cerebro = pd.read_csv(cerebro_file)
    st.write(f'Uploaded successfully, file contains {len(cerebro)} rows')
    asin_list = ', '.join(asins) + ' '
    st.write(asin_list)
    st.write('Done')
date1,date2 = st.slider(
    "Select date range",
    min_value = datetime(2020,1,1), max_value = datetime(2023,1,1),
    value=(datetime(2021,1,1),datetime(2022,1,1)),
    format="MM/DD/YY", 
    )
st.write("Start time:", date1.strftime("%Y-%m-%d"),' - ', date2.strftime("%Y-%m-%d"))
    