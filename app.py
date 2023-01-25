# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:36:08 2022

@author: Sergey
"""

import streamlit as st
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
    st.write('Love you')
    