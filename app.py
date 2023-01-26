# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:36:08 2022

@author: Sergey
"""

import streamlit as st
from io import BytesIO
import pyperclip
from datetime import datetime
import pandas as pd
import re
from mellanni_modules import format_header
st.set_page_config(layout="wide")

bins = [0.4,0.7]
labels = ['low','med','high']
n_clusters = 5
cerebro_file, ba_file, magnet_file,file_ba_matched,file_ba_missed = None, None,None,None,None
example_asins = ['B08CZVWR21','B07N7KFHVH','B08N2RDBHT','B00HHLNRVE','B07M74PH8P']
asin_str = '(B[A-Z0-9]{9})'
cerebro_columns = ['Keyword Phrase', 'ABA Total Click Share', 'ABA Total Conv. Share',
       'ABA SFR', 'Keyword Sales', 'Cerebro IQ Score', 'Search Volume',
       'Search Volume Trend', 'H10 PPC Sugg. Bid', 'H10 PPC Sugg. Min Bid',
       'H10 PPC Sugg. Max Bid', 'Sponsored ASINs', 'Competing Products', 'CPR',
       'Title Density', 'Amazon Recommended', 'Sponsored', 'Organic',
       'Sponsored Rank (avg)', 'Sponsored Rank (count)',
       'Amazon Recommended Rank (avg)', 'Amazon Recommended Rank (count)',
       'Position (Rank)', 'Relative Rank', 'Competitor Rank (avg)',
       'Ranking Competitors (count)', 'Competitor Performance Score']

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

def lemmatize(file, column):
    import nltk
    # if nltk.download('all') == False:
    #     nltk.download('all')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    import re
    kw = file[column].values.tolist()
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in range(len(kw)):
        r = re.sub('[^a-zA-Z]', ' ', kw[i]).lower().split()
        r = [word for word in r if word not in stopwords.words('english')]
        r = [lemmatizer.lemmatize(word) for word in r]
        r = ' '.join(r)
        corpus.append(r)
    file['clean kw'] = corpus
    cv = CountVectorizer()
    vectors = cv.fit_transform(kw)
    word_freq = {}
    for text in corpus:
        words = text.split(' ')
        for word in words:
            if word in word_freq:
                word_freq[word] +=1
            else:
                word_freq[word] = 1
    word_freq = pd.DataFrame.from_dict(word_freq, orient = 'index').reset_index()
    word_freq.columns = ['word','frequency']
    word_freq = word_freq.sort_values('frequency', ascending = False)
    sums = {}
    top_words = {}
    for keyword in corpus:
        text = keyword.split(' ')
        score = sum(word_freq[word_freq['word'].isin(text)]['frequency'])
        sums[keyword] = score
        top_words[keyword] = ' '.join(word_freq[word_freq['word'].isin(text)].sort_values('frequency', ascending = False)['word'].values[:3])
    sums = pd.DataFrame.from_dict(sums, orient = 'index', columns = ['frequency score'])
    top_words = pd.DataFrame.from_dict(top_words, orient = 'index', columns = ['top_word(s)'])
    file = pd.merge(file, sums, left_on = 'clean kw', right_index = True)
    file = pd.merge(file, top_words, left_on = 'clean kw', right_index = True)
    file = file[[column,'frequency score','top_word(s)']]
    
    return file, word_freq, vectors

def clusterize(file,vectors,cols,num_clusters):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = num_clusters)
    if vectors is not None:
        model.fit(vectors)
        file['word similarity'] = model.labels_
    else:
        model.fit(file[cols])
        file['cluster'] = model.labels_
    return file

def visualize_clusters(df,columns,num_clusters):
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(projection = '3d')
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    legend = []
    values = [columns[2],columns[1],columns[0]]
    for n in range(num_clusters):
        clustered_df = df[df['cluster'] == n]
        # sns.scatterplot(x = clustered_df[cols[0]],y= clustered_df[cols[1]], color=colors[n-1],linewidth = 0)
        ax.scatter(xs = clustered_df[values[0]],
                   ys = clustered_df[values[1]],
                   zs = clustered_df[values[2]],
                   s = 40,depthshade = True,
                   color=colors[n-1],linewidth = 0)
        legend.append(n)
    ax.set_ylim(max(df[columns[1]]),0)
    ax.set_xlabel(values[0])
    ax.set_ylabel(values[1])
    ax.set_zlabel(values[2])
    ax.view_init(5, -70)
    ax.legend(legend)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),'clusters.png'))
    plt.close()
    return None

def process_file(asins,cerebro,ba,magnet,n_clusters,bins, file_ba_matched = file_ba_matched, file_ba_missed = file_ba_missed):
    bin_labels = [str(int(x*100))+'%' for x in bins]

    file = cerebro.copy()

    # if 'Position (Rank)' in file.columns.tolist():
    #     file[asins[0]] = file['Position (Rank)'].copy()
    #     del file['Position (Rank)']
    
    stat_columns = ['Keyword Phrase','ABA Total Click Share','H10 PPC Sugg. Bid','Keyword Sales','Search Volume','CPR','Ranking Competitors (count)']
    asin_columns = asins.copy()
    r = len(stat_columns)
    all_columns = stat_columns+asin_columns
    file = file[all_columns]
    file = file[file['Search Volume'] != '-']
    file['Search Volume'] = file['Search Volume'].astype(int)
    file['Keyword Sales'] = file['Keyword Sales'].replace('-',0)
    file['Keyword Sales'] = file['Keyword Sales'].astype(int)
    file = file.sort_values('Search Volume', ascending = False)
    file = file.replace('>306',306).replace('N/R',306).replace('-',306)#.replace(0,306)
    file.iloc[:,r:] = file.iloc[:,r:].astype(int)
    file['Top30'] = round(file.iloc[:,r:].isin(range(1,31)).sum(axis = 1)/len(asins),2)
    file['Top10'] = round(file.iloc[:,r:-1].isin(range(1,11)).sum(axis = 1)/len(asins),2)
    file['KW conversion'] = round(file['Keyword Sales'] / file['Search Volume'] * 100,2)
    file['Sales normalized'] = (file['Keyword Sales']-file['Keyword Sales'].min())/(file['Keyword Sales'].max()-file['Keyword Sales'].min())
    file['Conversion normalized'] = (file['KW conversion']-file['KW conversion'].min())/(file['KW conversion'].max()-file['KW conversion'].min())
    file['SV normalized'] = (file['Search Volume']-file['Search Volume'].min())/(file['Search Volume'].max()-file['Search Volume'].min())
    file["Sergey's score"] = round(
        file['Conversion normalized']*1.1+file['Sales normalized']*.8+file['Top30']*2+file['Top10'],
        3)
    file["Sergey's score"] = round(((file["Sergey's score"]-file["Sergey's score"].min())
                              / (file["Sergey's score"].max()-file["Sergey's score"].min()))*100,1)
    file = file.sort_values(["Sergey's score"],ascending = False)
    search_terms = file['Keyword Phrase'].drop_duplicates().tolist()

# define alpha-asins
    sums = []
    percs = []
    for a in asin_columns:
        n = file.loc[(file[a] < 30)]['Keyword Sales'].sum()
        sums.append(n)

    for a in sums:
        p = a/sum(sums)
        percs.append(p)

    # sums = [int(float(x.replace(',',''))) for x in sums]
    sums_db = pd.DataFrame([sums,percs], columns = asin_columns, index = ['Keyword Sales','% share by sales'])
    sums_db.loc['% share by sales'] = round(sums_db.loc['% share by sales'].astype(float)*100,1)
    
    # get Brand Analytics file results
    if isinstance(ba,pd.core.frame.DataFrame):
        # try:
        #     file_ba = pd.read_csv(ba, skiprows=1)
        # except:
        #     file_ba = pd.read_excel(ba, skiprows=1)
        file_ba = ba.copy()
            
        file_ba = file_ba.drop('Department', axis = 1)
        file_ba_missed = file_ba[~file_ba['Search Term'].isin(search_terms)]
        file_ba_matched = file_ba[file_ba['Search Term'].isin(search_terms)]
        sv = file.copy()
        sv = sv[['Keyword Phrase','Keyword Sales']]
        sv['Search Term'] = sv['Keyword Phrase'].copy()
        sv = sv.drop('Keyword Phrase', axis = 1)
        file_ba_matched = pd.merge(file_ba_matched,sv,on = 'Search Term', how = 'left')
        
    #apply boolean conditions to sales,conversion and relevance
    #alternative way using pandas cut
    file['sales'] = pd.cut(file['Sales normalized'],
        bins = [-1,
            file['Sales normalized'].describe(percentiles = bins)[bin_labels[0]],
            file['Sales normalized'].describe(percentiles = bins)[bin_labels[1]],
            1],
        labels = ['low','med','high']
        )
    file['conversion'] = pd.cut(file['Conversion normalized'],
        bins = [-1,
            file['Conversion normalized'].describe(percentiles = bins)[bin_labels[0]],
            file['Conversion normalized'].describe(percentiles = bins)[bin_labels[1]],
            1],
        labels = ['low','med','high']
        )

    file['competition'] = pd.cut(file['Top30'],bins = 3,labels = labels)

    sales_cols = pd.get_dummies(file['sales'],prefix = 'sales')
    conversion_cols = pd.get_dummies(file['conversion'],prefix = 'conversion')
    competition_cols = pd.get_dummies(file['competition'],prefix = 'competition')
    file = pd.concat([file,sales_cols,conversion_cols,competition_cols], axis = 1)
    clusterize_columns = sales_cols.columns.tolist()+conversion_cols.columns.tolist()+competition_cols.columns.tolist()
    normalized_columns = ['Sales normalized','Conversion normalized','SV normalized']
    # feed the file to KMeans model to clusterize
    

    file = clusterize(file,vectors = None,cols = clusterize_columns,num_clusters = n_clusters)
    # visualize_clusters(file,columns,n_clusters)
    file = file.drop(clusterize_columns, axis = 1)
    file = file.drop(normalized_columns, axis = 1)

    
    top_kws = file['Keyword Phrase'].head(10).tolist()
    cerebro_kws = file['Keyword Phrase'].unique()
    
    magnet_words.text_area('Magnet keyword research', value = "\n".join(top_kws), height = 250)
    if isinstance(magnet,pd.core.frame.DataFrame):
        magnet = magnet[~magnet['Keyword Phrase'].isin(cerebro_kws)]
            
        magnet = magnet[magnet['Search Volume'] != '-']
        magnet['Search Volume'] = magnet['Search Volume'].str.replace(',','').astype(int)
        magnet['Keyword Sales'] = magnet['Keyword Sales'].replace('-',0).replace(',','')
        magnet['Keyword Sales'] = magnet['Keyword Sales'].astype(int)
        magnet['KW conversion'] = round(magnet['Keyword Sales'] / magnet['Search Volume'] * 100,2)
        magnet = magnet.sort_values('KW conversion', ascending = False)
        magnet_cols = magnet.columns.tolist()
        file_cols = file.columns.tolist()
        drop_cols = list(set(magnet_cols) - set(file_cols))
        magnet = magnet.drop(drop_cols,axis = 1)
        file = pd.concat([file,magnet],axis = 0)

    ## add colors from dictionary
    # dictionary = pd.read_excel(d_path,usecols = ['Color','Color Map']).dropna()
    # dictionary = dictionary.applymap(str.lower)
    # colors = dictionary['Color'].dropna().unique().tolist()+dictionary['Color Map'].dropna().unique().tolist()
    # color_kws = file[file['Keyword Phrase'].str.contains(('|').join(colors))]

    #add word counts and frequency scores
    lemm, word_freq, vectors = lemmatize(file, 'Keyword Phrase')
    file = pd.merge(file, lemm, how = 'left', on = 'Keyword Phrase')
    file = clusterize(file,vectors,cols = None,num_clusters=8)
    return file, sums_db, file_ba_matched,file_ba_missed, word_freq, asins

st.title('Keyword processing tool')
asins_area, magnet_words, alpha_asin = st.columns(3)
# asins = asins_area.text_area('Input ASINs. Make sure they are the same ASINs that are included in your Cerebro file').split('\n')
link = '[Goto Cerebro](https://members.helium10.com/cerebro?accountId=268)'
st.markdown(link, unsafe_allow_html=True)
if st.button('Load sample ASINs'):
    asins = asins_area.text_area('Input ASINs. Make sure they are the same ASINs that are included in your Cerebro file','\n'.join(example_asins)).split('\n')
# asins = [x for x in asins if x != '']
# st.write(asins)

with st.expander('Upload files'):
    if st.checkbox('Add Cerebro file (mandatory), .csv or .xlsx supported'):
        cerebro_file = st.file_uploader('Select Cerebro file')
    if cerebro_file:
        if '.csv' in cerebro_file.name:
            cerebro = pd.read_csv(cerebro_file).fillna(0)
        elif '.xlsx' in cerebro_file.name:
            cerebro = pd.read_excel(cerebro_file).fillna(0)
        if all([x in cerebro.columns for x in cerebro_columns]):
            asins = [re.findall(asin_str, x) for x in cerebro.columns]
            asins = ['Position (Rank)'] + [x[0] for x in asins if x != []]
            asins_area.text_area('ASINs in Cerebro file:','\n'.join(asins), height = 250)
            st.write(f'Uploaded successfully, file contains {len(cerebro)} rows')
        else:
            st.warning('This is not a Cerebro file!')
            st.stop()

    if st.checkbox('Add Brand Analytics file (optional), .csv or .xlsx supported'):
        ba_file = st.file_uploader('Select Brand Analytics file')
    if ba_file:
        if '.csv' in ba_file.name:
            ba = pd.read_csv(ba_file, skiprows = 1)
        elif '.xlsx' in ba_file.name:
            ba = pd.read_excel(ba_file, skiprows = 1)
        st.write(f'Uploaded successfully, file contains {len(ba)} rows')
    else:
        ba = ''

    if st.checkbox('Add Magnet file (optional), .csv or .xlsx supported'):
        magnet_file = st.file_uploader('Select Magnet file')
    if magnet_file:
        if '.csv' in ba_file.name:
            magnet = pd.read_csv(magnet_file)
        elif '.xlsx' in ba_file.name:
            magnet = pd.read_excel(magnet_file)
        st.write(f'Uploaded successfully, file contains {len(magnet)} rows')
    else:
        magnet = ''

if st.button('Process keywords'):
    file, sums_db, file_ba_matched,file_ba_missed, word_freq,asins = process_file(asins,cerebro,ba,magnet,n_clusters,bins)
    # asins_area.text_area('Input ASINs. Make sure they are the same ASINs that are included in your Cerebro file','\n'.join(example_asins)).split('\n')
    alpha_asin.bar_chart(sums_db.T['% share by sales'])
    # alpha_asin.text_area('Alpha ASINs', sums_db)
    st.write('Cerebro results',file)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        file.to_excel(writer, sheet_name = 'Keywords', index = False)
        workbook = writer.book
        worksheet = writer.sheets['Keywords']
        format_header(file,writer,'Keywords')
        max_row, max_col = file.shape
        worksheet.conditional_format(
            1,max_col-1,max_row,max_col-1,
            {'type': '3_color_scale','max_color':'red','min_color':'green'})
        try:
            magnet_row = len(file)-len(magnet)
            bg = workbook.add_format({'bg_color': 'red'})
            worksheet.conditional_format(
                magnet_row+1,0,max_row,0,{'type':'no_blanks','format':bg})
        except:
            pass
        
        sums_db.to_excel(writer, sheet_name = 'Alpha ASIN', index = False)
        format_header(sums_db, writer, 'Alpha ASIN')
        worksheet = writer.sheets['Alpha ASIN']
        worksheet.conditional_format(
            2,0,2,max_col-1,
            {'type': '3_color_scale','max_color':'red','min_color':'green'})
        try:
            file_ba_matched.to_excel(writer, sheet_name = 'BA_match', index = False)
            format_header(file_ba_matched, writer, 'BA_match')
            file_ba_missed.to_excel(writer, sheet_name = 'BA_missed', index = False)
            format_header(file_ba_missed, writer, 'BA_missed')
        except:
            pass
        try:
            color_kws.to_excel(writer, sheet_name = 'Color KWs', index = False)
            worksheet = writer.sheets['Color KWs']
            format_header(color_kws,writer,'Color KWs')
            max_row, max_col = color_kws.shape
            worksheet.conditional_format(
                1,max_col-1,max_row,max_col-1,
                {'type': '3_color_scale','max_color':'red','min_color':'green'})
        except:
            pass
        word_freq.to_excel(writer, sheet_name = 'word_frequency', index = False)
        format_header(word_freq, writer, 'word_frequency')    
    
    st.download_button('Download results',output.getvalue(), file_name = 'test.xlsx')

# date1,date2 = st.slider(
#     "Select date range",
#     min_value = datetime(2020,1,1), max_value = datetime(2023,1,1),
#     value=(datetime(2021,1,1),datetime(2022,1,1)),
#     format="MM/DD/YY", 
#     )
# st.write("Start time:", date1.strftime("%Y-%m-%d"),' - ', date2.strftime("%Y-%m-%d"))
