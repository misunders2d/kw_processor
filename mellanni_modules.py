# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:40:00 2022

@author: djoha
"""
import os
try:
    import pandas as pd
except:
    os.system('pip install pandas openpyxl xlsxwriter')
    import pandas as pd

def kw_ranking_weighted(positions = None, sales = None):
    '''
    Generate a weighted position (by sales) for our KW ranking standings

    Returns
    -------
    (int) Number of our position.

    '''
    import numpy as np
    if all([positions is None, sales is None]):
        positions = input('Input list of positions\n').split('\n')
        sales = input('Input keyword sales\n').split('\n')
    positions = [int(x.replace('>','').replace('#Н/Д','306')) for x in positions]
    sales = [int(x.replace('-','0').replace(',','').replace('#Н/Д','0')) for x in sales]
    if len(positions) == len(sales):
        result = int(np.dot(sales,positions) / sum(sales))
        return result
    else:
        return 0


def bash_quote(dump = False,lang = 'ru'):
    wpath = get_db_path('US')[9]
    link = "http://bash.org/?random" if lang == 'en' else "http://bashorg.org/random"
    search = 'qt' if lang == 'en' else 'quote'
    bash_file = os.path.join(wpath,'bash') if lang == 'en' else os.path.join(wpath,'bash_ru')
    import random
    import pickle
    def get_jokes():
        from bs4 import BeautifulSoup as bs
        import requests
        page = requests.get(link)
        soup = bs(page.content, features = 'lxml')
        jokes = soup.find_all(class_ = search)
        jokes = [joke.get_text('<br/>').replace('<br/>','\n') for joke in jokes] if lang == 'ru' else [joke.get_text() for joke in jokes]
        return jokes
    def read_bash():
        with open(bash_file,'rb') as f:
            check = pickle.load(f)
        return check
    def write_bash(text):
        with open(bash_file,'wb') as f:
            pickle.dump(text,f)
        return None
        
    if os.path.isfile(bash_file):
        if dump == False:
            with open(bash_file,'rb') as f:
                jokes = pickle.load(f)
            joke = jokes[random.randint(0,len(jokes)-1)]
            return joke
        if dump == True:
            jokes = get_jokes()
            check = read_bash()
            for j in jokes:
                if j not in check:
                    check.append(j)
            write_bash(check)
            joke = check[random.randint(0,len(check)-1)]
            return joke
    elif not os.path.isfile(bash_file):
        jokes = get_jokes()
        with open(bash_file,'wb') as f:
            pickle.dump(jokes,f)
        joke = jokes[random.randint(0,len(jokes)-1)]
        return joke

def cancelled_shipments(account = 'US'):
    paths = get_db_path('US')
    path = os.path.join(paths[7],'Cancelled shipments')
    d_path = get_db_path(account)[1]
    d = pd.read_excel(d_path, usecols = ['ASIN','SKU'])
    file = get_file_paths([path])[-1]
    print(f'Latest cancelled shipments file is: {os.path.basename(file)}')
    cancelled = pd.read_excel(file)
    cancelled = pd.merge(cancelled, d, how = 'left', left_on = 'Sku', right_on = 'SKU').dropna(subset = ['ASIN'])
    # if account == 'CA':
    #     cancelled = cancelled[cancelled['Comments'] == 'Canada']
    cancelled = cancelled.pivot_table(
        values = 'Units to Cancel',
        index = 'ASIN',
        aggfunc = 'sum'
        ).reset_index()
    return cancelled

def fba_shipments(account = 'US'):
    import PySimpleGUI as sg
    paths = get_db_path('US')[7]
    downloads = os.path.join(paths,r'Current FBA Shipments\for combining')
    target = os.path.join(paths,r'Current FBA Shipments')
    files = get_file_paths([downloads])
    if len(files) == 0:
        os.startfile(downloads)
        answer = sg.PopupYesNo('No files to combine, continue?')
        if answer == 'Yes':
            return None
        elif answer == 'No':
            fba_shipments(account)
            return None
    combined = pd.DataFrame()
    for f in files:
        temp = pd.read_csv(f)
        combined = pd.concat([combined,temp])
        try:
            os.remove(f)
            print('File removed')
        except:
            print('File NOT removed')
    file_name = f'FBA Shimpents_{pd.to_datetime("today").strftime("%Y-%m-%d")}.csv'
    combined.to_csv(os.path.join(target,file_name),index = False)
    os.startfile(target)
    return None

def password_generator(x):
    '''
    Generates a password of 'x' lenght from letters, digits and punctuation marks

    Parameters
    ----------
    x : int
        number of symbols to use.

    Returns
    str
    password
    '''
    import string
    import random
    text = string.ascii_letters+string.digits+string.punctuation
    password = ''.join([random.choice(text) for x in range(x)])
    return password

def get_db_path(account = 'US'):
    '''
    check for user's system language and return correct paths.
    return correct dictionary for specified account
    '''
    #check system locale###############################
    import locale
    import ctypes
    windll = ctypes.windll.kernel32
    windll.GetUserDefaultUILanguage()
    sys_language = locale.windows_locale[ windll.GetUserDefaultUILanguage() ]
    current_user = os.path.expanduser('~')
    print(f'System language: {sys_language}\nCurrent user: {current_user}')
    

    if sys_language == 'ru_RU':
        prefix = r'G:\Общие диски'
    elif sys_language == 'uk_UA':
        prefix = r'G:\Спільні диски'
    elif sys_language == 'en_US':
        prefix = r'G:\Shared Drives'
    elif sys_language == 'en_GB':
        prefix = r'G:\Shared Drives'
    path = r'30 Sales\30.1 MELLANNI\30.11 AMAZON'
    matts_path = os.path.join(prefix,path,r'30.111 US\Inventory\Matts files')
    script_path = os.path.join(prefix,r'70 Data & Technology\70.03 Scripts')
    if account == 'US':
        db_path = os.path.join(prefix, path,r'30.111 US\Sales')
        dictionary = os.path.join(db_path, r'Dictionary.xlsx')
        db = os.path.join(db_path, r'sales.db')
        ppc = os.path.join(db_path,r'Ad_report\PPC_report.db')
        report_path = os.path.join(prefix, path,r'30.111 US\Daily reports')
        inventory_path = os.path.join(prefix, path,r'30.111 US\Inventory')
    elif account == 'CA':
        db_path = os.path.join(prefix, path,r'30.112 CA\Sales')
        dictionary = os.path.join(db_path, r'Dictionary_CA.xlsx')
        db = os.path.join(db_path, r'sales_Canada.db')
        ppc = os.path.join(db_path,r'Ad_report\canada_db_ppc.db')
        report_path = os.path.join(prefix, path,r'30.112 CA\Daily reports')
        inventory_path = os.path.join(prefix, path,r'30.112 CA\Inventory')
    elif account == 'MP':
        path = r'30 Sales\30.2 MELLANNI_HOME\30.21 AMAZON'
        db_path = os.path.join(prefix,path, 'Sales')
        dictionary = os.path.join(db_path,'Dictionary - Mellanni Home.xlsx')
        db = os.path.join(db_path, 'sales_MellPlus.db')
        ppc = os.path.join(db_path,'Ad_report\mellplus_db_ppc.db')
        report_path = os.path.join(prefix, path,'Daily reports')
        inventory_path = os.path.join(prefix, path,'Inventory')
    else:
        print('no account specified')
        return None
    print(f'DB path: {db_path}\nAccount: {account}')
    return db_path, dictionary, db, prefix, ppc, current_user,report_path, inventory_path,matts_path,script_path


def min_max_scale(df,cols):
    from sklearn.preprocessing import minmax_scale
    for c in cols:
        df[f'{c}_scaled'] = minmax_scale(df[c])
    return df

def feature_selection(df,model,target):
    from sklearn.feature_selection import RFECV
    df = df.select_dtypes('number').dropna()
    all_X = df.drop(target, axis = 1)
    selector = RFECV(model, cv = 10)
    selector = selector.fit(all_X,df[target])
    cols = all_X.columns[selector.support_]
    return cols

# def sales_forecast():
def add_trailing_average(df):
    df['day7'] = (df['Units Ordered'].shift(1)).rolling(window = 7).mean()
    df['day14'] = (df['Units Ordered'].shift(1)).rolling(window = 14).mean()
    df['day30'] = (df['Units Ordered'].shift(1)).rolling(window = 30).mean()
    df['std_7'] = (df['day7'].shift(1)).rolling(7).std()
    df['std_30'] = (df['day30'].shift(1)).rolling(30).std()
    return df

def add_date_columns(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        pass
    df['Day of month'] = df['Date'].dt.day
    df['Day of week'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    return df

def create_collection_sales():
    import sqlite3
    import pyodbc
    paths = get_db_path('US')
    
    work_path = paths[0]
    d_path = paths[1]
    db2 = paths[2]
    db1 = os.path.join(work_path,'Archive\sales2019.db')
    
    start_date = '2015-01-01'
    start_date = str(start_date)
    conn = sqlite3.connect(db1)
    sql_query = '''
    SELECT
        Date,"(Child) ASIN" as ASIN, SKU,
        "Units Ordered", "Units Ordered - B2B", "Ordered Product Sales"
    FROM Sales
    WHERE Date >= '''+f'"{start_date}"'+';'
    sales1 = pd.read_sql(sql_query, conn)#read sales
    conn.close()
    
    conn = sqlite3.connect(db2)
    sql_query = '''
    SELECT
        Date,"(Child) ASIN" as ASIN, SKU,
        "Units Ordered", "Units Ordered - B2B", "Ordered Product Sales"
    FROM Sales
    WHERE Date >= '''+f'"{start_date}"'+';'
    sales2 = pd.read_sql(sql_query, conn)#read sales
    conn.close()
    
    sales = pd.concat([sales1,sales2],axis = 0)
    
    dictionary = pd.read_excel(d_path, usecols = ['SKU', 'ASIN','Collection','Actuality'])
    active = dictionary[dictionary['Actuality'] == 'Active']
    inactive = dictionary[dictionary['Actuality'] == 'Inactive']
    discontinued = dictionary[dictionary['Actuality'] == 'Discontinued']
    d = pd.concat([active,inactive,discontinued],axis = 0)
    d = d.drop_duplicates('ASIN',keep = 'first')
    
    sales = pd.merge(sales,d, how = 'left', on = 'ASIN')
    sales['Date'] = pd.to_datetime(sales['Date'])
    pivot = sales.pivot_table(values = 'Units Ordered', index = ['Date','Collection'], aggfunc = 'sum').reset_index()
    
    #check changes
    conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+os.path.join(work_path,'SKU_changelog.accdb')+';')
    
    SQL_Query = pd.read_sql_query(
    '''SELECT * FROM "SKU changes"''', conn)
    sku_change = pd.DataFrame(SQL_Query)
    
    conn.close()
    
    del sku_change['Collection']
    
    sku_change = pd.merge(sku_change,d, how = 'left', on = 'ASIN')
    sku_change = sku_change.drop_duplicates(['Date','Collection','Change Type'])
    sku_change = sku_change[sku_change['Change Type'] == 'LD']
    sku_change = sku_change.dropna()
    sku_change = sku_change.sort_values('Date')
    sku_change['LD'] = True
    sku_change = sku_change[['Date','Collection','LD']]
    
    pivot = pd.merge(pivot, sku_change, how = 'left', on = ['Date','Collection'])
    pivot['LD'] = pivot['LD'].fillna(False)
    
    pivot = add_date_columns(pivot)
    
    # pivot = add_trailing_average(pivot)
    
    pivot.to_csv(os.path.join(work_path,'Collection_sales.csv'), index = False)
    return None

def create_forecast_new(out = False):
    '''
    create a 180-day forecast for most important collections
    '''
    from PySimpleGUI import PopupGetFile
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings('ignore')
    paths = get_db_path('US')
    work_path = paths[0]
    #load pivot from csv
    pivot = pd.read_csv(os.path.join(work_path,'Collection_sales.csv'),parse_dates=(['Date']))
    latest_date = pivot['Date'].max()+ pd.to_timedelta(1,'days')
    pivot['Date'] = pd.to_datetime(pivot['Date'])
    dates = pd.date_range(latest_date,latest_date+pd.to_timedelta(180,'days'))
    
    previous_forecast = PopupGetFile('Select previous forecast file',initial_folder=work_path,file_types = (('CSV Files,','*.csv'),))
    predecessor = pd.read_csv(previous_forecast,parse_dates=['Date'])
    predecessor = predecessor[
        (predecessor['Date']<latest_date)&
        (predecessor['Date']>(latest_date - pd.Timedelta(days = 30)))]
    collection = sorted(pivot['Collection'].unique().tolist())
    
    result = predecessor.copy()#pd.DataFrame()
    for c in collection:
        df = pivot[pivot['Collection'] == c]
        first_date = pd.to_datetime(df.sort_values('Date')['Date'].values[0]).date()
        last_date = pd.to_datetime(df.sort_values('Date')['Date'].values[-1]).date()
        df_dates = pd.DataFrame(pd.date_range(first_date,last_date),columns = ['Date'])
        df = pd.merge(df_dates,df, how = 'left', on = 'Date')
        df['Collection'] = c
        df['Units Ordered'] = df['Units Ordered'].fillna(0)
        df = add_date_columns(df)
        df['LD'] = df['LD'].fillna(False)
        if True in df['LD'].unique():
            ld = True
        else:
            ld = False
        if len(df) < 180:
            pass
        else:
            print(f'Forecasting {c}')
            df = add_trailing_average(df)
            # df = min_max_scale(
            #     df,[
            #         'Day of month','Day of week', 'Month', 'day7', 'day14', 'day30', 'std_7', 'std_30'
            #         ])
            lr = LinearRegression()
            train = df.dropna()
            print('Selecting features')
            features = feature_selection(train,lr,'Units Ordered').tolist()
            if 'LD' not in features and ld:
                features.append('LD')
            print(f'Best features selected:\n{[x for x in features]}')
            training_dates = pd.date_range(last_date,latest_date+pd.to_timedelta(180,'days'))
            for i,date in enumerate(training_dates):
                df = add_trailing_average(df)
                # df = min_max_scale(
                #     df,[
                #         'Day of month','Day of week', 'Month', 'day7', 'day14', 'day30', 'std_7', 'std_30'
                #         ])
                last_row = df.iloc[-1].values
                forecast = pd.DataFrame([last_row],columns = df.columns)
                forecast['Date'] = date + pd.Timedelta(days = 1)
                forecast = add_date_columns(forecast)
                forecast['Collection'] = c
                forecast['LD'] = False

                # dummies = pd.get_dummies(train['LD'],prefix = 'LD')
                # train = pd.concat([train,dummies], axis = 1)
                if ld:
                    if i % 7 == 0:
                        forecast['LD'] = True
            
            
                lr = LinearRegression()
                train = df.dropna()

                lr.fit(train[features],train['Units Ordered'])
                predictions = lr.predict(forecast[features])
                forecast['Units Ordered'] = int(predictions)
                df = pd.concat([df,forecast], axis = 0)
                forecast = forecast[['Date','Units Ordered','Collection','LD']]
                result = pd.concat([result,forecast],axis = 0)
    if out == True:
        with pd.ExcelWriter(os.path.join(r'c:\temp\pics','forecast.xlsx'), engine = 'xlsxwriter') as writer:
            result.to_excel(writer,sheet_name = 'Forecast', index = False)
            format_header(result,writer,'Forecast')
        result.to_csv(os.path.join(work_path,'Forecast.csv'), index = False)
        os.startfile(work_path)
        return None
    return result
    
    # return None
    
    

def model_assessment(df,features,labels):
    '''
    assess the best performing model for predictions
    on a specific dataset
    
    Args:
        df (DataFrame)
        features (set of 2-d arrays)
        labels: 2-d array of target labels
    
    Returns:
        list of dictionaries with best assessments
    '''
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression    
    from sklearn.ensemble import RandomForestClassifier

    all_X = df[features]
    all_y = df[labels]
    models = [{
        'name':'LogisticRegression',
        'estimator':LogisticRegression(),
        'hyperparameters':
        {
            'solver':["newton-cg", "lbfgs", "liblinear"]
        }},{
        'name':'KNeighborsClassifier',
        'estimator':KNeighborsClassifier(),
        'hyperparameters':
        {
            "n_neighbors":range(1,20,2),
            'weights':['distance','uniform'],
            'algorithm':['ball_tree','kd_tree','brute'],
            'p':[1,2]
        }},{
        'name':'RandomForestClassifier',
        'estimator':RandomForestClassifier(),
        'hyperparameters':
        {
            'n_estimators':[4,6,9],
            'criterion':['entropy','gini'],
            'max_depth':[2,5,10],
            'max_features':['log2','sqrt'],
            'min_samples_leaf':[1,5,8],
            'min_samples_split':[2,3,5]
        }}
    ]
    for m in models:
        print(m['name'])
        print('-'*len(m['name']))
        grid = GridSearchCV(m['estimator'],param_grid = m['hyperparameters'],cv = 10)
        grid.fit(all_X,all_y)
        m['best_params'] = grid.best_params_
        m['best_score'] = grid.best_score_
        m['best_model'] = grid.best_estimator_
        print(f"Best score: {m['best_score']}")
        print(f"Best parameters: {m['best_params']}")
        print('\n')    
    return models

def convert_to_pacific(db,columns):
    import pytz
    pacific = pytz.timezone('US/Pacific')
    db['pacific-date'] = pd.to_datetime(db[columns]).dt.tz_convert(pacific)
    db['pacific-date'] = pd.to_datetime(db['pacific-date']).dt.tz_localize(None)
    return db['pacific-date']

def format_header(df,writer,sheet):
    workbook  = writer.book
    cell_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'center', 'font_size':9})
    worksheet = writer.sheets[sheet]
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, cell_format)
    max_row, max_col = df.shape
    worksheet.autofilter(0, 0, max_row, max_col - 1)
    worksheet.freeze_panes(1,0)
    return None

def format_columns(df,writer,sheet,col_num):
    worksheet = writer.sheets[sheet]
    if not isinstance(col_num,list):
        col_num = [col_num]
    else:
        pass
    for c in col_num:
        width = max(df.iloc[:,c].astype(str).map(len).max(),len(df.iloc[:,c].name))
        worksheet.set_column(c,c,width)
    return None

def restock_limits(account,prefix):
    from datetime import datetime as dt
    import PySimpleGUI as sg
    import pyperclip
    import re
    if account == 'US':
        source_folder = os.path.join(prefix,'Restock limits')
        link = 'https://sellercentral.amazon.com/inventory-performance/dashboard'
    elif account == 'CA':
        source_folder = os.path.join(prefix,'Restock limits')
        link = 'https://sellercentral.amazon.ca/inventory-performance/dashboard'
    else:
        return None
    def process_restock():    
        source = pyperclip.waitForPaste().replace(',','')
        data = re.split('\r\n |\r\n|\r|\n|\t',source)
        restock = pd.DataFrame(data).dropna()
    
        restock[0] = pd.to_numeric(restock[0], errors = 'ignore')
    
        date = pd.to_datetime(dt.today().strftime('%m/%d/%y %H:%M'))
        s_used = int(restock.iloc[4].values[0])
        s_max = int(restock.iloc[6].values[0])
        s_perc = round(s_used / s_max * 100, 1)
        s_left = s_max - s_used
        o_used = int(restock.iloc[13].values[0])
        o_max = int(restock.iloc[15].values[0])
        o_perc = round(o_used / o_max * 100, 1)
        o_left = o_max - o_used
        limits = pd.read_csv(os.path.join(source_folder,'data\data.csv'))
        temp = pd.DataFrame([[date, s_used, s_max, s_perc, s_left, o_used, o_max, o_perc, o_left]], columns = limits.columns.tolist())
        limits = pd.concat([limits, temp])
        limits.to_csv(os.path.join(source_folder,'data\data.csv'), index = False)
        os.startfile(source_folder)
    
    def main():
        layout = [
            [sg.Text(
                'First, copy the restock limits\nfrom Seller Central\nUse this "Copy Link" button\nto copy the link to clipboard'
            ),sg.Button('Copy link', size = (8,5))],
            [sg.Text('Then, click on "Process" button')],
            [sg.Button('Process'), sg.Button('Cancel')]
        ]
    
        window = sg.Window('Restock Limits', layout)
        while True:
            event, values = window.read()
    
            if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                break
            elif event == 'Copy link':
                pyperclip.copy(link)
            elif event == 'Process':
                try:
                    process_restock()
                except:
                    sg.Popup('Something went wrong')
        window.close()
    main()
    

def install_modules():
    import os
    os.system('pip install --upgrade PySimpleGUI')
    os.system('pip install --upgrade pyautogui')
    os.system('pip install --upgrade pyperclip')
    os.system('pip install --upgrade keepa')
    os.system('pip install --upgrade pandas')
    os.system('pip install --upgrade seaborn')
    os.system('pip install --upgrade openpyxl')
    os.system('pip install --upgrade pyodbc')
    os.system('pip install --upgrade opencv-python')
    os.system('pip install --upgrade shutil')
    os.system('pip install --upgrade pyperclip')
    os.system('pip install --upgrade arrow')
    os.system('pip install --upgrade xlsxwriter')
    return None

def account_list():
    '''
    main list of existing accounts and marketplaces
    '''
    markets = ['US','CA','MP']
    accounts = ['MellPlus admin', 'MellPlus user', 'Mellanni admin',
              'Mellanni user', 'Canada admin', 'Canada user']
    return markets, accounts

def get_file_paths(folders):
    '''
    create full paths for all included files
    in the list of directories (submit as a list)
    '''
    file_list = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f != 'desktop.ini':
                    file = os.path.join(root,f)
                    file_list.append(file)
    return sorted(file_list)

def read_csv_files(file_list, filter_column = None,
                   filter_value = None, columns = None, drop_duplicates = False,
                   delimiter = None, encoding = None, parse_dates = None):
    '''
    read the files from the file list to one dataframe;
    use selected columns if necessary;
    use filter by column if necessary
    '''
    df = pd.DataFrame()
    for f in file_list:
        print(f'reading {f}')
        try:
            file = pd.read_csv(f, delimiter = delimiter, usecols = columns, low_memory=False)
        except:
            file = pd.read_excel(f, usecols = columns)
        if filter_column == None:
            pass
        else:
            file = file[file[filter_column].str.lower().isin([x.lower() for x in filter_value])]
        if drop_duplicates == True:
            file = file.drop_duplicates()
        df = df.append(file)
    return df

def process_batch_result(header = 3, chars = 70):
    '''
    new batch upload results files are extremely uncomprehensible,
    this helps group them to make more obvious
    '''
    import PySimpleGUI as sg
    import textwrap
    file = sg.PopupGetFile('Select the results file')
    result = pd.read_csv(file, sep = '\t', header = header)
    # result['error-type'] = result['error-message'].apply(
    #     lambda x: '\n'.join(textwrap.wrap(x, width = chars)))

    # result['error-type'] = result['error-message'].str[:chars]
    result = result.groupby(['error-code','error-message'])['sku'].apply(lambda x: ', '.join(x))
    out_path = sg.PopupGetFolder('Select output folder')
    writer = pd.ExcelWriter(os.path.join(out_path,'result.xlsx'), engine = 'xlsxwriter')
    workbook = writer.book
    col_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'center', 'font_size':10})
    result.to_excel(writer, sheet_name = 'Results')
    worksheet = writer.sheets['Results']
    worksheet.set_column(1, 1, 78, col_format)
    writer.save()
    os.startfile(out_path)
    return None
    

def matt(matts_path):
    file_list = []
    folder_path = matts_path
    file_list = get_file_paths([folder_path])
    file_list.sort()
    for f in file_list:
        if '~' in f:
            file_list.remove(f)
    file_path = file_list[-1]
    
    print('Latest file is',os.path.basename(file_path))
    
    matt = pd.read_excel(file_path, skiprows = 3)
    start = matt.columns.get_loc('INCOMING ORDERS')
    end = matt.columns.get_loc('Forcasted Sales')
    #add incoming inventory
    incoming2 = matt[2:].copy()
    # incoming.columns = incoming.iloc[0]
    # start = incoming.columns.get_loc('INCOMING ORDERS')
    # end = incoming.columns.get_loc('Forcasted Sales')
    incoming2.columns = matt.iloc[1].tolist()
    asin = incoming2['ASIN']
    incoming2 = incoming2.iloc[:,start:end-1]
    # incoming2.columns = incoming2.iloc[2]
    # incoming2 = incoming2.drop([0,1,2,2,3])
    incoming2 = incoming2.astype(int)
    incoming2['Total containers'] = incoming2.sum(axis = 1, numeric_only = True)
    incoming2['ASIN'] = asin
    
    matt.columns = matt.iloc[1].values.tolist()#.reset_index()
    matt = matt[2:]
    matt = matt[['ASIN', 'WH', 'Unprocessed', 'Master SKU', 'Stock, months', 'AVG, months']]
    matt.rename(columns = {'WH':'WH Processed', 'Unprocessed':'WH Unprocessed', 'Master SKU':'SKU'}, inplace = True) 
    return matt, incoming2

def manage_fba_inv(report_path,skus):
    from PySimpleGUI import PopupGetFile
    if os.path.isfile(os.path.join(report_path,'Manage FBA Inventory.csv')):
        file = os.path.join(report_path,'Manage FBA Inventory.csv')
    else:
        file = PopupGetFile('Select Manage FBA Inventory file')
    try:
        FBA_inv = pd.read_csv(file, encoding = 'CP1252')
    except:
        FBA_inv = pd.read_csv(file, encoding = 'utf-8')
    FBA_inv = FBA_inv[FBA_inv['sku'].isin(skus)]
    FBA_inv_trim = FBA_inv[['sku', 'asin', 'afn-fulfillable-quantity', 
                           'afn-inbound-working-quantity', 'afn-inbound-shipped-quantity',
                            'afn-inbound-receiving-quantity','afn-future-supply-buyable', 'afn-reserved-quantity',
                            'afn-researching-quantity']]
    FBA_inv_trim.rename(columns={'sku':'SKU','asin':'ASIN', 'afn-fulfillable-quantity':'FBA inventory', 
                                          'afn-inbound-working-quantity':'Inbound - Working',
                                 'afn-inbound-shipped-quantity':'Inbound - Shipped',
                                 'afn-inbound-receiving-quantity':'Inbound - Receiving',
                                'afn-future-supply-buyable':'Backordered', 
                                         'afn-reserved-quantity':'Reserved',
                                         'afn-researching-quantity':'Researching'}, inplace=True)
    FBA_inv_trim = FBA_inv_trim.pivot_table(values = ['Inbound - Working','Inbound - Shipped',
                                                      'Inbound - Receiving','Reserved','Researching'],
                                            index = 'ASIN', aggfunc = sum).reset_index() 
    return FBA_inv_trim
        

def create_dataset(dict_path, db, start_date, account = 'US',google = False):
    '''
    create a dataset from sales
    and inventory databases. Requires paths to dictionary
    and database files, along with the start date to filter
    
    Args:
        dictionary (str): path to dictionary file.
        db (str): path to db file.
        start_date (str): first date to filter database off.
        account (str): account to use.
        
    Returns:
        (tuple): dataset, dictionary, skus
    '''
    start_date = str(start_date)
    import time
    start = time.time()
    if google == False:
        import sqlite3
        print('Accessing sales database, please wait...')
    
        conn = sqlite3.connect(db)
        sql_query = '''
        SELECT
            Date,"(Child) ASIN" as ASIN, SKU, "Sessions - Total",
            "Units Ordered", "Units Ordered - B2B", "Ordered Product Sales"
        FROM Sales
        WHERE Date >= '''+f'"{start_date}"'+';'
        sales = pd.read_sql(sql_query, conn)#read sales
        sql_query = '''
        SELECT
            Date,"seller-sku" as SKU, asin as ASIN, price,"Quantity Available"
        FROM Inventory
        WHERE Date >= '''+f'"{start_date}"'+';'
        inventory = pd.read_sql(sql_query, conn)#read inventory
        conn.close()
    
    elif google == True:
        sales_cols = [
            'date','childAsin', 'sku','sessions','unitsOrdered',
            'unitsOrderedB2B','orderedProductSales']
        sales_cols_ren = [
            'Date','ASIN','SKU', 'Sessions - Total',
            'Units Ordered', 'Units Ordered - B2B', 'Ordered Product Sales']
        
        inv_cols = [
            'Date', 'seller_sku', 'asin', 'price','Quantity_Available']
        inv_cols_ren = [
            'Date','SKU','ASIN', 'price','Quantity Available']
        
        e_date = time.strftime("%Y-%m-%d")
        print('Reading data from GCloud')
        import gcloud_modules as gc
        client = gc.gcloud_connect()
        sales = gc.gcloud_daterange(client, start_date, e_date, 'business_report',sales_cols)
        sales = sales.rename(columns = dict(zip(sales_cols,sales_cols_ren)))
        inventory = gc.gcloud_daterange(client, start_date, e_date, 'inventory')
        inventory = inventory.rename(columns = dict(zip(inv_cols, inv_cols_ren)))
    
    print('Reading dictionary')
    dictionary_full = pd.read_excel(dict_path, usecols = ['SKU','ASIN','Collection',
                                                           'Sub-collection','Size', 'Color', 'Actuality',
                                                           'Top Collection','Size Map','Life stage'])
    if account == 'US':
        market = 'US'
    elif account == 'CA':
        market = 'CA'
    dictionary_full['Marketplace'] = market
    skus = dictionary_full['SKU'].unique()
    d = dictionary_full[['SKU','Marketplace']]
    sales = pd.merge(sales,d, how = 'left', on = 'SKU')
    inventory = pd.merge(inventory,d, how = 'left', on = 'SKU')
    sales = sales[sales['Marketplace'] == market]
    inventory = inventory[inventory['Marketplace'] == market]
    asins = dictionary_full[['ASIN','SKU']]
    asins_sku = asins.groupby('ASIN')['SKU'].apply(", ".join).reset_index()
    dictionary_full = dictionary_full.drop('SKU', axis = 1).drop_duplicates('ASIN')
    dictionary = pd.merge(dictionary_full, asins_sku, how = 'right', on = 'ASIN')

    ###################################################################################
    # clean and arrange sales and inventory databases
    sales = sales.pivot_table(values = ['Units Ordered','Units Ordered - B2B'
                                        ,'Sessions - Total','Ordered Product Sales']
                                        ,index = ['Date', 'ASIN']
                                        ,aggfunc = sum).reset_index()
    # sales['Date'] = pd.to_datetime(sales['Date'])
    inventory = inventory.pivot_table(values = ['Quantity Available','price'],
                                  index = ['Date', 'ASIN'],aggfunc = {
                                      'Quantity Available':'sum',
                                      'price':'min'}
                                           ).reset_index()
    # inventory['Date'] = pd.to_datetime(inventory['Date'])
    inventory = inventory.sort_values('Date')
    finish = round(time.time() - start, 1)

    # inventory['In stock'] = inventory['Quantity Available'].apply(lambda x: 1 if x >2 else 0)
    inventory['In stock'] = 0
    inventory.loc[inventory['Quantity Available']>2, 'In stock'] = 1
    sales_inventory = pd.merge(sales, inventory, how = 'outer', on = ['Date', 'ASIN'])
    dataset = pd.merge(sales_inventory, dictionary, how = 'left', on = 'ASIN')
    dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date

    latest_date = str(dataset['Date'].values[-1]).split('T')[0]
    print(f'Latest date record in database: {latest_date}\nData uploaded in {finish} seconds')
    return dataset, dictionary,skus


def encrypt_string(hash_string):
    '''
    Create a hashed string from any input
    '''
    import hashlib
    string_encoded = hashlib.sha256(str(hash_string).encode()).hexdigest()
    return string_encoded    

def create_user(action = 'append'):
    import pandas as pd
    '''
    create a user in user database with empty permissions
    '''
    prefix = get_db_path()[3]
    script_path = os.path.join(prefix,r'70 Data & Technology\70.03 Scripts')
    username = user_login()[0]
    current_user = get_db_path()[5]
    import sqlite3
    import json
    db_file = os.path.join(prefix,script_path,r'mellanni_2\user_data\users.db')
    user_file = os.path.join(current_user,'mellanni','user_data.ini')
    import PySimpleGUI as sg
    
    layout = [
        [sg.Text(f'Current user: {username}')],
        [sg.Text('Input user name:'), sg.In(key = 'NAME')],
        [sg.Text('Input user email:'), sg.In(key = 'EMAIL')],
        [sg.Text('Create a password:'), sg.In(password_char = '*',key = 'PASSWORD')],
        # [sg.Text('Select user group'),
        #  sg.Listbox(groups,select_mode='multiple', size = (30,6),key = 'GROUPS')],
        [sg.Button('OK',focus = True), sg.Button('Cancel')]
        ]
    window = sg.Window('Create User', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
        elif event ==  'OK':
            Name = values['NAME']
            Email = values['EMAIL']
            Password = encrypt_string(values['PASSWORD'])
            # group = values['GROUPS']
            # user = User(Name, Email, Password, group)
            user = {
                'name':Name,
                'email':Email,
                'password':Password,
                'group':['New user'],
                'user_hash':None,
                'user_markets':['New user']
                }
            user_db = pd.DataFrame.from_dict(user)
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            if action == 'append':
                sql = 'SELECT email FROM Users'
                result = cursor.execute(sql).fetchall()
                if any([user['email'] in x for x in result]):
                    sg.Popup('User already exists, skipping')
                else:
                    user_db.to_sql('Users', conn, if_exists = 'append', index = False)
            elif action == 'replace':
                user_db.to_sql('Users', conn, if_exists = 'replace', index = False)
            # if os.path.isfile(user_file):
            #     with open(user_file, 'w') as f:
            #         json.dump(user,f)
            # else:
            try:
                os.makedirs(os.path.join(current_user,'mellanni'))
            except:
                pass
            with open(user_file, 'w') as f:
                json.dump(user,f)
            encode = user['password']+user_file
            user_hash = encrypt_string(encode)
            sql = 'UPDATE Users SET user_hash = "'+user_hash+'" WHERE email = "'+user['email']+'";'
            cursor.execute(sql)
            conn.commit()
            conn.close()
            break
    window.close()
    return None

def user_login():
    '''
    Check for user credentials using the file and file path as a key
    '''
    user_groups = None
    user_markets = None
    import json
    import sqlite3
    current_user,script_folder = get_db_path()[5],get_db_path()[9]
    db_file = os.path.join(script_folder,'mellanni_2','user_data','users.db')
    user_file = os.path.join(current_user,'mellanni','user_data.ini')
    if os.path.isfile(user_file):
        with open(user_file, 'r') as f:
            text = json.load(f)['password']
        hash_check = encrypt_string(text+user_file)
        conn = sqlite3.connect(db_file)
        sql = 'SELECT email, "group", user_markets, user_hash FROM Users'
        test = pd.read_sql(sql, conn)
        conn.close()
        check = test[test['user_hash'] == hash_check]
        if len(check)>0:
            user = check['email'].unique()[0]
            user_groups = check['group'].unique().tolist()
            user_markets = check['user_markets'].unique().tolist()
        else:
            user, user_groups, user_markets = 'Unknown','Unknown','Unknown'
    else:
        user, user_groups, user_markets = 'Unknown','Unknown','Unknown'
    return user, user_groups, user_markets

def delete_user(user_email):
    '''
    Delete user from the users database. Requires user email as input.
    '''
    import sqlite3
    db_file = os.path.join('user_data','users.db')
    conn = sqlite3.connect(db_file)
    sql = 'DELETE FROM Users WHERE email = "'+user_email+'";'
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    print(f'Deleted {cursor.rowcount} rows from Users db')
    conn.close()
    return None

def recreate_user():
    import PySimpleGUI as sg
    delete_user(input('Input user email?\n'))
    current_user = get_db_path()[5]
    user_file = os.path.join(current_user,'mellanni','user_data.ini')
    if os.path.isfile(user_file):
        os.remove(user_file)
    else:
        sg.Popup('User not found')
    create_user()
    return None

def main_check():
    current_user = user_login()[0]
    if current_user != 'Unknown':
        return current_user
    else:
        create_user()
        return None
    
def user_markets():
    '''
    update permissions for existing users (access to marketplaces
    and selling accounts reporting)
    '''
    import pandas as pd
    import PySimpleGUI as sg
    txt = (25,1)
    markets, accounts = account_list()
    if user_login()[0] == 'sergey@poluco.co':
        db_file = os.path.join('user_data','users.db')
        import sqlite3
        conn = sqlite3.connect(db_file)
        sql = 'SELECT * FROM Users;'
        user_info = pd.read_sql(sql, conn)
        conn.close()
        emails = user_info['email'].unique()
        layout = [
            [sg.Text('Select user email to update', size = txt), sg.DropDown(emails, key = 'EMAILS', enable_events = True)],
            [sg.Text('Current user permissions:', key = 'PERMISSIONS')],
            [sg.Text('Select markets for user', size = txt),
             sg.Listbox(markets, key = 'MARKETS',select_mode='multiple', size = (12,3))],
            [sg.Text('Select accounts for user', size = txt),
             sg.Listbox(accounts, key = 'ACCOUNTS',select_mode='multiple', size = (12,5))],
            [sg.Button('OK'), sg.Button('Cancel'), sg.Button('Delete user')]
            ]
        window = sg.Window('Assign markets', layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                break
                
            elif event == 'EMAILS':
                groups = user_info[user_info['email']==values['EMAILS']]['group'].unique()
                markets = user_info[user_info['email']==values['EMAILS']]['user_markets'].unique()
                try:
                    window['PERMISSIONS'].update(f"Current user permissions:\n{', '.join(groups)} | {', '.join(markets)}")
                except:
                    window['PERMISSIONS'].update("Current user permissions currently not set")
            
            elif event == 'Delete user':
                if values['EMAILS'] != '':
                    delete_user(values['EMAIL'])

            elif event ==  'OK':
                email = values['EMAILS']
                new_markets = values['MARKETS']
                new_accounts = values['ACCOUNTS']
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                sql = 'UPDATE Users SET "group" = "'+', '.join(new_accounts)+'" WHERE email = "'+email+'";'
                cursor.execute(sql)
                sql = 'UPDATE Users SET user_markets = "'+', '.join(new_markets)+'" WHERE email = "'+email+'";'
                cursor.execute(sql)
                conn.commit()
                conn.close()
                break
        window.close()
    return None

