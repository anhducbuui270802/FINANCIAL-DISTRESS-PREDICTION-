import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import load

loaded_scaler = load('scaler.joblib')

label_encoder = LabelEncoder()

def convert_string_to_float(x):
    if type(x) == str:
        x = x.replace("(", "")
        x = x.replace(")", "")
        x = x.replace(",", "")
        return float(x)
    return x

# Title
st.header("FINANCIAL DISTRESS PREDICTION FOR NON-FINANCIAL ENTERPRISES IN VIETNAM")

col1, col2 = st.columns(2)
with col1:
    EBIT = st.number_input("EBIT")
with col2:
    EBITDA = st.number_input("EBITDA")

col3, col4 = st.columns(2)
with col3:
    TOTAL_EQUITY_TOTAL_ASSETS = st.number_input("TOTAL EQUITY/TOTAL ASSETS")
with col4:
    EPS = st.number_input("EPS")

col5, col6 = st.columns(2)
with col5:
    CASH_TOTAL_CURRENT_ASSETS = st.number_input("CASH/TOTAL CURRENT ASSETS")
with col6:
    TOTAL_CURRENT_ASSET_TOTAL_ASSET = st.number_input("TOTAL CURRENT ASSET/TOTAL ASSET")

col7, col8 = st.columns(2)
with col7:
    LONG_TERM_ASSETS_TOTAL_ASSETS = st.number_input("LONG-TERM ASSETS/TOTAL ASSETS")
with col8:
    QUICK_RATIO = st.number_input("QUICK RATIO")

col9, col10 = st.columns(2)
with col9:
    Market_Value_of_Total_Equity_Book_Values_of_Total_Liabilities = st.number_input("Market Value of Total Equity / Book Values of Total...")
with col10:
    Sales_Total_Assets = st.number_input("Sales/Total Assets")

col11, col12 = st.columns(2)
with col11:
    EBIT_Total_Assets = st.number_input("EBIT/Total Assets")
with col12:
    Retain_Earnings_Total_Assets = st.number_input("Retain Earnings/Total Assets")

col13, col14 = st.columns(2)
with col13:
    Working_Capitals_Total_Asset = st.number_input("Working Capitals/Total Asset")
with col14:
    ROIC = st.number_input("ROIC")

col15, col16 = st.columns(2)
with col15:
    AGE = st.number_input("AGE")
with col16:
    NET_INCOM_TOTAL_ASSET = st.number_input("NET INCOME/TOTAL ASSET")




button_style = '''
    <style>
        .stButton button {
            
            background-color: #0072B1;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 8px 16px;
            box-shadow: none;
            
        }
        .stButton button:hover {
            color: white;
            background-color: #0072B1;
            box-shadow: none;
            border: none;
        }
    </style>'''

st.markdown(button_style, unsafe_allow_html=True)

# If button is pressed
if st.button("PREDICT"):
    # Unpickle classifier
    clf = joblib.load("./best_model.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame(
        [[EBIT, EBITDA, TOTAL_EQUITY_TOTAL_ASSETS, EPS, CASH_TOTAL_CURRENT_ASSETS, TOTAL_CURRENT_ASSET_TOTAL_ASSET, LONG_TERM_ASSETS_TOTAL_ASSETS, QUICK_RATIO, Market_Value_of_Total_Equity_Book_Values_of_Total_Liabilities, Sales_Total_Assets, EBIT_Total_Assets, Retain_Earnings_Total_Assets, Working_Capitals_Total_Asset, ROIC, AGE, NET_INCOM_TOTAL_ASSET]],
        columns=[
            'EBIT', 'EBITDA',
            'TOTAL EQUITY/TOTAL ASSETS', 'EPS', 'CASH/TOTAL CURRENT ASSETS',
            'TOTAL CURRENT ASSET/TOTAL ASSET', 'LONG-TERM ASSETS/TOTAL ASSETS',
            'QUICK RATIO',
            'Market Value of Total Equity / Book Values of Total Liabilities',
            'Sales/Total Assets', 'EBIT/Total Assets',
            'Retain Earnings/Total Assets', 'Working Capitals/Total Asset', 'ROIC',
            'AGE', 'NET INCOME/TOTAL ASSET'

        ],
    )
    # X['STOCK EXCHANGE'] = label_encoder.fit_transform(X['STOCK EXCHANGE'])
    X = X.applymap(convert_string_to_float)
    regression_fearture = X.columns.tolist()
    # regression_fearture.remove('STOCK EXCHANGE')
    regression_fearture.remove('AGE')
    X[regression_fearture] = loaded_scaler.transform(X[regression_fearture])

    
    # Get prediction
    prediction = clf.predict(X)[0]
    probabilities = clf.predict_proba(X)
    
    # Output prediction
    if prediction == 0:
        predict = "Không có nguy cơ gặp khó khăn trong tương lai"
        # probabiliti = "{:.2f}%".format(probabilities[0, 1] * 100)
    else:
        predict = "⚠️ Có nguy cơ gặp khó khăn trong tương lai"
        # probabiliti = "{:.2f}%".format(probabilities[0, 0] * 100)
    probabiliti = max((probabilities[0])) * 100
    st.text(f"Dự đoán:     \n{predict}     [Mức tin cậy: {probabiliti:.2f}%]")
    

# Batch prediction
st.header("Dự đoán hàng loạt")
uploaded_file = st.file_uploader("Chọn file EXCEL", type=["xlsx"])

if uploaded_file is not None:
    # Load classifier
    clf = joblib.load("./best_model.pkl")

    origin_df = pd.read_excel(uploaded_file)
    df_na = origin_df[origin_df.isna().any(axis=1)]
    origin_df = origin_df.dropna()
    df = origin_df.copy()
    df = pd.DataFrame(df, columns=['CODE', 'NAME', 
            'STOCK EXCHANGE', 'YEAR','EBIT', 'EBITDA',
            'TOTAL EQUITY/TOTAL ASSETS', 'EPS', 'CASH/TOTAL CURRENT ASSETS',
            'TOTAL CURRENT ASSET/TOTAL ASSET', 'LONG-TERM ASSETS/TOTAL ASSETS',
            'QUICK RATIO',
            'Market Value of Total Equity / Book Values of Total Liabilities',
            'Sales/Total Assets', 'EBIT/Total Assets',
            'Retain Earnings/Total Assets', 'Working Capitals/Total Asset', 'ROIC',
            'AGE', 'NET INCOME/TOTAL ASSET'
        ],)

    # Preprocess input data
    df.drop(['CODE', 'NAME', 'STOCK EXCHANGE', 'YEAR'], axis=1, inplace=True)
    try:
        df.drop(['TARGET'], axis=1, inplace=True)
    except:
        pass
    # df['STOCK EXCHANGE'] = label_encoder.fit_transform(df['STOCK EXCHANGE'])
    df = df.applymap(convert_string_to_float)
    regression_fearture = df.columns.tolist()
    print(regression_fearture)
    # regression_fearture.remove('STOCK EXCHANGE')
    regression_fearture.remove('AGE')
    df[regression_fearture] = loaded_scaler.transform(df[regression_fearture])
    df = df.applymap(convert_string_to_float)
    
    # Make predictions
    predictions_df = clf.predict(df)
    probabilities_df = clf.predict_proba(df)
    outout_probabilities_df = np.max(probabilities_df, axis=1)
    outout_probabilities_df = np.round(outout_probabilities_df * 100, 2)

    # Add predictions as a new column in the DataFrame
    origin_df['TARGET PREDICT'] = predictions_df
    origin_df['PROBABILITIES'] = outout_probabilities_df

    # Output input data and predictions
    st.dataframe(origin_df)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
        # return df.to_excel(index=False).encode('utf-8')

    csv = convert_df(origin_df)
    
    st.download_button(
    "Press to Download cvs file",
    csv,
    "predicted.csv",
    "text/csv",
    key='download-csv'
    )
        
    # Filter
    Filter = st.selectbox("Filter", ("Các trường hợp dự đoán có độ tin cậy thấp", "Các trường hợp liệu bị thiếu", "Những công ty có khả năng gặp khó khăn", "Những công ty không có khả năng gặp khó khăn"))
    if st.button("Filter"):
        if Filter == "Các trường hợp dự đoán có độ tin cậy thấp" :
            df_low_probabilities = origin_df[origin_df["PROBABILITIES"] < 0.6]
            csv_low_probabilities = convert_df(df_low_probabilities)
            st.write(f"Các trường hợp dự đoán có độ tin cậy thấp:")
            st.dataframe(df_low_probabilities)
            st.download_button(
            "Press to Download CSV file",
            csv_low_probabilities,
            "predicted_low_probabilities.csv",
            "text/csv",
            key='download-csv-low-probabilities'
            )
        elif Filter == "Các trường hợp liệu bị thiếu" :
            csv_df_na = convert_df(df_na)
            st.write(f"Các trường hợp liệu bị thiếu:")
            st.dataframe(df_na)
            st.download_button(
            "Press to Download CSV file",
            csv_df_na,
            "lack_feature.csv",
            "text/csv",
            key='download-csv-low-probabilities'
            )
        elif Filter == "Những công ty có khả năng gặp khó khăn":
            df_predict_1 = origin_df[origin_df["TARGET PREDICT"] == 1]
            csv_predict_1 = convert_df(df_predict_1)
            st.write(f"Những công ty có khả năng gặp khó khăn:")
            st.dataframe(df_predict_1)
            st.download_button(
            "Press to Download CSV file",
            csv_predict_1,
            "company_is_in_trouble.csv",
            "text/csv",
            key='download-csv-low-probabilities'
            )
        elif Filter == "Những công ty không có khả năng gặp khó khăn":
            df_predict_0 = origin_df[origin_df["TARGET PREDICT"] == 0]
            csv_predict_0 = convert_df(df_predict_0)
            st.write(f"Những công ty không có khả năng gặp khó khăn:")
            st.dataframe(df_predict_0)
            st.download_button(
            "Press to Download CSV file",
            csv_predict_0,
            "company_is_not_in_trouble.csv",
            "text/csv",
            key='download-csv-low-probabilities'
            )


