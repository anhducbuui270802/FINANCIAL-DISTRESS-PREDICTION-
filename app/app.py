import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Preprocessing input data
## Khởi tạo LabelEncoder
label_encoder = LabelEncoder()
## Convert categorical data to numerical data
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
    STOCK_EXCHANGE = st.selectbox("STOCK EXCHANGE", ("HOSE", "HNX"))
with col2:
    EBIT = st.number_input("EBIT")

col3, col4 = st.columns(2)
with col3:
    EBITDA = st.number_input("EBITDA")
with col4:
    TOTAL_EQUITY_TOTAL_ASSETS = st.number_input("TOTAL EQUITY/TOTAL ASSETS")

col5, col6 = st.columns(2)
with col5:
    EPS = st.number_input("EPS")
with col6:
    CASH_TOTAL_CURRENT_ASSETS = st.number_input("CASH/TOTAL CURRENT ASSETS")

col7, col8 = st.columns(2)
with col7:
    TOTAL_CURRENT_ASSET_TOTAL_ASSET = st.number_input("TOTAL CURRENT ASSET/TOTAL ASSET")
with col8:
    LONG_TERM_ASSETS_TOTAL_ASSETS = st.number_input("LONG-TERM ASSETS/TOTAL ASSETS")

col9, col10 = st.columns(2)
with col9:
    QUICK_RATIO = st.number_input("QUICK RATIO")
with col10:
    Market_Value_of_Total_Equity_Book_Values_of_Total_Liabilities = st.number_input("Market Value of Total Equity / Book Values of Total...")

col11, col12 = st.columns(2)
with col11:
    Sales_Total_Assets = st.number_input("Sales/Total Assets")
with col12:
    EBIT_Total_Assets = st.number_input("EBIT/Total Assets")

col13, col14 = st.columns(2)
with col13:
    Retain_Earnings_Total_Assets = st.number_input("Retain Earnings/Total Assets")
with col14:
    Working_Capitals_Total_Asset = st.number_input("Working Capitals/Total Asset")

col15, col16 = st.columns(2)
with col15:
    ROIC = st.number_input("ROIC")
with col16:
    YEAR = st.number_input("YEAR")

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
    clf = joblib.load("./catboost_model.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame(
        [[STOCK_EXCHANGE, EBIT, EBITDA, TOTAL_EQUITY_TOTAL_ASSETS, EPS, CASH_TOTAL_CURRENT_ASSETS, TOTAL_CURRENT_ASSET_TOTAL_ASSET, LONG_TERM_ASSETS_TOTAL_ASSETS, QUICK_RATIO, Market_Value_of_Total_Equity_Book_Values_of_Total_Liabilities, Sales_Total_Assets, EBIT_Total_Assets, Retain_Earnings_Total_Assets, Working_Capitals_Total_Asset, ROIC]],
        columns=[
            'STOCK EXCHANGE', 'EBIT', 'EBITDA',
         'TOTAL EQUITY/TOTAL ASSETS', 'EPS', 'CASH/TOTAL CURRENT ASSETS',
            'TOTAL CURRENT ASSET/TOTAL ASSET', 'LONG-TERM ASSETS/TOTAL ASSETS',
            'QUICK RATIO',
            'Market Value of Total Equity / Book Values of Total Liabilities',
            'Sales/Total Assets', 'EBIT/Total Assets',
            'Retain Earnings/Total Assets', 'Working Capitals/Total Asset', 'ROIC'
        ],
    )
    # X = X.replace(["Nam", "Nữ"], [1, 0])
    # Mã hóa cột "STOCK_EXCHANGE"
    X['STOCK EXCHANGE'] = label_encoder.fit_transform(X['STOCK EXCHANGE'])


    print(X.info())
    print(X)

    X = X.applymap(convert_string_to_float)
    
    # Get prediction
    y_pred = clf.predict(X)
    prediction = clf.predict(X)[0]
    probabilities = clf.predict_proba(X)
    
    # Output prediction
    if prediction == 0:
        predict = "Không có nguy cơ gặp khó khăn trong tương lai"
        probabiliti = "{:.2f}%".format(probabilities[0, 1] * 100)
    else:
        predict = "⚠️ Có nguy cơ gặp khó khăn trong tương lai"
        probabiliti = "{:.2f}%".format(probabilities[0, 0] * 100)

    st.text(f"Dự đoán:     \n{predict}     [Mức tin cậy: {probabiliti}]")
    
    

# Batch prediction
st.header("Dự đoán hàng loạt")
uploaded_file = st.file_uploader("Chọn file EXCEL", type=["xlsx"])

if uploaded_file is not None:
    # Load classifier
    clf = joblib.load("./demo/clf.pkl")

    df = pd.read_excel(uploaded_file)
    
    # Preprocess input data
    df_na = df[df.isna().any(axis=1)]
    df = df.dropna()
    df.drop(['CODE', 'NAME', 'YEAR'], axis=1, inplace=True)
    try:
        df.drop(['TARGET'], axis=1, inplace=True)
    except:
        pass
    df['STOCK EXCHANGE'] = label_encoder.fit_transform(X['STOCK EdfCHANGE'])


    print(df.info())
    print(df)

    df = df.applymap(convert_string_to_float)
    
    # Make predictions
    predictions = clf.predict(df)
    probabilities = clf.predict_proba(df)
    # print(predictions)

    # Add predictions as a new column in the DataFrame
    df['TARGET'] = predictions

    # Add probabilities as a new column in the DataFrame
    max_values = [max(row) for row in probabilities]
    df['probabilities'] = df.apply(lambda x: max_values[x.name], axis=1)

    # # Output input data and predictions
    # st.dataframe(df)


    # Export DataFrame to csv
    # df.to_csv('output.csv', index=False)
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df)
    


    # Count the labels
    label_counts = df['predict'].value_counts()

    # Output label counts
    if len(label_counts) == 1:
        label = label_counts.index[0]
        if label == 0:
            st.write(f"Tất cả sinh viên đều rớt")
        else:
            st.write(f"Tất cả sinh viên đều đậu")
    else:
        st.write("Số sinh viên dự đoán học lực yếu kém:", label_counts[0])
        st.write("Số sinh viên dự đoán học lực trên trung bình:", label_counts[1])
        st.download_button(
        "Press to Download cvs file",
        csv,
        "predicted.csv",
        "text/csv",
        key='download-csv'
        )
        
        # Filter

        Filter = st.selectbox("Filter:", ("Các trường hợp dự đoán có độ tin cậy thấp", "Các dòng dữ liệu bị thiếu" ))
            

        if st.button("Filter"):
            if Filter == "Các trường hợp dự đoán có độ tin cậy thấp" :
                df_low_probabilities = df[df["probabilities"] < 0.6]
                csv_low_probabilities = convert_df(df_low_probabilities)
                st.write(f"Các trường hợp dự đoán có độ tin cậy thấp:")
                st.dataframe(df_low_probabilities)
                st.download_button(
                "Press to Download cvs file",
                csv_low_probabilities,
                "predicted_low_probabilities.csv",
                "text/csv",
                key='download-csv-low-probabilities'
                )
            elif Filter == "Các dòng dữ liệu bị thiếu" :
                    csv_df_na = convert_df(df_na)
                    st.write(f"Các trường hợp dự đoán có độ tin cậy thấp:")
                    st.dataframe(df_na)
                    st.download_button(
                    "Press to Download cvs file",
                    csv_df_na,
                    "predicted_low_probabilities.csv",
                    "text/csv",
                    key='download-csv-low-probabilities'
                    )


