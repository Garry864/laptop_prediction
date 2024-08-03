import pickle
import streamlit as st
import numpy as np
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

# Load pre-trained models
rf = pickle.load(open('rf1_28July.sav', 'rb'))
xgb = pickle.load(open('xgb1_28July.sav', 'rb'))

# Set page configuration
st.set_page_config(page_title="AI Data Prediction",
                   layout="wide",
                   page_icon="🚘")


Prediction, Data_info, Visualization = st.tabs(["Prediction", "Dataset Overview", "Visualization"])

with Prediction:

    # Title and header
    st.title('Laptop Price Prediction Web App')
    st.header('Fill in the details to generate a laptop price prediction')

    # st.image('Design 2.png')


    # ppt = """
    #     <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRjyiYzrwAk1k0KMe_xuazejngWVlQo485K0DEa8maduD7nI80cTNIr_IR8ZAQqGg/embed?start=false&loop=false&delayms=3000" frameborder="0" width="1125" height="675" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
    # """

    # st.markdown(ppt, unsafe_allow_html=True)

    st.video("Laptop-prediction.mp4", start_time=0)

    # components.iframe("https://www.canva.com/design/DAGMu5ilkHw/Rq9w5uIDH3oyZZEUQNG8yQ/view?utm_content=DAGMu5ilkHw&utm_campaign=designshare&utm_medium=link&utm_source=editor",width=1024, height=628)

    # Sidebar for model selection
    options = st.sidebar.selectbox('Select ML model', ['RF_Reg', 'XGB_Reg'])





    # Input fields for user data

    col1, col2, col3 = st.columns(3)

    with col1:
        company = st.selectbox('Company', ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Other', 'Toshiba', 'Apple'])

    with col2:
        typename = st.selectbox('TypeName', ['Notebook', 'Gaming', 'Ultrabook', '2in1', 'Workstation', 'Netbook'])

    with col3:
        ram = st.selectbox('RAM', [2, 4, 6, 8, 12, 16, 24, 32, 64])

    # with col1:
    weight = st.slider('Weight', min_value=0.7, max_value=4.2, step=0.1)

    with col2:
        touchscreen = st.selectbox('TouchScreen', ['Yes', 'No'])

    with col3:
        ips = st.selectbox('IPS', ['Yes', 'No'])

    with col1:
        cpu = st.selectbox('CPU', ['I7', 'I5', 'I3', 'AMD', 'Other'])

    with col2:
        hdd = st.selectbox('HDD', [0, 128, 500, 1000, 2000])

    with col3:
        ssd = st.selectbox('SSD', [0, 8, 16, 32, 64, 128, 180, 256, 512, 768, 1000])

    with col1:
        gpu = st.selectbox('GPU', ['Intel', 'Nvidia', 'AMD'])

    with col2:
        os = st.selectbox('OS', ['Win', 'Linux/Other', 'MAC'])




    # Encoding categorical variables
    company_dict = {'Dell': 3, 'Lenovo': 5, 'HP': 4, 'Asus': 2, 'Acer': 0, 'MSI': 6, 'Other': 7, 'Toshiba': 8, 'Apple': 1}
    typename_dict = {'Notebook': 3, 'Gaming': 1, 'Ultrabook': 4, '2in1': 0, 'Workstation': 5, 'Netbook': 2}
    cpu_dict = {'I7': 0, 'I5': 1, 'I3': 2, 'AMD': 3, 'Other': 4}
    gpu_dict = {'Intel': 0, 'Nvidia': 1, 'AMD': 2}
    os_dict = {'Win': 0, 'Linux/Other': 1, 'MAC': 2}

    # Encode user inputs
    company = company_dict[company]
    typename = typename_dict[typename]
    cpu = cpu_dict[cpu]
    gpu = gpu_dict[gpu]
    os = os_dict[os]

    # Convert Yes/No to binary
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Prepare input features
    features = np.array([company, typename, ram, weight, touchscreen, ips, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

    # Make prediction
    if st.button('Predict'):
        if options == 'RF_Reg':
            prediction = rf.predict(features)[0]
        else:
            prediction = xgb.predict(features)[0]
        
        st.success(f'The predicted price of the laptop is: ${prediction:.2f}')



with Data_info:

    # Add title to the page
        st.title("Data Info page")

        # Add subheader for the section
        st.subheader("View Data")

        # Create an expansion option to check the data
        with st.expander("View Raw data"):
            df = pd.read_csv("laptop_price_data.csv")
            st.dataframe(df)
            st.subheader("This is Raw Dataset Befor Preprocessing")
            
    
        st.subheader("Columns Description:")


         # Create multiple check box in row
        col_name, col_dtype, col_data = st.columns(3)

        # Create a checkbox to get the summary.
        # with veiw_summary:
        if st.checkbox("View Summary"):
            st.dataframe(df.describe())

        # Create multiple check box in row

       

        # Show name of all dataframe
        with col_name:
            if st.checkbox("Column Names"):
                st.dataframe(df.columns)

        # Show datatype of all columns 
        with col_dtype:
            if st.checkbox("Columns data types"):
                dtypes = df.dtypes.apply(lambda x: x.name)
                st.dataframe(dtypes)
        
        # Show data for each columns
        with col_data: 
            if st.checkbox("Columns Data"):
                col = st.selectbox("Column Name", list(df.columns))
                st.dataframe(df[col])


with Visualization:
    # Load your dataset
    df = pd.read_csv("laptop_price_data.csv")

    # Group by CPU brand and sum the prices
    cpu_brand = df.groupby(['Cpu brand'])['Price'].sum().reset_index()

    # Creating two columns for visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CPU Wise Prices")

        # Creating a bar chart for CPU brand prices
        fig = px.bar(cpu_brand, x='Cpu brand', y='Price', 
                    template='seaborn', 
                    text=cpu_brand['Price'].apply(lambda x: '${:,.2f}'.format(x)))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Copying the dataframe for market share manipulation
        market_df = df.copy()

        # Mapping less common companies to 'Other'
        comp_map = {'Samsung':'Other', 'Razer':'Other', 'Mediacom':'Other', 'Microsoft':'Other',
                    'Xiaomi':'Other', 'Vero':'Other', 'Google':'Other', 'LG':'Other', 
                    'Chuwi':'Other', 'Fujitsu':'Other', 'Huawei':'Other'}
        market_df['Company'] = market_df['Company'].replace(comp_map)

        st.subheader("Market Share")

        # Creating a pie chart for market share
        fig = px.pie(market_df, values='Price', names='Company', hole=0.5)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
