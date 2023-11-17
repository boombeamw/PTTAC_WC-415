import streamlit as st
import pandas as pd
#import numpy as np
import pickle
#import plotly.express as px
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#from matplotlib import pyplot as plt

st.set_page_config(page_title="Homepage",page_icon="")

st.sidebar.success("Select a page above.")

st.write('# WC-415 Health Index Prediction') 
st.image('picture/dashboard.png', width=90)
st.image('picture/GCME_Logo.png', width=200)

#from annotated_text import annotated_text
#annotated_text(
#    ("This is a dashboard showing the WC-415 Health Index Prediction by GCME.",),)


st.markdown(
    """
    **🙋 Select a demo dataset from the sidebar**
"""
)



st.sidebar.header("Input features for simulation")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


select = st.sidebar.radio("Select Model",('DecisionTree','XGBoost'))

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df.drop(columns=['Date'])
else:
    def user_input_features():
        TEMP_DE = st.sidebar.slider('TEMP DE',0, 200,40)
        VIB_DE = st.sidebar.slider('VIB DE',0, 50,2)
        TEMP_NDE = st.sidebar.slider('TEMP NDE',0, 200,40)
        VIB_NDE = st.sidebar.slider('VIB NDE',0, 50,2)
        Damper_Value_PO = st.sidebar.slider('Damper_Value_PO',0, 100,12)
        Motor_Current = st.sidebar.slider('Motor_Current',0, 100,30)
        COMBUSTION_AIR_TEMP = st.sidebar.slider('COMBUSTION_AIR_TEMP',0, 200,60)
        WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP = st.sidebar.slider('WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',0, 100,15)
        LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER = st.sidebar.slider('LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER',0, 8000,5000)
        LV_BNR_COMBUSTION_AIR_FLOW = st.sidebar.slider('LV_BNR_COMBUSTION_AIR_FLOW',0, 3000,1400)
        WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP = st.sidebar.slider('WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',0, 200,20)
        HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER = st.sidebar.slider('HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER',0, 3000,1540)
        WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP = st.sidebar.slider('WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',0, 100,35)
        WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER = st.sidebar.slider('WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER',0, 10000,5800)

    
        data = {
            'TEMP_DE':TEMP_DE,
            'VIB_DE':VIB_DE,
            'TEMP_NDE':TEMP_NDE,
            'VIB_NDE':VIB_NDE,
            'Damper_Value_PO':Damper_Value_PO,
            'Motor_Current':Motor_Current,
            'COMBUSTION_AIR_TEMP':COMBUSTION_AIR_TEMP,
            'WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP':WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP,
            'LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER':LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER,
            'LV_BNR_COMBUSTION_AIR_FLOW':LV_BNR_COMBUSTION_AIR_FLOW,
            'WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP':WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP,
            'HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER':HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER,
            'WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP':WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP,
            'WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER':WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


#-----------------------------------------------------------------
lastrow = len(input_df.index)
st.write('Number of row for predition :')
st.write(lastrow)

data_train = pd.read_csv('./01.Dataset_for_deployment/20231116_For_deployment.csv')
data_train = data_train.drop(columns=['Date',])
#st.write(data_train.shape)
#st.table(input_df)


# Selects only the first row (the user input data)
input_df = input_df[:] 
#st.table(df)

#st.write(input_df)

# Displays the user input features
st.subheader('1. Features for Simulation')

if uploaded_file is not None:
   st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(input_df)

#Create a function for LabelEncoder
def Encoder(input_df):
          columnsToEncode = list(input_df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  input_df[feature] = le.fit_transform(input_df[feature])
              except:
                  print('Error encoding '+feature)
          return input_df
columnsToEncode = list(input_df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
              try:
                  input_df[feature] = le.fit_transform(input_df[feature])
              except:
                  print('Error encoding '+feature)
input_df = Encoder(input_df)

df_total = pd.concat([data_train,input_df],axis=0,ignore_index=True)
df = input_df
#Supervised
#1.temp
def label_temp(temp_value):
    if temp_value <37:
        return 2
    elif 37<= temp_value <=49:
        return 1
    elif 49< temp_value <=60:
        return 3
    else:
        return 4
    
df['label_TEMP_DE'] = df['TEMP_DE'].apply(label_temp)
df['label_TEMP_NDE'] = df['TEMP_NDE'].apply(label_temp)


#2.VIB
def label_vib(vib_value):
    if vib_value <2.5:
        return 1
    elif 2.5<= vib_value <=6:
        return 2
    elif 6< vib_value <=11:
        return 3
    else:
        return 4
    
df['label_VIB_DE'] = df['VIB_DE'].apply(label_vib)
df['label_VIB_NDE'] = df['VIB_NDE'].apply(label_vib)


#3.Damper Value OP_1 Labeling
def label_DamperValueOP_1(DamperValueOP1_value):
    if DamperValueOP1_value <12:
        return 2
    elif 12 <= DamperValueOP1_value <=20:
        return 1
    elif 20< DamperValueOP1_value <=40:
        #mean=12 SD=2.1
        return 3
    else:
        return 4
    
df['label_Damper_Value_PO'] = df['Damper_Value_PO'].apply(label_DamperValueOP_1)


#4.WC-415 Main motor current_1 Labeling
def label_current(current_value):
    if current_value <35:
        return 1
    elif 35 <= current_value <=40:
        return 2
    else:
        return 3
    
df['label_Motor_current'] = df['Motor_Current'].apply(label_current)



#5.COMBUSTION AIR TEMP. Labeling
def label_ComAirTemp(ComAirTemp_value):
    if ComAirTemp_value <60:
        return 2
    elif 60 <= ComAirTemp_value <=80:
        return 1
    elif 80 < ComAirTemp_value <=100:
#86 is mean+4*SD
        return 3
    else:
        return 4
    
df['label_COMBUSTION_AIR_TEMP'] = df['COMBUSTION_AIR_TEMP'].apply(label_ComAirTemp)


#6.WZ-411A LV BNR. COMBUSTION AIR FLOW CONTROL VALUE OP_1
def label_LV_ConValve(LV_ConValve_value):
    if LV_ConValve_value <15:
        return 2
    elif 15 <= LV_ConValve_value <=33:
        #mean value is 17.61
        #33.44 is mean+(2.5*SD)
        return 1
    elif 33 < LV_ConValve_value <=43:
        #42.93 is mean+(4*SD)
        return 3
    else:
        return 4
    
df['label_WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'] = df['WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'].apply(label_LV_ConValve)


#7.WZ-411A LV BNR. COMBUSTION AIR FLOW TRANSMITTER 
def label_LV_Trans(LV_Trans_value):
    if LV_Trans_value <5000:
        return 2
    elif 5000 <= LV_Trans_value <=6400:
        #mean value is 5699
        return 1
    elif 6400 < LV_Trans_value <=9053:
        #9052.23 is mean+(2.5*SD)
        return 2
    else:
        return 3
    
df['label_LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER'] = df['LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER'].apply(label_LV_Trans)


#8.WZ-411A LV BNR. Combustion air flow 
def label_411A_LVBRG_AirFlow_1(LVBRG_AirFlow_1_value):
    if LVBRG_AirFlow_1_value <1400:
        return 2
    elif 1400 <= LVBRG_AirFlow_1_value <=2076:
        #mean value is 1972
        return 1
    elif 2076 < LVBRG_AirFlow_1_value <=3496:
        return 2
    #3496 = mean+2.5*SD
    elif 3496 < LVBRG_AirFlow_1_value <=4410:
        #4410 = mean+4*SD
        return 3
    else:
        return 4
    
df['label_LV_BNR_COMBUSTION_AIR_FLOW'] = df['LV_BNR_COMBUSTION_AIR_FLOW'].apply(label_411A_LVBRG_AirFlow_1)


#9.WZ-411B LV BNR. COMBUSTION AIR FLOW CONTROL VALUE OP
def label_411B_LVBRG_Control_Valve_OP(B_LVBRG_Control_Valve_OP_value):
    if B_LVBRG_Control_Valve_OP_value <20:
        return 2
    elif 20 <= B_LVBRG_Control_Valve_OP_value <=32:
        return 1
    elif 32 < B_LVBRG_Control_Valve_OP_value <=63:
        #62.45 = mean+2.5*SD
        return 2
    elif 63 < B_LVBRG_Control_Valve_OP_value <=82:
        return 3
    #81.8 = mean+4*SD
    else:
        return 4
    
df['label_WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'] = df['WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'].apply(label_411B_LVBRG_Control_Valve_OP)


#10.WZ-411B HV BNR. COMBUSTION AIR FLOW TRANSMITTER 
def label_411B_HVBRG_Transmitter(B_HVBRG_Transmitter_value):
    if B_HVBRG_Transmitter_value <1540:
        return 2
    elif 1540 <= B_HVBRG_Transmitter_value <=1940:
        return 1
    elif 1940 < B_HVBRG_Transmitter_value <=2362:
        #2362 = mean+2.5*SD
        return 2
    elif 2362 < B_HVBRG_Transmitter_value <=2669:
        #2668.6 = mean+2.5*SD
        return 3
    else:
        return 4
    
df['label_HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER'] = df['HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER'].apply(label_411B_HVBRG_Transmitter)


#11.WR-410 WASTEWATER COMBUSTION AIR FLOW CONTROL VALUE OP
def label_410_WASTEWATER_Control_Valve_OP(WASTEWATER_Control_Valve_value):
    if WASTEWATER_Control_Valve_value <33:
        return 2
    elif 33 <= WASTEWATER_Control_Valve_value <=50:
        return 1
    elif 50 < WASTEWATER_Control_Valve_value <=63:
        #63=50+2.5*SD
        return 3
    else:
        return 4
    
df['label_WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'] = df['WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP'].apply(label_410_WASTEWATER_Control_Valve_OP)


#12.WR-410 WASTE WATER COMBUSTION AIR FLOW TRANSMITTER
def label_410_WASTEWATER_Transmitter(WASTEWATER_Transmitter_value):
    if WASTEWATER_Transmitter_value <5100:
        return 2
    elif 5100 <= WASTEWATER_Transmitter_value <=9000:
        return 1
    elif 9000 < WASTEWATER_Transmitter_value <=10328:
        #10328=mean+4*SD
        return 3
    else:
        return 4
    
    
df['label_WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER'] = df['WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER'].apply(label_410_WASTEWATER_Transmitter)


df_label = df.iloc[:,14:29]

#st.write(df_label)
#st.write(df_label.shape)
# Scale data
#st.write(df_total)
scaler = StandardScaler()
df_Scaler = scaler.fit_transform(df_total.iloc[:,0:13])
#st.write(df_Scaler)
#st.write(df_Scaler.shape)
df_ForPredict = df_Scaler[694040:,:]
#st.write(df_ForPredict)
#st.write(df_ForPredict.shape)

# Reads in saved regression model
#df_ForPredict = 

load_clf1 = pickle.load(open('20231116_DTModel.pkl','rb'))
load_clf2 = pickle.load(open('20231116_XGBModel.pkl','rb'))

# Apply model to make predictions
predict = pd.DataFrame(df_ForPredict).iloc[:lastrow]
#st.write(predict)
prediction = load_clf2.predict(predict)

#st.write(prediction)
if select == 'DecisionTree':
    prediction = load_clf1.predict(predict)
elif select == 'XGBoost':
    prediction = load_clf2.predict(predict)

#----------------------------------------------------------

st.subheader('2.Health Percentage Prediction')
st.write('Health Percentage')
prediction = prediction.astype(int)

st.write(prediction)

#-----------------------------------------------------------
last_health_percent = prediction[-1]
st.markdown(
    """
    **🎯The Recent of WC415 s Health Percentage is :**
"""
)

text = last_health_percent
color = "blue"
st.markdown(f'<h4 style="color:{color};">{text}%</h4>', unsafe_allow_html=True)
#st.image('picture/percentage.png', width=50)
#st.markdown(f'<div style="background-color:{background_color}; padding: 20px;">{text}</div>', unsafe_allow_html=True)
#st.subheader(last_health_percent)


# Create a colored box using Markdown

st.subheader('3.Health Percentage Chart')
st.write("- line chart of Health Percentage")
st.bar_chart(prediction)


st.subheader('4.Severity of each parameter')

label = df_label.iloc[:lastrow]
#st.write(label.columns)
label['Max_of_severity'] = label[['label_TEMP_DE','label_VIB_DE','label_TEMP_NDE','label_VIB_NDE',
                                  'label_Damper_Value_PO','label_Motor_current','label_COMBUSTION_AIR_TEMP',
                                  'label_WFC416ALV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',
                                  'label_LV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER',
                                  'label_LV_BNR_COMBUSTION_AIR_FLOW',
                                  'label_WFC416BLV_BNR_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',
                                  'label_HV_BNR_COMBUSTION_AIR_FLOW_TRANSMITTER',
                                  'label_WASTE_WATER_COMBUSTION_AIR_FLOW_CONTROL_VALUE_OP',
                                  'label_WASTE_WATER_COMBUSTION_AIR_FLOW_TRANSMITTER']].max(axis=1)
#st.write(label)

#last_max = pd.DataFrame(label)

#Save prediction file to csv

predic = pd.Series(prediction, name='Health Percentage')
df_concat = pd.concat([input_df.iloc[:,:], label.iloc[:,0:14]], axis=1)
df_final = pd.concat([df_concat.iloc[:,:], pd.Series(predic)], axis=1)
st.write(predic)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv()

csv = convert_df(df_final)


st.download_button(
    label="Download prediction as CSV",
    data=csv,
    file_name='prediction_file.csv',
    mime='text/csv',
)