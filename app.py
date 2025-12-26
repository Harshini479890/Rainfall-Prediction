import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rainfall Prediction",layout="centered")

st.title("Rainfall Prediction System")
st.write("Rainfall Prediction Project")

states=["Maharashtra","Kerala","Tamil Nadu","Rajasthan","Punjab"]
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]

data=[]

for state in states:
    for month_index,month in enumerate(months):
        for _ in range(4): #4 records per month
            if month in ["Jun","Jul","Aug","Sep"]:
                rainfall=150+month_index*5
                humidity=75
                temp=28
            else:
                rainfall=20+month_index*2
                humidity=45
                temp=35

            if state=="Kerala":
                rainfall+=40
            elif state=="Rajasthan":
                rainfall-=15

            wind_speed=10+month_index

            data.append([state,month,temp,humidity,wind_speed,rainfall])

df=pd.DataFrame(
    data,
    columns=[
        "State","Month","Avg_Temperature",
        "Humidity","Wind_Speed","Rainfall_mm"
    ]
)

if st.checkbox("Show Dataset"):
    st.write(df.head(20))
    st.write("Total Records:",df.shape[0])

df_encoded=pd.get_dummies(df,drop_first=True)

x=df_encoded.drop("Rainfall_mm",axis=1)
y=df_encoded["Rainfall_mm"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,shuffle=False
)

model=LinearRegression()
model.fit(x_train,y_train)

st.subheader("Enter Climate Details ")

state_input=st.selectbox("Select State",states)
month_input=st.selectbox("Select Month",months)

temperature_input=st.slider("Average Temperature (C)",20,45,30)
humidity_input=st.slider("Humidity (%)",30,90,60)
wind_input=st.slider("Wind Speed (km/h)",5,25,12)

input_data=pd.DataFrame([{
    "State":state_input,
    "Month":month_input,
    "Avg_Temperature":temperature_input,
    "Humidity":humidity_input,
    "Wind_Speed":wind_input
}])

input_encoded=pd.get_dummies(input_data)
input_encoded=input_encoded.reindex(columns=x.columns,fill_value=0)

if st.button("Predict Rainfall"):
    prediction=model.predict(input_encoded)[0]
    st.success(f"Predicted Rainfall: {prediction:.2f} mm")

st.write("---")
st.caption("Synthetic India Climate Data | Linear Regression")
