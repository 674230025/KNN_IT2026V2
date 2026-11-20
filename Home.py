from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ")
#st.image("./img/eveiw.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("HeartDisease 0")
   st.image("./img/eveiw.jpg")

with col2:
   st.header("HeartDisease 1")
   st.image("./img/heart1.jpg")

with col3:
   st.header("HeartDisease 2")
   st.image("./img/heart2.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].mean()
dt2 = dt['RestingBP'].mean()
dt3 = dt['Cholesterol'].mean()
dt4 = dt['MaxHR'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age", "RestingBP", "Cholesterol", "MaxHR"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล Age", 20, 100)
pt_wd = st.slider("กรุณาเลือกข้อมูล RestingBP", 0, 250)

sp_len = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
sp_wd = st.number_input("กรุณาเลือกข้อมูล MaxHR")

if st.button("ทำนายผล"):

   X = dt[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
   y = dt['HeartDisease']

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])

   st.write(Knn_model.predict(x_input))
   out = Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./img/eveiw.jpg")
   elif out[0] == 'Versicolor':       
    st.image("./img/heart1.jpg")
   else:
    st.image("./img/heart2.jpg")
else:
    st.write("ไม่ทำนาย")
