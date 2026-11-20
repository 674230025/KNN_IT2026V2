from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease | KNN Classifier",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fb 0%, #eef2f8 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .hero {
        padding: 30px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(110deg, #172554, #2563eb);
        margin-bottom: 22px;
    }
    .hero h1 { color: white; margin-bottom: 8px; }
    .hero p { color: #dbeafe; margin-bottom: 0; font-size: 1.05rem; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #172554;
        margin: 10px 0 14px 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf3;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }
    .note {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1e3a8a;
        margin: 10px 0 18px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# อ่านข้อมูลจากไฟล์ Heart3.csv
# โครงสร้างโฟลเดอร์: project/data/Heart3.csv
# ---------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "Heart3.csv"

FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope"
]
TARGET = "HeartDisease"

st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Classification</h1>
    <p>ระบบจำแนกข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("ไม่พบไฟล์ data/Heart3.csv กรุณาวางไฟล์ไว้ในโฟลเดอร์ data")
    st.stop()

dt = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in dt.columns]

if missing_columns:
    st.error("ไฟล์ CSV ขาดคอลัมน์: " + ", ".join(missing_columns))
    st.stop()

if dt[required_columns].isnull().any().any():
    st.error("พบค่าว่างในคอลัมน์ที่ใช้ฝึกโมเดล กรุณาตรวจสอบไฟล์ Heart3.csv")
    st.stop()

# ใช้ข้อมูลทุกคอลัมน์เป็นตัวแปรอิสระ ยกเว้น HeartDisease
X = dt[FEATURES]
y = dt[TARGET]

if y.nunique() != 2:
    st.error("คอลัมน์ HeartDisease ต้องมี 2 คลาส คือ 0 และ 1")
    st.stop()


# ---------------------------------------------------------
# แถบข้อมูลภาพรวม
# ---------------------------------------------------------
st.markdown('<div class="section-title">ภาพรวมชุดข้อมูล</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนข้อมูลทั้งหมด", f"{len(dt):,} แถว")
c2.metric("จำนวนคุณลักษณะ", len(FEATURES))
c3.metric("ไม่พบโรคหัวใจ (0)", f"{int((y == 0).sum()):,} รายการ")
c4.metric("พบโรคหัวใจ (1)", f"{int((y == 1).sum()):,} รายการ")

with st.expander("ดูตัวอย่างข้อมูล Heart3.csv"):
    st.dataframe(dt.head(10), use_container_width=True)

with st.expander("ดูสถิติข้อมูล"):
    st.dataframe(dt[FEATURES].describe().T, use_container_width=True)

st.markdown('<div class="section-title">การกระจายของผลลัพธ์ HeartDisease</div>', unsafe_allow_html=True)
class_counts = y.value_counts().sort_index()
class_chart = pd.DataFrame({
    "จำนวนข้อมูล": class_counts.values
}, index=["0 - ไม่พบโรคหัวใจ", "1 - พบโรคหัวใจ"])
st.bar_chart(class_chart)


# ---------------------------------------------------------
# ฝึกและประเมินโมเดล KNN
# ใช้ StandardScaler เพราะ KNN คำนวณระยะทาง
# ---------------------------------------------------------
st.markdown('<div class="section-title">ตั้งค่าและประเมินโมเดล KNN</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("ตั้งค่าโมเดล")
    k_value = st.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=15, value=5, step=2)
    test_size = st.slider("สัดส่วนข้อมูลทดสอบ (%)", min_value=10, max_value=40, value=20, step=5)
    st.caption("โมเดลใช้ StandardScaler เพื่อปรับสเกลข้อมูลก่อนคำนวณระยะทาง")
    st.markdown("---")
    st.image("img/nm.jpg", width=115)
    

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k_value))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
m2.metric("ข้อมูลฝึกสอน", f"{len(X_train):,} แถว")
m3.metric("ข้อมูลทดสอบ", f"{len(X_test):,} แถว")

with st.expander("ดู Confusion Matrix และรายงานผล"):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["HeartDisease 0", "HeartDisease 1"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)


# ---------------------------------------------------------
# แบบฟอร์มป้อนข้อมูลใหม่เพื่อทำนาย
# ---------------------------------------------------------
st.markdown('<div class="section-title">ป้อนข้อมูลชุดใหม่เพื่อทำนาย</div>', unsafe_allow_html=True)

st.markdown("""
<div class="note">
กรอกค่าคุณลักษณะทั้ง 11 รายการ จากนั้นกดปุ่มทำนายผล
คอลัมน์ที่เป็นรหัสตัวเลขให้เลือกค่ารหัสตามที่ปรากฏในชุดข้อมูล
</div>
""", unsafe_allow_html=True)

# คอลัมน์ที่เก็บเป็นรหัสตัวเลขในไฟล์ต้นฉบับ
coded_columns = [
    "Sex", "ChestPainType", "FastingBS",
    "RestingECG", "ExerciseAngina", "ST_Slope"
]

# แบ่งแบบฟอร์มเป็นสองคอลัมน์เพื่อให้อ่านง่าย
with st.form("prediction_form"):
    left, right = st.columns(2)
    new_data = {}

    with left:
        st.subheader("ข้อมูลทั่วไป")
        for col in ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS"]:
            if col in coded_columns:
                options = sorted(dt[col].dropna().unique().tolist())
                new_data[col] = st.selectbox(
                    col,
                    options=options,
                    format_func=lambda value: str(int(value))
                )
            else:
                min_value = int(dt[col].min())
                max_value = int(dt[col].max())
                default_value = int(dt[col].median())
                new_data[col] = st.number_input(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=1
                )

    with right:
        st.subheader("ข้อมูลการตรวจ")
        for col in ["RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]:
            if col in coded_columns:
                options = sorted(dt[col].dropna().unique().tolist())
                new_data[col] = st.selectbox(
                    col,
                    options=options,
                    format_func=lambda value: str(int(value))
                )
            else:
                min_value = float(dt[col].min())
                max_value = float(dt[col].max())
                default_value = float(dt[col].median())
                step_value = 0.1 if col == "Oldpeak" else 1.0
                new_data[col] = st.number_input(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=step_value,
                    format="%.1f" if col == "Oldpeak" else "%.0f"
                )

    submitted = st.form_submit_button("🔍 ทำนายผลข้อมูลใหม่", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([new_data], columns=FEATURES)
    prediction = int(model.predict(input_df)[0])

    st.markdown("---")
    st.subheader("ผลการจำแนก")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_index = list(model.named_steps["knn"].classes_).index(prediction)
        confidence = probabilities[class_index] * 100
    else:
        confidence = None

    if prediction == 1:
        st.error("ผลการทำนาย: HeartDisease = 1 — โมเดลจำแนกอยู่ในกลุ่มพบโรคหัวใจ")
    else:
        st.success("ผลการทำนาย: HeartDisease = 0 — โมเดลจำแนกอยู่ในกลุ่มไม่พบโรคหัวใจ")

    if confidence is not None:
        st.metric("ความน่าจะเป็นของคลาสที่ทำนาย", f"{confidence:.2f}%")

    with st.expander("ตรวจสอบข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(input_df, use_container_width=True)

st.caption(
    "หมายเหตุ: ผลลัพธ์เป็นการจำแนกจากโมเดล Machine Learning เพื่อการศึกษา "
    "ไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์"
)
