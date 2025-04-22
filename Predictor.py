import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
#pip install xgboost==2.0.3 --no-deps
import xgboost
#model = xgboost.Booster()
#model.load_model('XGB.json')
df2 =pd.read_csv('x_train.csv')
x_train = df2[['AGE', 'BMI', 'WAIST', 'Hypertension', 'EDU']]

model = joblib.load('XGB.pkl')

feature_names = ['AGE', 'BMI', 'WAIST', 'Hypertension', 'EDU']
    
    
# 设置 Streamlit 应用的标题
st.title("CircS diagnostic model")
st.sidebar.header("Selection Panel") # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")
AGE = st.number_input("AGE", min_value=0, max_value=120, value=1)
Height = st.number_input("Height", min_value=0, max_value=250, value=1)
Weight = st.number_input("Weight", min_value=0, max_value=150, value=1)
WAIST = st.selectbox("WAIST", options=[0, 1], format_func=lambda x:"YES"if x == 1 else "NO")
Hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x:"YES"if x == 1 else "NO")
EDU = st.selectbox("EDU", options=[0, 1], format_func=lambda x:"YES"if x == 1 else "NO")





BMI = Weight*10000/Height/Height


feature_values = [AGE, BMI, WAIST, Hypertension, EDU]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: Low risk of CircS, 1: High risk of CircS)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice =(
            f"According to our model, you have a high risk of CircS. "
            f"The model predicts that your probability of having CircS is {probability:.1f}%."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice =(
            f"According to our model, you have a low risk of CircS. "
            f"The model predicts that your probability of not having CircS is {probability:.1f}%."
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    print(explainer_shap.expected_value)
    if predicted_class == 1:
       shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    #解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=x_train.values, 
        feature_names=x_train.columns.tolist(),
        class_names=['Low risk of CircS', 'High risk of CircS'],# Adjust class names to match your classification task
        mode='classification'
    )

    #Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=13
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=True) # Disable feature value table
    st.components.v1.html(lime_html, height=800,scrolling=True)
