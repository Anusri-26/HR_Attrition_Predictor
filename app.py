import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model and label encoders
model = joblib.load('attrition_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('model_features.pkl')

# Load data
data = pd.read_csv('HR-Employee-Attrition.csv')

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# Title and Description
st.title("🔍 HR Attrition Prediction Dashboard")
st.markdown("""
This dashboard allows you to predict employee attrition based on HR-related attributes.
You can also explore the data and gain insights into employee trends using interactive visualizations.
""")

# Sidebar - User Input
st.sidebar.header("Employee Details Input")

user_input = {}

for feature in feature_names:
    if feature in label_encoders:
        user_input[feature] = st.sidebar.selectbox(f"{feature}", label_encoders[feature].classes_)
    else:
        feature_min = data[feature].min()
        feature_max = data[feature].max()
        user_input[feature] = st.sidebar.number_input(
            f"{feature}", min_value=feature_min, max_value=feature_max,
            value=int((feature_min + feature_max) / 2)
        )

# Prediction
if st.sidebar.button("Predict Attrition"):
    input_df = pd.DataFrame([user_input])

    for column in input_df.columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Attrition: **{'Yes' if prediction == 1 else 'No'}**")

    # Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(prediction_proba, labels=["No", "Yes"], autopct='%1.1f%%',
            colors=['#76c7c0', '#ff6f61'], startangle=90, explode=(0, 0.1))
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar Chart
    st.subheader("🔍 Prediction Probabilities")
    prob_df = pd.DataFrame({'Attrition': ['No', 'Yes'], 'Probability': prediction_proba})
    st.bar_chart(prob_df.set_index('Attrition'))

    if prediction == 1:
        st.warning("⚠️ The employee is likely to leave. Consider engagement strategies.")
    else:
        st.info("✅ The employee is likely to stay.")

# --- Data Insights ---
st.subheader("🔍 Data Insights")

total_employees = len(data)
attrition_yes = len(data[data['Attrition'] == 'Yes'])
attrition_no = len(data[data['Attrition'] == 'No'])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Employees", value=total_employees)
with col2:
    st.metric(label="Employees Left", value=attrition_yes)
with col3:
    st.metric(label="Employees Stayed", value=attrition_no)

# Attrition by Department
department_counts = data['Department'].value_counts()
st.subheader("🔸 Employee Distribution by Department")
fig2, ax2 = plt.subplots()
ax2.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('Set3', len(department_counts)))
ax2.axis('equal')
st.pyplot(fig2)

# Correlation Heatmap
st.subheader("💡 Feature Correlation Heatmap")
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Attrition by YearsAtCompany
st.subheader("📅 Attrition by Years at Company")
attrition_by_years = data.groupby('YearsAtCompany')['Attrition'].value_counts().unstack().fillna(0)
fig4, ax4 = plt.subplots()
attrition_by_years.plot(kind='bar', stacked=True, ax=ax4, color=['#ff6f61', '#76c7c0'])
ax4.set_title('Attrition by Years at Company')
ax4.set_xlabel('Years at Company')
ax4.set_ylabel('Number of Employees')
st.pyplot(fig4)

# --- Additional Visualizations ---

# 💰 Monthly Income vs Attrition (Box Plot)
st.subheader("💰 Monthly Income vs Attrition")
fig5, ax5 = plt.subplots()
sns.boxplot(x='Attrition', y='MonthlyIncome', data=data, ax=ax5)
st.pyplot(fig5)

# 📌 Attrition Rate by Job Role (Bar Chart)
st.subheader("🧑‍💼 Attrition Rate by Job Role")
jobrole_attr = data.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack().fillna(0)['Yes']
fig6, ax6 = plt.subplots()
jobrole_attr.sort_values().plot(kind='barh', color='#ff6f61', ax=ax6)
ax6.set_title("Attrition Rate by Job Role")
st.pyplot(fig6)

# --- Filterable Table ---
st.subheader("📋 Filter Employee Data")
attrition_filter = st.selectbox("Filter by Attrition Status", ['All', 'Yes', 'No'])

if attrition_filter != 'All':
    filtered_data = data[data['Attrition'] == attrition_filter]
else:
    filtered_data = data

st.dataframe(filtered_data)

# --- Conclusion ---
st.markdown("""
The **HR Attrition Predictor** provides a comprehensive tool to predict and analyze employee attrition based on a variety of features.
Use this to understand which employees are at risk of leaving and gain valuable insights into employee behavior trends.
""")





