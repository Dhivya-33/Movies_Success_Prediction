import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

st.title(" Movie Success Prediction (Decision Tree)")
st.write("Dataset: movies.csv must be in the same folder.")

# Load the dataset
df = pd.read_csv("movies.csv")
st.subheader(" Dataset Preview")
st.dataframe(df.head())

# =======================
# 1. EDA
# =======================
st.subheader(" Exploratory Data Analysis")

# Success Count
fig1 = plt.figure()
sns.countplot(data=df, x='Success')
plt.title("Hit vs Flop")
st.pyplot(fig1)

# Budget distribution
fig2 = plt.figure()
sns.histplot(df['Budget'], kde=True)
plt.title("Budget Distribution")
st.pyplot(fig2)

# =======================
# 2. Label Encoding
# =======================
st.subheader(" Label Encoding")

le_genre = LabelEncoder()
df['Genre'] = le_genre.fit_transform(df['Genre'])

le_target = LabelEncoder()
df['Success'] = le_target.fit_transform(df['Success'])

# =======================
# 3. Feature Scaling
# =======================
st.subheader("📏 Feature Scaling")

features = ['Budget', 'Actor_Popularity', 'Screens', 'Genre']
X = df[features]
y = df['Success']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write(pd.DataFrame(X_scaled, columns=features).head())

# =======================
# 4. Train-test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =======================
# 5. Train Decision Tree
# =======================
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {accuracy:.2f}")

# =======================
# 6. Decision Tree Visualization
# =======================
st.subheader(" Decision Tree Visualization")

fig3 = plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=features, class_names=['Flop', 'Hit'], filled=True)
st.pyplot(fig3)

# =======================
# 7. Prediction Section
# =======================
st.subheader(" Predict New Movie")

budget = st.slider("Budget (in crores)", 10, 200, 60)
actor_pop = st.slider("Actor Popularity (1 to 10)", 1, 10, 8)
screens = st.slider("Screens", 100, 3000, 1500)
genre = st.selectbox("Genre", le_genre.classes_)

new_data = [[
    budget,
    actor_pop,
    screens,
    le_genre.transform([genre])[0]
]]

new_data_scaled = scaler.transform(new_data)

if st.button("Predict"):
    pred = model.predict(new_data_scaled)[0]
    result = le_target.inverse_transform([pred])[0]
    st.info(f"🎬 Prediction: **{result}**")
