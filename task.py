import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


df = pd.read_csv("movies.csv")
print(df.head())


print(df.info())
print(df.describe())

print("Missing Values:\n", df.isnull().sum())

sns.countplot(x='Success', data=df)
plt.title("Hit vs Flop Movies")
plt.show()


sns.histplot(df['Budget'], kde=True)
plt.title("Budget Distribution")
plt.show()


le_genre = LabelEncoder()
df['Genre'] = le_genre.fit_transform(df['Genre'])

le_target = LabelEncoder()
df['Success'] = le_target.fit_transform(df['Success'])

X = df[['Budget','Actor_Popularity','Screens','Genre']]
y = df['Success']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(12,6))
plot_tree(model, feature_names=['Budget','Actor_Popularity','Screens','Genre'],
          class_names=['Flop','Hit'], filled=True)
plt.show()

new_movie = [[70, 8, 1600, le_genre.transform(['Action'])[0]]]
new_movie_scaled = scaler.transform(new_movie)

result = le_target.inverse_transform(model.predict(new_movie_scaled))
print("Prediction for new movie:", result[0])
