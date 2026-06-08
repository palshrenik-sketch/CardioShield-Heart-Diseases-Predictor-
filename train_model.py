import pandas as pd
df=pd.read_csv('/home/kiit/HEART DISEASES PREDICTION/heart.csv')
df.drop_duplicates(inplace=True)

x=df.drop("target",axis=1)
y=df["target"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)
print("Heart Diseases Prediction:",rf_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,rf_pred)
print("Accuracy:", accuracy)

import pickle
with open("model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Model saved successfully!")