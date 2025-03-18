import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

X_train = train_data.drop("meal", axis=1)
y_train = train_data["meal"]
X_test = test_data.drop("meal", axis=1)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

modelFit = model.fit(X_train, y_train)

pred = modelFit.predict(X_test)
pred = [int(i) for i in pred]