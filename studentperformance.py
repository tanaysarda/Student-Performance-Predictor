import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Hours':[1,2,3,4,5], 'Scores':[20,40,60,80,100]}
df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

print("Predicted score for 6 hrs:", model.predict([[6]])[0])
