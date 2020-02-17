import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array(x).reshape(-1,1)
result=np.array(result).reshape(-1,1)
model = LinearRegression(fit_intercept=True, normalize=False) 
model.fit(x, result)
print("Coefficients: ",model.coef_)
print("Intercept: ",model.intercept_)

print(model.predict([[2070]]))

plt.scatter(x,result)
plt.plot(x ,model.predict(x) ,color='red',linewidth =3)