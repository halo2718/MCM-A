from sklearn.preprocessing import PolynomialFeatures

print(x)
print(result)
deg2 = PolynomialFeatures(degree = 2)  ## degree = 2
model2 = LinearRegression()
model2.fit(deg2.fit_transform(x),result)
print('Cofficients:',model2.coef_)
print("Intercept: ",model2.intercept_)

print(model2.predict(deg2.fit_transform([[2070],[2100]])))

plt.scatter(x,result)
plt.plot(x,model2.predict(deg2.fit_transform(x)),color='red')
