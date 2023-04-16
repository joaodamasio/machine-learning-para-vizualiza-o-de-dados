import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

base = pd.read_excel("Investimento_x_Venda.xlsx")
print(base.head())

#vizualização das informações
plt.scatter(base["Investimento em marketing"],base['Venda Qtd'])
#plt.show()

reg = linear_model.LinearRegression()
reg.fit(base["Investimento em marketing"].values.reshape(-1,1),base["Venda Qtd"])

reg.coef_
reg.intercept_

plt.scatter(base["Investimento em marketing"],base["Venda Qtd"])
plt.scatter(75,reg.predict([[75]])[0],color="k")
x = np.array(base["Investimento em marketing"])
y = reg.intercept_ + x*reg.coef_
plt.plot(x,y,"r")
print(plt.show())
print(reg.predict([[75]]))