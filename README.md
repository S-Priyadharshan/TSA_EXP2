# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 26/08/2025
Name: Priyadharshan S

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/co2_gr_mlo.csv", comment="#")

df.head()
df.tail()

years = df['year'].tolist()
co2 = df['ann inc'].tolist()

X = [i - years[len(years) // 2] for i in years]

x2 = [i**2 for i in X]
xy = [i*j for i, j in zip(X, co2)]
n = len(years)

b = (n * sum(xy) - sum(co2) * sum(X)) / (n * sum(x2) - (sum(X)**2))
a = (sum(co2) - b * sum(X)) / n

linear_trend = [a + b * X[i] for i in range(n)]

x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i*j for i, j in zip(x2, co2)]

coeff = [
    [len(X), sum(X), sum(x2)],
    [sum(X), sum(x2), sum(x3)],
    [sum(x2), sum(x3), sum(x4)]
]
Y = [sum(co2), sum(xy), sum(x2y)]

A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution

poly_trend = [a_poly + b_poly*X[i] + c_poly*(X[i]**2) for i in range(n)]

print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

df['Linear Trend'] = linear_trend
df['Polynomial Trend'] = poly_trend

plt.figure(figsize=(10, 5))
plt.plot(years, co2, 'bo-', label="Real CO₂")
plt.plot(years, linear_trend, 'k--', label="Linear Trend")
plt.title("CO₂ Levels with Linear Trend")
plt.xlabel("Year")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(years, co2, 'bo-', label="Real CO₂")
plt.plot(years, poly_trend, 'k-', label="Polynomial Trend (Quadratic)")
plt.title("CO₂ Levels with Polynomial Trend")
plt.xlabel("Year")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/f3c2863a-5a5c-4af8-a91c-6bb53ca4014b" />

B- POLYNOMIAL TREND ESTIMATION
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/e8131c36-ad4d-4e58-bcb6-ddb4088124fd" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
