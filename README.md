# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 25/5/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv("DailyDelhiClimate.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Sort values by date
data = data.sort_values('date')

# Select mean temperature column
X = data['meantemp']

# Plot original data
plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title('Original Temperature Data')
plt.xlabel('Days')
plt.ylabel('Mean Temperature')
plt.show()

# ACF and PACF plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# -----------------------------
# ARMA(1,1) Model
# -----------------------------

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Model Summary")
print(arma11_model.summary())

# Extract AR and MA coefficients
phi1 = arma11_model.arparams[0]
theta1 = arma11_model.maparams[0]

# Define AR and MA arrays
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

# Generate ARMA(1,1) sample
N = 1000

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot simulated ARMA(1,1)
plt.figure(figsize=(12, 6))

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Temperature Data')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

# ACF Plot
plot_acf(ARMA_1)

plt.title('ACF of ARMA(1,1)')

plt.show()

# PACF Plot
plot_pacf(ARMA_1)

plt.title('PACF of ARMA(1,1)')

plt.show()

# -----------------------------
# ARMA(2,2) Model
# -----------------------------

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Model Summary")
print(arma22_model.summary())

# Extract coefficients
phi1_2 = arma22_model.arparams[0]
phi2_2 = arma22_model.arparams[1]

theta1_2 = arma22_model.maparams[0]
theta2_2 = arma22_model.maparams[1]

# Define AR and MA arrays
ar2 = np.array([1, -phi1_2, -phi2_2])
ma2 = np.array([1, theta1_2, theta2_2])

# Generate ARMA(2,2) sample
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

# Plot simulated ARMA(2,2)
plt.figure(figsize=(12, 6))

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Temperature Data')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

# ACF Plot
plot_acf(ARMA_2)

plt.title('ACF of ARMA(2,2)')

plt.show()

# PACF Plot
plot_pacf(ARMA_2)

plt.title('PACF of ARMA(2,2)')

plt.show()

```

OUTPUT:
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/2b574fe9-50bf-483e-9aac-f000d5715617" />

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/fe1772af-18be-42f3-b693-325985ea294e" />
<img width="1025" height="547" alt="image" src="https://github.com/user-attachments/assets/a36a2c1d-3dcd-4698-8e8b-a18c9c9c0d49" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/38882aff-482d-422a-a511-c7afc53b2002" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/eb8e17ed-d007-4f16-9e17-3a68bf98d3a4" />



RESULT:
Thus, a python program is created to fir ARMA Model successfully.
