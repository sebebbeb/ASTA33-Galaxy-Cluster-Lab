import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
datadir_early = './early1.dat'
datadir_late = './late1.dat'

early_data = pd.read_csv(datadir_early, sep='\s+', header=0, names=['mag', 'emag', 'col', 'ecol'])
late_data = pd.read_csv(datadir_late, sep='\s+', header=0, names=['mag', 'emag', 'col', 'ecol'])

def clean_data(data):
    data['mag'] = pd.to_numeric(data['mag'], errors='coerce')
    data['emag'] = pd.to_numeric(data['emag'], errors='coerce')
    data['col'] = pd.to_numeric(data['col'], errors='coerce')
    data['ecol'] = pd.to_numeric(data['ecol'], errors='coerce')
    data = data.dropna()
    return data

early_data = clean_data(early_data)
late_data = clean_data(late_data)

# Filter out faint galaxies
def remove_outliers(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    return data[(data[column] > mean - threshold * std) & (data[column] < mean + threshold * std)]

early_data = early_data[early_data['mag'] < 24]
late_data = late_data[late_data['mag'] < 24]

early_data = remove_outliers(early_data, 'col')
early_data = remove_outliers(early_data, 'mag')
late_data = remove_outliers(late_data, 'col')
late_data = remove_outliers(late_data, 'mag')

# Perform weighted linear regression
def weighted_linregress(x, y, x_err, y_err):
    weights = 1 / (x_err**2 + y_err**2)
    weighted_mean_x = np.sum(weights * x) / np.sum(weights)
    weighted_mean_y = np.sum(weights * y) / np.sum(weights)
    
    slope = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y)) / np.sum(weights * (x - weighted_mean_x)**2)
    intercept = weighted_mean_y - slope * weighted_mean_x
    
    slope_uncertainty = np.sqrt(1 / np.sum(weights * (x - weighted_mean_x)**2))
    intercept_uncertainty = slope_uncertainty * np.sqrt(np.mean(x**2))

    return slope, intercept, slope_uncertainty, intercept_uncertainty

# Early data fit
early_slope, early_intercept, early_slope_err, early_intercept_err = weighted_linregress(
    early_data['mag'], early_data['col'], early_data['emag'], early_data['ecol']
)

# Late data fit
late_slope, late_intercept, late_slope_err, late_intercept_err = weighted_linregress(
    late_data['mag'], late_data['col'], late_data['emag'], late_data['ecol']
)

# Plot
plt.style.use('classic')
fig = plt.figure(figsize=(8, 6))
plt.minorticks_on()
plt.gca().tick_params(axis='both', which='both', direction='in', labelsize=14)
plt.ylabel(r'${\rm Colour,}\ (V-I)$', fontsize=18)
plt.xlabel(r'${\rm Magnitude,}\ I$', fontsize=18)
plt.ylim(0, 4)
plt.xlim(18, 26)

# Plot early data and fit
plt.errorbar(
    early_data['mag'], early_data['col'], xerr=early_data['emag'], yerr=early_data['ecol'],
    color='darkred', fmt=' ', ecolor='red', elinewidth=1, capsize=2, label='Early Type'
)
x_early = np.linspace(early_data['mag'].min(), early_data['mag'].max(), 500)
plt.plot(x_early, early_slope * x_early + early_intercept, '-r', lw=2, alpha=0.7, label='Early Fit')

# Plot late data and fit
plt.errorbar(
    late_data['mag'], late_data['col'], xerr=late_data['emag'], yerr=late_data['ecol'],
    color='darkblue', fmt=' ', ecolor='blue', elinewidth=1, capsize=2, label='Late Type'
)
x_late = np.linspace(late_data['mag'].min(), late_data['mag'].max(), 500)
plt.plot(x_late, late_slope * x_late + late_intercept, '-b', lw=2, alpha=0.7, label='Late Fit')

plt.title('Colour-Magnitude Diagram of Early and Late Type Galaxies')
plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# Calculate predicted (V-I) for I=21
I_21 = 21
predicted_color = early_slope * I_21 + early_intercept
predicted_color_err = np.sqrt((early_slope_err * I_21)**2 + early_intercept_err**2)

# Reference color and uncertainty
observed_color = 2.68
observed_uncertainty = 0.03

# Difference in color
delta_color = predicted_color - observed_color
delta_color_err = np.sqrt(predicted_color_err**2 + observed_uncertainty**2)

# Determine redder or bluer
trend = "redder" if delta_color > 0 else "bluer"

# Look-back time
reddening_rate = 0.05  # magnitudes per Gyr
lookback_time = delta_color / reddening_rate
lookback_time_err = delta_color_err / reddening_rate

# Output results
print(f"Predicted color (V-I) for I=21: {predicted_color:.2f} ± {predicted_color_err:.2f}")
print(f"Observed color (V-I): {observed_color:.2f} ± {observed_uncertainty}")
print(f"Difference in color (Δ(V-I)): {delta_color:.2f} ± {delta_color_err:.2f}")
print(f"Galaxies are {trend} than prediction.")
print(f"Look-back time to Cl0016+16: {lookback_time:.2f} ± {lookback_time_err:.2f} Gyr")

# Solve for apparent magnitude (I) with error
V_I_target = 2.4
I_apparent = (V_I_target - early_intercept) / early_slope
I_apparent_err = np.sqrt(
    ((1 / early_slope)**2 * early_intercept_err**2) + 
    (((V_I_target - early_intercept) / early_slope**2)**2 * early_slope_err**2)
)

# Distance modulus
absolute_magnitude = -21.3
absolute_magnitude_error = 0.1
distance_modulus = I_apparent - absolute_magnitude
distance_modulus_err = np.sqrt(I_apparent_err**2 + absolute_magnitude_error**2)

# Distance in parsecs
distance_pc = 10**((distance_modulus + 5) / 5)
distance_pc_err = distance_pc * np.log(10) / 5 * distance_modulus_err

# Output results
print(f"Apparent magnitude (I): {I_apparent:.2f} ± {I_apparent_err:.2f}")
print(f"Distance modulus (μ): {distance_modulus:.2f} ± {distance_modulus_err:.2f}")
print(f"Distance to Cl0016+16 (parsecs): {distance_pc:.2e} ± {distance_pc_err:.2e} pc")
