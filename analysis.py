import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##########################################
# Outdated use ImprovedAnalysis.py instead
##########################################

datadir_early = './early1.dat'
datadir_late = './late1.dat'

early_data = pd.read_csv(datadir_early, sep='\s+', header=0, names=['mag', 'emag', 'col', 'ecol'])
late_data = pd.read_csv(datadir_late, sep='\s+', header=0, names=['mag', 'emag', 'col', 'ecol'])

def clean_data(data):
    data['mag'] = pd.to_numeric(data['mag'], errors='coerce')
    data['emag'] = pd.to_numeric(data['emag'], errors='coerce')
    data['col'] = pd.to_numeric(data['col'], errors='coerce')
    data['ecol'] = pd.to_numeric(data['ecol'], errors='coerce')
    return data

early_data = clean_data(early_data)
late_data = clean_data(late_data)

early_data = early_data[early_data['mag'] < 24]
late_data = late_data[late_data['mag'] < 24]

# Linear fit for early data with errors
early_fit, early_cov_matrix = np.polyfit(early_data['mag'], early_data['col'], deg=1, cov=True)
early_p1, early_p2 = early_fit
early_p1_err, early_p2_err = np.sqrt(np.diag(early_cov_matrix))

# Linear fit for late data with errors
late_fit, late_cov_matrix = np.polyfit(late_data['mag'], late_data['col'], deg=1, cov=True)
late_p1, late_p2 = late_fit
late_p1_err, late_p2_err = np.sqrt(np.diag(late_cov_matrix))

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
plt.errorbar(early_data['mag'], early_data['col'], xerr=early_data['emag'], yerr=early_data['ecol'],
             color='darkred', fmt=' ', ecolor='red', elinewidth=1, capsize=2, label='Early Type')
x_early = np.linspace(early_data['mag'].min(), early_data['mag'].max(), 500)
plt.plot(x_early, early_p1 * x_early + early_p2, '-r', lw=2, alpha=0.7, label='Early Fit')

# Plot late data and fit
plt.errorbar(late_data['mag'], late_data['col'], xerr=late_data['emag'], yerr=late_data['ecol'],
             color='darkblue', fmt=' ', ecolor='blue', elinewidth=1, capsize=2, label='Late Type')
x_late = np.linspace(late_data['mag'].min(), late_data['mag'].max(), 500)
plt.plot(x_late, late_p1 * x_late + late_p2, '-b', lw=2, alpha=0.7, label='Late Fit')

plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


# Calculate predicted (V-I) for I=21
I_21 = 21
predicted_color = early_p1 * I_21 + early_p2
predicted_color_err = np.sqrt((I_21 * early_p1_err)**2 + early_p2_err**2)

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
print(f"Predicted color (V-I) for I=21: {predicted_color:.2f}")
print(f"Observed color (V-I): {observed_color:.2f} ± {observed_uncertainty}")
print(f"Difference in color (Δ(V-I)): {delta_color:.2f}")
print(f"Galaxies are {trend} than prediction.")
print(f"Look-back time to Cl0016+16: {lookback_time:.2f} Gyr")


# Solve for apparent magnitude (I) with error
V_I_target = 2.4  # Target color
I_apparent = (V_I_target - early_p2) / early_p1
I_apparent_err = np.sqrt(
    (1 / early_p1)**2 * early_p2_err**2 +
    ((V_I_target - early_p2) / early_p1**2)**2 * early_p1_err**2
)
# Check for reasonable early_p1 and uncertainties
if early_p1 <= 0 or early_p1_err / early_p1 > 0.1:
    print("Warning: Large relative uncertainty in slope (early_p1). Results may be unreliable.")

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