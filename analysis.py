import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

early_p1, early_p2 = np.polyfit(early_data['mag'], early_data['col'], deg=1)
late_p1, late_p2 = np.polyfit(late_data['mag'], late_data['col'], deg=1)

plt.style.use('classic')
fig = plt.figure(figsize=(8, 6))
plt.minorticks_on()
plt.gca().tick_params(axis='both', which='both', direction='in', labelsize=14)
plt.ylabel(r'${\rm Colour,}\ (V-I)$', fontsize=18)
plt.xlabel(r'${\rm Magnitude,}\ I$', fontsize=18)
plt.ylim(0, 4)
plt.xlim(18, 26)

plt.errorbar(early_data['mag'], early_data['col'], xerr=early_data['emag'], yerr=early_data['ecol'],
             color='darkred', fmt=' ', ecolor='red', elinewidth=1, capsize=2, label='Early Type')

plt.errorbar(late_data['mag'], late_data['col'], xerr=late_data['emag'], yerr=late_data['ecol'],
             color='blue', fmt=' ', ecolor='blue', elinewidth=1, capsize=2, label='Late Type')

x_early = np.linspace(early_data['mag'].min(), early_data['mag'].max(), 500)
plt.plot(x_early, early_p1 * x_early + early_p2, '-r', lw=2, alpha=0.7, label='Early Fit')

x_late = np.linspace(late_data['mag'].min(), late_data['mag'].max(), 500)
plt.plot(x_late, late_p1 * x_late + late_p2, '-b', lw=2, alpha=0.7, label='Late Fit')
plt.grid()
plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
