import matplotlib.pyplot as plt

# Eksempeldata – erstatt disse med dine egne verdier
# Målesituasjon 1
teoretisk_sakte = [0.53, 0.49, 0.52, 0.51,0.50]
målt_sakte = [0.54,0.54,0.53,0.54,0.53]

# Målesituasjon 2
teoretisk_fort = [1.60, 1.79, 1.55, 1.61, 1.70]
målt_fort = [1.69, 1.70,1.72,1.69,1.72]

# Målesituasjon 3
teoretisk_bak = [0.85, 0.83, 0.90, 0.88, 0.84]
målt_bak = [0.91,0.91,0.91,0.92,0.92]

# Indekser for målingene
x_akse = [1, 2, 3, 4, 5]

plt.figure(figsize=(10, 6))

# Plotter teoretiske og målte verdier som stiplete linjer
plt.plot(x_akse, teoretisk_sakte, 'o--', label='Teoretisk - Lav hastighet fram')
plt.plot(x_akse, målt_sakte, 'o--', label='Målt - Lav hastighet fram')

plt.plot(x_akse, teoretisk_fort, 's--', label='Teoretisk - Høy hastighet fram')
plt.plot(x_akse, målt_fort, 's--', label='Målt - Høy hastighet fram')

plt.plot(x_akse, teoretisk_bak, 'd--', label='Teoretisk - Bakover')
plt.plot(x_akse, målt_bak, 'd--', label='Målt - Bakover')

# Aksetitler og tittel
plt.xlabel('Måling')
plt.ylabel('Hastighet (m/s)')
plt.title('Teoretisk og målt hastighet - alle målesituasjoner')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

