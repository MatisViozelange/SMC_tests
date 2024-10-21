# import pandas as pd

# # Initialisation d'une liste pour stocker les résultats
# results = []

# # Exemple de tests avec différents paramètres
# for param in range(1, 10):  # Simule différents tests avec des paramètres
#     error = param * 0.1  # Simule un résultat d'erreur
#     performance = 100 - error  # Simule un autre critère de performance
#     # Ajoute les résultats sous forme de dictionnaire
#     results.append({"Paramètre": param, "Erreur": error, "Performance": performance})

# # Création du DataFrame avec les résultats
# df = pd.DataFrame(results)

# # Affichage du tableau de résultats
# print(df)

# df_sorted = df.sort_values(by='Erreur')
# print(df_sorted)

# print("Erreur moyenne:", df['Erreur'].mean())

# df.to_csv('results.csv', index=False)
import pandas as pd
import matplotlib.pyplot as plt

# Initialisation d'une liste pour stocker les résultats
results = []

# Exemple de tests avec différents paramètres
for param in range(1, 10):  # Simule différents tests avec des paramètres
    error = param * 0.1  # Simule un résultat d'erreur
    performance = 100 - error  # Simule un autre critère de performance
    # Ajoute les résultats sous forme de dictionnaire
    results.append({"Paramètre": param, "Erreur": error, "Performance": performance})

# Création du DataFrame avec les résultats
df = pd.DataFrame(results)

# Affichage du tableau dans Jupyter Notebook ou IDE supportant HTML rendering
print(df)

# Générer un graphique pour la performance en fonction du paramètre
plt.figure(figsize=(10, 6))
plt.plot(df['Paramètre'], df['Performance'], marker='o', label="Performance")
plt.plot(df['Paramètre'], df['Erreur'], marker='x', label="Erreur")
plt.xlabel('Paramètre')
plt.ylabel('Valeur')
plt.title('Comparaison des performances et des erreurs')
plt.legend()
plt.grid(True)
plt.show()
