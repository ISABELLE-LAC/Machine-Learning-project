import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kruskal, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction pour calculer le V de Cramer
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

# Fonction principale pour l'analyse bivariée
def analyse_bivariee(df, target, quantitative_vars, qualitative_vars):
    # 1. Corrélation entre variables quantitatives et la variable cible (Spearman)
    print("Corrélation de Spearman avec la variable cible:")
    for var in quantitative_vars:
        coef, p = spearmanr(df[var], df[target])
        print(f"{var} -> {target} : Coefficient = {coef:.3f}, p-value = {p:.4f}")
    print("\n")

    # 2. Test de Kruskal-Wallis entre variables qualitatives et la variable cible
    print("Test de Kruskal-Wallis entre les variables qualitatives et la variable cible:")
    for var in qualitative_vars:
        groups = [df[target][df[var] == modality] for modality in df[var].unique()]
        stat, p = kruskal(*groups)
        print(f"{var} -> {target} : Statistique = {stat:.3f}, p-value = {p:.4f}")
    print("\n")

    # 3. Corrélation entre les variables quantitatives (Spearman)
    print("Corrélation de Spearman entre les variables quantitatives:")
    corr_spearman = df[quantitative_vars].corr(method='spearman')
    print(corr_spearman)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Corrélation de Spearman entre les variables quantitatives")
    plt.show()

    # 4. V de Cramer entre les variables qualitatives
    print("V de Cramer entre les variables qualitatives:")
    for i in range(len(qualitative_vars)):
        for j in range(i + 1, len(qualitative_vars)):
            v_cramer = cramers_v(df[qualitative_vars[i]], df[qualitative_vars[j]])
            print(f"{qualitative_vars[i]} <-> {qualitative_vars[j]} : V de Cramer = {v_cramer:.3f}")

