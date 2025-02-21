import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyse_univarie(data, var_quant, var_qual):
    """cette fonction prend en entrée une base de donnée et la liste des variables quantitative, et la liste des variable qualitative et 
    retourne les statistique descriptives

    Args:
        data (pd.dataframe): base de données
        var_quant (list): liste des variables quantitatives
        var_qual (list): liste des variables qualitatives
    
 
    """
    print ("Description des variables quantitatives:")

    for var in var_quant:
        print()
        print(f"Description de la variable {var}")
        print()
        print(f"- La moyenne est: {data[var].mean():.2f}")
        print(f"- La médiane est: {data[var].median():.2f}")
        print(f"- L'écart-type  est: {data[var].std():.2f}")
        print(f"- Le premier quantile est: {data[var].quantile(0.25):.2f}")
        print(f"- Le troisieme quantile   est: {data[var].quantile(0.75):.2f}")
        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.boxplot(x=data[var], ax=axes[0], color="skyblue")
        sns.histplot(data[var], bins=30, kde=True, ax=axes[1], color="gray")
        sns.kdeplot(data[var], fill=True, ax=axes[2], color="green")

        axes[0].set_title(f"Boxplot de {var}")
        axes[1].set_title(f"Histogramme de {var}")
        axes[2].set_title(f"Densité de {var}")

        plt.tight_layout()
        plt.show()
    
    for var in var_qual:
        print()
        print(f'Description de la variable {var}:')
        print(f"La valeur la plus observée est {data[var].mode()[0]}")
        print(f"Le nombre de modalité: {data[var].nunique()}")
        print(f"Effectif par modalité: {data[var].value_counts()}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(x=data[var], order=data[var].value_counts().index, ax=axes[0], palette="pastel")
        data[var].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1], colors=sns.color_palette("pastel"))

        axes[0].set_title(f"Histogramme de {var}")
        axes[1].set_title(f"Camembert de {var}")
        axes[1].set_ylabel("")  # Cacher le label y du camembert

        plt.tight_layout()
        plt.show()
    