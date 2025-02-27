from transformers import AutoTokenizer
from together import Together
import os
from dotenv import load_dotenv
load_dotenv()

# Clé API pour Together AI
TOGETHER_API_KEY = os.getenv("API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

# Fonction pour lire un fichier texte
def lire_fichier(chemin):
    with open(chemin, "r", encoding="utf-8") as f:
        return f.read()

# Charger les fichiers spécifiques
fichier_uk = "Le code de l'Éthique de Tunisie.txt"
fichier_usa = "Le code d'ethique de USA.txt"

code_uk = lire_fichier(fichier_uk)
code_usa = lire_fichier(fichier_usa)

# Générer le rapport via l'API Together LLM
prompt = f"""
Analyse comparative des codes d'éthique.

Texte 1 :
{code_uk}

Texte 2 :
{code_usa}

Génère un rapport court en français sous forme de points avec :
- Points similaires
- Différences principales
- Améliorations possibles
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
    messages=[
        {"role": "system", "content": "Tu es un assistant expert en analyse comparative des codes d'éthiques des ingénieurs."},
        {"role": "user", "content": prompt},
    ],
    stream=False
)

# Extraire et afficher le rapport
generated_report = response.choices[0].message.content
print("\nRapport généré :\n")
print(generated_report)
