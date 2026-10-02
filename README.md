# Générateur d'images IA avec Gradio

Application web locale qui génère une image à partir d'une description textuelle (un *prompt*).
Elle combine trois briques :

| Brique | Rôle |
|---|---|
| **Stable Diffusion v1.5** (`sd-legacy/stable-diffusion-v1-5`) | Modèle de diffusion texte → image, téléchargé depuis le Hugging Face Hub |
| **diffusers** (Hugging Face) | Bibliothèque qui charge le modèle et exécute la génération |
| **Gradio** | Construit l'interface web (zone de texte, bouton, affichage de l'image) en quelques lignes de Python |

Le projet contient deux versions de l'application :

- [app.py](app.py) : version de base, pour GPU NVIDIA (CUDA) avec repli sur le CPU.
- [app_multi_os.py](app_multi_os.py) : version qui détecte la machine : GPU NVIDIA (CUDA), Apple Silicon (MPS) ou CPU. Elle adapte la précision, la résolution et le nombre d'étapes au matériel.

---

## Sommaire

1. [Prérequis](#1-prérequis)
2. [Démarrage rapide](#2-démarrage-rapide)
3. [Utiliser l'application](#3-utiliser-lapplication)
4. [Reproduire le développement depuis zéro](#4-reproduire-le-développement-depuis-zéro)
5. [Personnaliser la génération](#5-personnaliser-la-génération)
6. [Dépannage](#6-dépannage)
7. [Structure du projet](#7-structure-du-projet)

---

## 1. Prérequis

| Élément | Recommandé | Remarque |
|---|---|---|
| Python | 3.10 à 3.13 | Testé avec Python 3.13 |
| GPU | NVIDIA avec au moins 6 Go de VRAM | Testé sur une RTX 3060 Ti (8 Go). Sans GPU, ça fonctionne sur le CPU, mais lentement (plusieurs minutes par image) |
| Pilote NVIDIA | Récent, compatible CUDA ≥ 12.9 | Vérifiez avec `nvidia-smi` |
| Espace disque | ~10 Go | ~4,5 Go pour PyTorch et les dépendances, ~4 Go pour le modèle |
| Connexion internet | Requise au premier lancement | Le modèle est téléchargé une seule fois puis mis en cache |

> Sur Mac Apple Silicon (M1/M2/M3…), utilisez `app_multi_os.py`.

---

## 2. Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone <url-du-repo> gradio-generate-image
cd gradio-generate-image

# 2. Créer l'environnement virtuel
python -m venv .venv

# 3. L'activer
#    Windows (PowerShell) :
.venv\Scripts\Activate.ps1
#    Windows (cmd) :
.venv\Scripts\activate.bat
#    macOS / Linux :
source .venv/bin/activate

# 4. Installer les dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Lancer l'application
python app.py            # ou : python app_multi_os.py
```

Ouvrez ensuite **http://localhost:7860** dans votre navigateur.

Le premier lancement télécharge le modèle (environ 4 Go). Les lancements suivants réutilisent le cache local de Hugging Face (`~/.cache/huggingface/hub`, ou `C:\Users\<vous>\.cache\huggingface\hub` sous Windows).

---

## 3. Utiliser l'application

1. Saisissez une description dans le champ **« Décrivez l'image que vous souhaitez générer »**.
   Écrivez de préférence **en anglais** : Stable Diffusion 1.5 a été entraîné sur des légendes en anglais.
2. Cliquez sur **« Générer l'Image »** ou appuyez sur **Entrée**.
3. Une barre de progression s'affiche. L'image apparaît à droite et le champ **Statut** indique le résultat.
4. Pour enregistrer l'image, utilisez l'icône de téléchargement en haut à droite de l'image.

### Conseils pour écrire un prompt

- Décrivez le **sujet**, puis le **contexte**, puis le **style** :
  `a red fox sitting in a snowy forest, golden hour lighting, digital art, highly detailed`
- Les mots-clés de style ont un effet visible : `oil painting`, `watercolor`, `photorealistic`, `35mm photo`, `anime style`, `isometric 3D render`…
- Chaque clic produit une image différente : la graine aléatoire (*seed*) n'est pas fixée.

### Performances indicatives

| Matériel | Résolution | Étapes | Temps par image |
|---|---|---|---|
| GPU NVIDIA (RTX 3060 Ti, 8 Go) | 512×512 | 20 | ~4 s (mesuré) |
| Apple Silicon (`app_multi_os.py`) | 512×512 | 20 | ~10 à 30 s |
| CPU (`app_multi_os.py`) | 256×256 | 10 | ~1 à 3 min |
| CPU (`app.py`) | 512×512 | 20 | plusieurs minutes |

### Accès depuis un autre appareil

L'application écoute sur `0.0.0.0:7860`. Depuis un autre appareil du même réseau local, ouvrez `http://<IP-de-votre-machine>:7860`. Il faut parfois autoriser le port dans le pare-feu.

---

## 4. Reproduire le développement depuis zéro

Cette section reprend la construction du projet pas à pas, en expliquant chaque choix.

### Étape 1 : créer le dossier et l'environnement virtuel

```bash
mkdir gradio-generate-image
cd gradio-generate-image
git init
python -m venv .venv
.venv\Scripts\Activate.ps1      # macOS/Linux : source .venv/bin/activate
python -m pip install --upgrade pip
```

**Pourquoi un environnement virtuel ?** Le projet dépend de versions précises de bibliothèques lourdes (PyTorch, diffusers…). Le venv les isole du Python système et des autres projets. Le dossier `.venv/` est exclu de git par le [.gitignore](.gitignore).

### Étape 2 : installer PyTorch avec le bon support matériel

PyTorch est le moteur de calcul. **C'est l'étape la plus délicate**, parce que la version à installer dépend du matériel :

```bash
# GPU NVIDIA (Windows / Linux), CUDA 12.9 :
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129

# macOS (Apple Silicon) ou CPU uniquement :
pip install torch==2.8.0 torchvision==0.23.0
```

**Pourquoi un index spécifique ?** Les versions de PyTorch compilées pour CUDA (suffixe `+cu129`, `+cu128`…) **ne sont pas publiées sur PyPI**. On les trouve uniquement sur l'index de PyTorch (`download.pytorch.org`). Sous Windows, le `torch` de PyPI est une version **CPU uniquement**. Pour choisir la commande adaptée à votre machine, utilisez le sélecteur de https://pytorch.org/get-started/locally/.

Vérification :

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Attendu sur GPU NVIDIA : 2.8.0+cu129 True
```

### Étape 3 : installer les bibliothèques de l'application

```bash
pip install diffusers transformers accelerate gradio
```

| Paquet | Pourquoi |
|---|---|
| `diffusers` | Fournit `StableDiffusionPipeline`, qui enchaîne l'encodeur de texte, le U-Net de débruitage, le planificateur (*scheduler*) et le décodeur VAE |
| `transformers` | Fournit l'encodeur de texte CLIP utilisé par Stable Diffusion pour comprendre le prompt |
| `accelerate` | Charge le modèle plus vite et avec moins de mémoire, et gère le paramètre `device_map` |
| `gradio` | Génère l'interface web et le serveur HTTP |

### Étape 4 : écrire l'application ([app.py](app.py))

Le fichier est organisé en quatre blocs.

#### 4.1 Configuration et choix du device

```python
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
```

- `DEVICE` choisit le GPU s'il est disponible, sinon le CPU.
- `MODEL_ID` est l'identifiant du modèle sur le Hugging Face Hub. `sd-legacy/stable-diffusion-v1-5` est le miroir officiel de Stable Diffusion 1.5 depuis le retrait du dépôt `runwayml`.
- `warnings.filterwarnings("ignore")` masque les avertissements de dépréciation des bibliothèques, pour garder une console lisible en démonstration.

#### 4.2 Chargement du modèle (une seule fois, au démarrage)

```python
if DEVICE == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # demi-précision : 2x moins de VRAM, plus rapide
        device_map="cuda",           # charge directement les poids sur le GPU
        safety_checker=None,         # désactive le filtre NSFW (plus léger et plus rapide)
        requires_safety_checker=False
    )
    pipe.enable_attention_slicing()  # calcule l'attention par morceaux : moins de VRAM
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # le CPU ne gère pas bien le float16
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(DEVICE)
```

Points clés :

- **Le modèle est chargé au niveau du module**, et non dans la fonction de génération. Le chargement prend de quelques secondes à plusieurs minutes. On ne le fait donc qu'une fois, au démarrage, et toutes les requêtes réutilisent le même `pipe`.
- `from_pretrained` télécharge les poids au premier appel puis les lit depuis le cache.
- **float16 sur GPU / float32 sur CPU** : la demi-précision divise la mémoire par deux sur GPU. Sur CPU, elle est lente ou mal supportée.
- `enable_attention_slicing()` réduit le pic de mémoire GPU au prix d'une légère perte de vitesse. C'est utile sur des cartes de 6 à 8 Go.
- Le code teste aussi `enable_memory_efficient_attention` avec `hasattr`. Cette méthode n'existe pas sur le pipeline (le nom réel est `enable_xformers_memory_efficient_attention`, qui demande le paquet `xformers`). Le test est donc toujours faux et sans effet. Depuis PyTorch 2, diffusers utilise de toute façon par défaut l'attention optimisée de PyTorch (SDPA).
- En cas d'erreur (pas d'internet, dépendance manquante), le script affiche un message et s'arrête avec `exit(1)`.

#### 4.3 La fonction de génération

```python
def generate_image(prompt, progress=gr.Progress()):
    try:
        if not prompt.strip():
            return None, "Veuillez saisir un prompt."

        progress(0.1, desc="Génération en cours...")
        with torch.autocast(DEVICE):
            result = pipe(
                prompt=prompt,
                num_inference_steps=20,  # nombre d'étapes de débruitage
                guidance_scale=7.5,      # fidélité au prompt
                width=512,
                height=512
            )
        progress(1.0, desc="Terminé!")
        return result.images[0], "Image générée avec succès!"
    except Exception as e:
        return None, f"Erreur: {e}"
```

- Gradio appelle cette fonction à chaque clic. **Ses paramètres correspondent aux composants d'entrée et ses valeurs de retour aux composants de sortie**, dans le même ordre : ici `(image, texte de statut)`.
- `progress=gr.Progress()` est un paramètre spécial : Gradio l'injecte automatiquement et affiche une barre de progression dans l'interface.
- `torch.autocast` laisse PyTorch choisir automatiquement la précision des opérations.
- Les paramètres de génération sont détaillés en [section 5](#5-personnaliser-la-génération).
- Le `try/except` renvoie l'erreur dans l'interface au lieu de faire planter le serveur.

#### 4.4 L'interface Gradio et le lancement

```python
with gr.Blocks(title="Générateur d'Images IA", theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1>Générateur d'Images IA</h1>")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Décrivez l'image...", lines=3)
            generate_btn = gr.Button("Générer l'Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Image générée", type="pil", height=500)
            info_text = gr.Textbox(label="Statut", interactive=False)

    generate_btn.click(fn=generate_image, inputs=[prompt_input], outputs=[output_image, info_text])
    prompt_input.submit(fn=generate_image, inputs=[prompt_input], outputs=[output_image, info_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

- `gr.Blocks` est l'API de mise en page flexible de Gradio (plus souple que `gr.Interface`). Les blocs `with` imbriqués décrivent la disposition : une ligne (`Row`) qui contient deux colonnes (`Column`).
- `type="pil"` : la fonction renvoie une image PIL, ce que fournit directement le pipeline diffusers.
- **Événements** : `.click()` (bouton) et `.submit()` (touche Entrée dans le champ de texte) relient les composants à la fonction Python.
- `demo.launch(...)` :
  - `server_name="0.0.0.0"` : accessible depuis le réseau local (utilisez `"127.0.0.1"` pour un accès limité à votre machine) ;
  - `server_port=7860` : port par défaut de Gradio ;
  - `share=True` créerait un lien public temporaire `*.gradio.live` (pratique pour une démo, mais l'application devient accessible à tous).

### Étape 5 : version multi-plateforme ([app_multi_os.py](app_multi_os.py))

Cette version reprend la même structure et ajoute :

1. **Une détection du matériel** avec `detect_best_device()` : CUDA si un GPU NVIDIA est présent, sinon **MPS** (le GPU des puces Apple Silicon, via `torch.backends.mps`) sous macOS, sinon CPU.
2. **Un chargement adapté** : float16 sur CUDA et MPS, float32 sur CPU. Sur MPS, on utilise `pipe.to("mps")` plutôt que `device_map`.
3. **Une génération adaptée** :
   - `torch.autocast` seulement sur CUDA (il pose des problèmes sur MPS) ;
   - sur CPU, une **résolution de 256×256 et 10 étapes** pour garder un temps d'attente acceptable.
4. **Une interface informative** : le titre et le sous-titre indiquent le matériel utilisé, la résolution et le système.

### Étape 6 : figer les dépendances

```bash
pip freeze > requirements.txt
```

> ⚠️ **Piège à éviter** : `pip freeze` écrit `torch==2.8.0+cu129` mais **n'enregistre pas l'index d'où vient ce paquet**. Quelqu'un qui clone le dépôt obtient alors l'erreur suivante :
>
> ```
> ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu129
> ```
>
> Après chaque `pip freeze`, corrigez le fichier :
>
> 1. Ajoutez en tête : `--extra-index-url https://download.pytorch.org/whl/cu129`
> 2. Retirez le suffixe local : `torch==2.8.0` et `torchvision==0.23.0`
>
> Avec cette configuration, pip prend la version CUDA (`2.8.0+cu129`) lorsqu'elle existe pour la plateforme (Windows et Linux). Sur macOS, où elle n'existe pas, il prend la version standard de PyPI. Le même `requirements.txt` fonctionne donc partout.

### Étape 7 : versionner

```bash
git add .gitignore app.py app_multi_os.py requirements.txt README.md
git commit -m "Générateur d'images Gradio + Stable Diffusion"
```

Le [.gitignore](.gitignore) exclut déjà l'environnement virtuel, les caches Python, les poids de modèles (`*.safetensors`, `*.ckpt`) et les images générées.

---

## 5. Personnaliser la génération

Les paramètres se trouvent dans l'appel `pipe(...)` de `generate_image` :

| Paramètre | Valeur actuelle | Effet |
|---|---|---|
| `num_inference_steps` | 20 (10 sur CPU en multi-OS) | Nombre d'étapes de débruitage. Plus d'étapes donnent plus de détails mais plus de temps. Au-delà de 30 à 50, le gain est faible |
| `guidance_scale` | 7.5 | Fidélité au prompt. Une valeur basse (3 à 5) donne plus de créativité, une valeur haute (10 à 15) un résultat plus littéral, avec un risque de saturation |
| `width` / `height` | 512 (256 sur CPU en multi-OS) | Taille de l'image, en multiples de 8. SD 1.5 est entraîné en 512×512 : au-delà, les sujets ont tendance à se dupliquer |

Pistes d'évolution :

- **Prompt négatif** : passez `negative_prompt="blurry, low quality, deformed"` à `pipe(...)` pour écarter des défauts.
- **Résultats reproductibles** : passez `generator=torch.Generator(DEVICE).manual_seed(42)` pour obtenir la même image à prompt identique.
- **Réglages dans l'interface** : ajoutez des `gr.Slider` (étapes, guidance) et ajoutez-les aux `inputs` des événements et aux paramètres de `generate_image`.
- **Autre modèle** : changez `MODEL_ID`, par exemple pour un modèle SDXL avec `StableDiffusionXLPipeline`. Prévoyez plus de VRAM.

---

## 6. Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `Could not find a version that satisfies the requirement torch==2.8.0+cu129` | Les versions CUDA de PyTorch ne sont pas sur PyPI | Utilisez le `requirements.txt` corrigé (ligne `--extra-index-url`) ; voir [étape 6](#étape-6--figer-les-dépendances) |
| `ERROR: Ignored the following versions that require a different python version: ...` | Simple information de pip sur d'anciennes versions de numpy incompatibles avec votre Python | Sans conséquence, à ignorer |
| `torch.cuda.is_available()` renvoie `False` alors qu'il y a un GPU NVIDIA | PyTorch CPU installé (version sans `+cu…`) ou pilote trop ancien | `pip uninstall torch torchvision` puis réinstallez avec l'index CUDA ([étape 2](#étape-2--installer-pytorch-avec-le-bon-support-matériel)). Mettez à jour le pilote NVIDIA |
| `CUDA out of memory` | VRAM insuffisante | Réduisez `width`/`height`, fermez les autres applications qui utilisent le GPU, ou remplacez `device_map="cuda"` par `pipe.enable_model_cpu_offload()` |
| Image entièrement noire | Instabilité numérique en float16 (rare) | Relancez la génération ou passez en `torch.float32` |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Politique d'exécution de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou activez le venv depuis `cmd` |
| `Erreur lors du chargement du modèle` au démarrage | Pas de connexion au premier lancement, ou téléchargement interrompu | Vérifiez la connexion et relancez. En cas de cache corrompu, supprimez `~/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5` |
| `OSError: [Errno 10048]` / port déjà utilisé | Une autre instance tourne sur le port 7860 | Fermez-la ou changez `server_port` |
| Génération très lente | L'application tourne sur le CPU | Vérifiez la ligne `Device:` affichée au démarrage. Sur CPU, préférez `app_multi_os.py` |

---

## 7. Structure du projet

```
gradio-generate-image/
├── app.py              # Application de base (CUDA, repli sur le CPU)
├── app_multi_os.py     # Application multi-plateforme (CUDA / MPS / CPU)
├── requirements.txt    # Dépendances figées (+ index PyTorch CUDA)
├── .gitignore
└── README.md
```
