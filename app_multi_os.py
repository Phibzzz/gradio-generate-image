import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import warnings
import platform
warnings.filterwarnings("ignore")

# Détection automatique du setup machine et du device optimal
def detect_best_device():
    system = platform.system().lower()
    
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU"
    elif torch.backends.mps.is_available() and system == "darwin":
        return "mps", "Apple Silicon GPU"
    else:
        return "cpu", "CPU"

# Configuration
DEVICE, DEVICE_TYPE = detect_best_device()
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

print(f"Système détecté: {platform.system()} {platform.machine()}")
print(f"Device optimal: {DEVICE} ({DEVICE_TYPE})")
print(f"Modèle: {MODEL_ID}")

# Chargement du modèle selon le setup machine
print(f"Chargement du modèle {MODEL_ID} sur {DEVICE}...")
try:
    if DEVICE == "cuda":
        # Configuration CUDA (Windows/Linux avec NVIDIA)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_memory_efficient_attention'):
            pipe.enable_memory_efficient_attention()
        print("Modèle chargé avec succès sur CUDA !")
        
    elif DEVICE == "mps":
        # Configuration MPS (macOS Apple Silicon)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        print("Modèle chargé avec succès sur Apple Silicon !")
        
    else:
        # Configuration CPU (tous les systèmes)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(DEVICE)
        print("Modèle chargé avec succès sur CPU !")

except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    print("Vérifiez votre connexion internet et les dépendances")
    exit(1)

def generate_image(prompt, progress=gr.Progress()):
    try:
        if not prompt.strip():
            return None, "Veuillez saisir un prompt."
        
        progress(0, desc="Initialisation...")
        print(f"Génération de l'image pour: {prompt[:50]}...")
        
        progress(0.1, desc="Génération en cours...")
        
        # Configuration selon le device
        if DEVICE == "cuda":
            # CUDA utilise torch.autocast
            with torch.autocast(DEVICE):
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                )
        elif DEVICE == "mps":
            # MPS: Pas de torch.autocast (peut causer des problèmes)
            result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            )
        else:
            # CPU: Pas de torch.autocast
            result = pipe(
                prompt=prompt,
                num_inference_steps=10,  # Moins d'étapes sur CPU
                guidance_scale=7.5,
                width=256,  # Et résolution réduite sur CPU pour gagner en perf...
                height=256
            )
        
        progress(0.9, desc="Finalisation...")
        image = result.images[0]
        
        progress(1.0, desc="Terminé!")
        print("Image générée avec succès!")
        return image, "Image générée avec succès!"
    
    except Exception as e:
        error_msg = f"Erreur: {e}"
        print(error_msg)
        return None, error_msg

# Titre de l'interface selon le device
def get_interface_title():
    if DEVICE == "cuda":
        return "Générateur d'Images IA (NVIDIA GPU)"
    elif DEVICE == "mps":
        return "Générateur d'Images IA (Apple Silicon)"
    else:
        return "Générateur d'Images IA (CPU)"

def get_resolution_info():
    if DEVICE == "cpu":
        return "Résolution: 256x256 | Étapes: 10 | Device: CPU"
    else:
        return "Résolution: 512x512 | Étapes: 20 | Device: " + DEVICE.upper()

# Interface Gradio
with gr.Blocks(
    title=get_interface_title(),
    theme=gr.themes.Soft()
) as demo:
    
    gr.HTML(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>{get_interface_title()}</h1>
        <p>Créez des images à partir de descriptions textuelles</p>
        <p style="color: #666; font-size: 0.9em;">{get_resolution_info()}</p>
        <p style="color: #888; font-size: 0.8em;">Système: {platform.system()} {platform.machine()}</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Décrivez l'image que vous souhaitez générer",
                placeholder="Ex: A beautiful sunset over a mountain landscape...",
                lines=3,
                max_lines=5
            )
            
            generate_btn = gr.Button(
                "🎨 Générer l'Image",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Image générée",
                type="pil",
                height=500
            )
            
            info_text = gr.Textbox(
                label="Statut",
                interactive=False,
                lines=2
            )

    # Connexion des événements
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, info_text]
    )
    
    prompt_input.submit(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, info_text]
    )

# Lancement de l'application
if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    print("Interface disponible sur: http://localhost:7860")
    print(f"Modèle: {MODEL_ID}")
    print(f"Device: {DEVICE} ({DEVICE_TYPE})")
    print(f"Plateforme: {platform.system()} {platform.machine()}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
