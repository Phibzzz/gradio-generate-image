import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

# Chargement du modèle selon la documentation officielle
print(f"Chargement du modèle {MODEL_ID} sur {DEVICE}...")
try:
    if DEVICE == "cuda":
        # Configuration optimisée pour CUDA
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            safety_checker=None,
            requires_safety_checker=False
        )
        # Optimisations CUDA
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_memory_efficient_attention'):
            pipe.enable_memory_efficient_attention()
    else:
        # Configuration pour CPU
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(DEVICE)

    print("Modèle chargé avec succès!")
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
        with torch.autocast(DEVICE):
            result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
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

# Interface Gradio
with gr.Blocks(
    title="Générateur d'Images IA",
    theme=gr.themes.Soft()
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Générateur d'Images IA</h1>
        <p>Créez des images à partir d'une description textuelle</p>
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
                "Générer l'Image",
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
    print(f"Interface disponible sur: http://localhost:7860")
    print(f"Modèle: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
