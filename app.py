import gradio as gr
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_similarity(image, text, model_name):

    model, preprocess = load_model(model_name)
    # 画像の前処理
    image = preprocess(image).unsqueeze(0).to(device)
    
    # テキストの前処理
    text = clip.tokenize([text]).to(device)
    
    # 類似度の計算
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        similarity = torch.cosine_similarity(image_features, text_features).cpu().numpy()[0]
        
    return similarity

def load_model(model_name):
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

iface = gr.Interface(
    fn=calculate_similarity,
    inputs=[
        gr.Image(type="pil"), 
        gr.Textbox(lines=2, placeholder="A photo of a ..."),
        gr.Radio(["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"], label="モデル選択")
    ],
    outputs="number",
    title="CLIPによる画像とテキストの類似度計算",
    description="類似度を計算したい画像とテキストを入力し，使用するCLIPモデルを選択してください．"
)

iface.launch()