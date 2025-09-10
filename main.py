#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import qi
import clip
import torch
from PIL import Image
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------------------
# Funções utilitárias
# ---------------------------

def setup_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"nao_captures_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_image(image: Image.Image, output_dir, iteration, caption="", top_score=0, time=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nao_image_{iteration:04d}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath, quality=95)
    if caption:
        txt_filename = f"nao_image_{iteration:04d}_{timestamp}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Caption (CLIP): {caption}\n")
            f.write(f"Score: {top_score}\n")
            f.write(f"Tempo: {time} s\n")
    return filepath

def setup_live_display():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_title("NAO Camera Feed", fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    return fig, ax1, ax2

def update_live_display(fig, ax1, ax2, image, caption, iteration):
    ax1.clear(); ax2.clear()
    ax1.imshow(image); ax1.axis('off')
    ax1.set_title(f"NAO Camera - Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}", fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, f"Caption (CLIP): {caption}", horizontalalignment='center', verticalalignment='center',
             fontsize=12, wrap=True, transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.axis('off'); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    plt.tight_layout()
    fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.1)

def connect_to_nao(ip="172.15.1.29", port=9559):
    session = qi.Session()
    session.connect(f"tcp://{ip}:{port}")
    return session

def capture_image_from_nao(session):
    try:
        camera_service = session.service("ALVideoDevice")
        camera_id = 0; resolution = 2; color_space = 11; fps = 5
        try:
            video_client = camera_service.subscribeCamera("python_client", camera_id, resolution, color_space, fps)
        except AttributeError:
            try:
                video_client = camera_service.subscribe("python_client", resolution, color_space, fps)
            except AttributeError:
                video_client = "python_client"
                camera_service.setActiveCamera(camera_id)
                camera_service.setResolution(video_client, resolution)
                camera_service.setColorSpace(video_client, color_space)
                camera_service.setFrameRate(video_client, fps)
        time.sleep(0.1)
        nao_image = camera_service.getImageRemote(video_client)
        if nao_image is None or len(nao_image) < 7:
            raise Exception("Falha ao capturar imagem da câmera")
        width = nao_image[0]; height = nao_image[1]; channels = nao_image[2]
        image_data = nao_image[6]
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
        image = Image.fromarray(image_array).convert("RGB")
        return image
    except Exception as e:
        return capture_image_alternative(session)
    finally:
        try:
            if 'video_client' in locals() and hasattr(camera_service, 'unsubscribe'):
                camera_service.unsubscribe(video_client)
        except:
            pass

def capture_image_alternative(session):
    try:
        photo_service = session.service("ALPhotoCapture")
        photo_service.setResolution(2)
        photo_service.setPictureFormat("jpg")
        temp_path = "/tmp/nao_temp_image.jpg"
        photo_service.takePicture(temp_path)
        image = Image.open(temp_path).convert("RGB")
        return image
    except:
        return Image.new('RGB', (640, 480), color='blue')

def speak_text(session, text):
    try:
        tts_service = session.service("ALTextToSpeech")
        try:
            tts_service.setLanguage("English")
        except:
            try:
                tts_service.setLanguage("en-US")
            except:
                pass
        try:
            tts_service.setVolume(0.1)
        except:
            pass
        tts_service.say(text)
    except:
        pass

# ---------------------------
# CLIP pipeline
# ---------------------------

def load_clip(device=None, model_name="ViT-B/32"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device

def clip_caption(image: Image.Image, model, preprocess, device, candidate_texts, top_k=3):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(candidate_texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T).squeeze(0)
        probs = logits.softmax(dim=0).cpu().numpy()
    indices = np.argsort(-probs)[:top_k]
    results = [(candidate_texts[i], float(probs[i])) for i in indices]
    return results

DEFAULT_CANDIDATES = [
    "a person", "a child", "an adult", "a man", "a woman", "a robot", "a NAO robot",
    "a ball", "a chair", "a table", "a bottle", "a laptop", "a phone", "a door",
    "a window", "a cup", "a book", "a bag", "a plant", "a cat", "a dog",
    "a bicycle", "a car", "a screen", "a keyboard", "a mouse", "a monitor",
    "an object", "no person", "outdoor scene", "indoor scene", "a person walking",
    "a person sitting", "people", "multiple people"
]

# ---------------------------
# Main
# ---------------------------

def main():
    print("=== Sistema de Visão NAO com CLIP (zero-shot) ===")

    output_dir = setup_output_directory()

    try:
        session = connect_to_nao("172.15.1.29", 9559)
    except Exception as e:
        print(f"Erro ao conectar: {e}")
        return

    try:
        model, preprocess, device = load_clip()
    except Exception as e:
        print(f"Erro ao carregar CLIP: {e}")
        return

    # Display interativo
    fig, ax1, ax2 = setup_live_display()
    iteration = 1

    try:
        while True:
            image = capture_image_from_nao(session)
            if image is None:
                time.sleep(5)
                iteration += 1
                continue

            # Processamento CLIP com medição de tempo
            start_time = time.time()
            try:
                top = clip_caption(image, model, preprocess, device, DEFAULT_CANDIDATES, top_k=1)
                top_caption = top[0][0]
                top_score = top[0][1]
            except:
                top_caption = "Error"
                top_score = 0.0
            elapsed = time.time() - start_time

            # Imprime na CLI
            print(f"Iteração {iteration}: {top_caption} | Score: {top_score*1000:.2f} | Tempo: {elapsed:.2f}s")

            # Salva imagem
            save_image(image, output_dir, iteration, top_caption, top_score*1000, elapsed)

            # Atualiza display em tempo real
            update_live_display(fig, ax1, ax2, image, f"{top_caption} ({top_score:.2f})", iteration)

            # Fala a legenda
            speak_text(session, top_caption)

            iteration += 1
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    finally:
        plt.close('all')
        print(f"Programa finalizado! Imagens salvas em: {output_dir}")

if __name__ == "__main__":
    main()
