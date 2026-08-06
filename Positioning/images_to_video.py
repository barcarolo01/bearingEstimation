import os
import cv2
import re
from pathlib import Path
from typing import List, Tuple

def get_sorted_image_files(folder_path: str) -> List[str]:
    """
    Read all the images in the folder and sort them in ascending numerical order
    """
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith('.png')]

    # Sort by name
    def extract_number(filename: str) -> int:
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')    
    image_files.sort(key=extract_number)

    return image_files

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Legge le dimensioni della prima immagine"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
    height, width = img.shape[:2]
    return width, height

def create_video(
    folder_path: str = "img",
    output_path: str = "output.mp4",
    fps: int = 30,
    codec: str = "mp4v"
) -> None:
    """
    Crea un video MP4 da una sequenza di immagini numerate.
    
    Args:
        folder_path: Percorso della cartella con le immagini (default: "img")
        output_path: Percorso del file video di output (default: "output.mp4")
        fps: Frame per secondo (default: 30)
        codec: Codec video, es. "mp4v", "MJPG", "XVID" (default: "mp4v")
    """
    print(f"📁 Lettura immagini da: {folder_path}")
    
    # Ottieni lista ordinata di immagini
    image_files = get_sorted_image_files(folder_path)
    print(f"✅ Trovate {len(image_files)} immagini")
    
    # Ottieni dimensioni dalla prima immagine
    first_image_path = os.path.join(folder_path, image_files[0])
    width, height = get_image_dimensions(first_image_path)
    print(f"📐 Dimensioni: {width}x{height}")
    
    # Crea VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Impossibile creare il video writer con codec {codec}")
    
    print(f"🎬 Creazione video a {fps} FPS con codec {codec}...")
    
    # Scrivi ogni immagine nel video
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"⚠️  Impossibile leggere: {filename}")
            continue
        
        # Ridimensiona se necessario
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        print(f"   [{i}/{len(image_files)}] {filename}")
    
    # Chiudi il writer
    out.release()
    
    # Verifica il file creato
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✨ Video creato con successo!")
        print(f"📄 File: {output_path}")
        print(f"💾 Dimensione: {file_size_mb:.2f} MB")
    else:
        raise RuntimeError("Errore nella creazione del video")

if __name__ == "__main__":
    # CONFIGURAZIONE
    FOLDER = "img"           # Cartella con le immagini
    OUTPUT = "output.mp4"    # File video di output
    FPS = 10                 # Frame per secondo
    CODEC = "mp4v"           # Codec: "mp4v" (MP4), "MJPG" (Motion JPEG), "XVID"
    
    try:
        create_video(
            folder_path=FOLDER,
            output_path=OUTPUT,
            fps=FPS,
            codec=CODEC
        )
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        exit(1)
