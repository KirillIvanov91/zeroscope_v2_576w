



import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
import torch, uuid
from pathlib import Path
import numpy as np
from PIL import Image


# Проверка GPU
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name}, VRAM: {vram_gb:.2f} GB")
    
else:
    vram_gb = 0
    print("⚠️ GPU не обнаружен — будет использован CPU")
# === Автоматический выбор режима по VRAM ===
def get_generation_settings(vram_gb: float):
    """Подбирает оптимальные параметры генерации под видеокарту."""
    if vram_gb < 6:
        mode = "fast"
        settings = dict(height=320, width=576, num_frames=16, num_inference_steps=20)
    elif vram_gb < 10:
        mode = "balanced"
        settings = dict(height=448, width=768, num_frames=24, num_inference_steps=25)
    else:
        mode = "quality"
        settings = dict(height=576, width=1024, num_frames=32, num_inference_steps=30)

    print(f"🧠 Режим: {mode.upper()} | {settings['height']}x{settings['width']} | "
          f"Кадров: {settings['num_frames']} | Шагов: {settings['num_inference_steps']}")
    return settings
# Инициализация FastAPI
app = FastAPI()

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("🔄 Загружается модель...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipev2_576w = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=dtype
    ).to(device)

    # Оптимизация памяти
    try:
        pipev2_576w.enable_model_cpu_offload()
    except Exception as e:
        print(f"⚠️ CPU offload недоступен: {e}")

    # Включаем xFormers, если хватает VRAM
    if torch.cuda.is_available() and vram_gb >= 6:
        try:
            pipev2_576w.enable_xformers_memory_efficient_attention()
            print("✅ Memory-efficient attention включен")
        except Exception as e:
            print(f"⚠️ Не удалось включить xFormers: {e}")
    else:
        print("⚠️ Слишком мало VRAM — xFormers не включен")

    print("✅ Модель готова к работе!")

except Exception as exc:
    raise RuntimeError(f"Ошибка загрузки модели: {exc}")


class VideoRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate(req: VideoRequest):
    try:
        print(f"🎬 Генерация: {req.prompt}")

        settings = get_generation_settings(vram_gb)

        start_time = time.time()  # старт таймера

        output = pipev2_576w(
            req.prompt,
            num_inference_steps=settings["num_inference_steps"],
            height=settings["height"],
            width=settings["width"],
            num_frames=settings["num_frames"]
        )

        if not hasattr(output, "frames") or output.frames is None:
            raise RuntimeError("⚠️ Не удалось сгенерировать видео (frames отсутствуют)")

        frames = output.frames
        processed_frames = []

        # --- Обработка кадров (как раньше) ---
        if isinstance(frames, np.ndarray):
            if frames.ndim == 5 and frames.shape[0] == 1:
                frames = frames[0]
            if frames.ndim == 4 and frames.shape[-1] == 3:
                for arr in frames:
                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                    processed_frames.append(Image.fromarray(arr))
        elif isinstance(frames, torch.Tensor):
            frames = frames.squeeze(0).detach().cpu().numpy()
            for arr in frames:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                processed_frames.append(Image.fromarray(arr))
        elif isinstance(frames, list):
            for f in frames:
                if isinstance(f, Image.Image):
                    processed_frames.append(f.convert("RGB"))
                elif isinstance(f, (np.ndarray, torch.Tensor)):
                    arr = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f
                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                    processed_frames.append(Image.fromarray(arr))

        # --- Сохраняем видео ---
        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = output_dir / filename
        export_to_video(processed_frames, str(output_path), fps=8)

        # --- Логируем время и FPS ---
        elapsed = time.time() - start_time
        fps = len(processed_frames) / elapsed if elapsed > 0 else 0
        print(f"⏱ Время генерации: {elapsed:.2f} сек | FPS: {fps:.2f}")

        return {"video_path": str(output_path), "generation_time_sec": elapsed, "fps": fps}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Запрос поступил, но генерация не началась: {str(e)}"
        )



#docker run --gpus all -p 8080:8080 -v ${PWD}:/app zeroscope-server
