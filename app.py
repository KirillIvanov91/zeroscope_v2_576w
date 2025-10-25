




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
import torch, uuid
from pathlib import Path

# Проверка GPU
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name}, VRAM: {vram_gb:.2f} GB")
else:
    vram_gb = 0
    print("⚠️ GPU не обнаружен — будет использован CPU")

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

        num_frames = 24 if vram_gb >= 6 else 8

        # Генерация видео
        output = pipev2_576w(
            req.prompt,
            num_inference_steps=10,
            height=320,
            width=576,
            num_frames=num_frames
        )

        if not hasattr(output, "frames") or len(output.frames) == 0:
            raise RuntimeError("⚠️ Не удалось сгенерировать видео (frames отсутствуют)")

        print("✅ Генерация успешна!")

        # Сохранение видео
        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = output_dir / filename
        print(f"💾 Сохранение видео: {filename}")

        export_to_video(output.frames, str(output_path), fps=8)

        print(f"✅ Видео сохранено: {output_path}")

        return {"video_path": str(output_path)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Запрос поступил, но генерация не началась: {str(e)}"
        )


@app.get("/")
async def read_root():
    return {"message": "Сервис генерации видео запущен! Используйте эндпоинт /generate."}




#docker run --gpus all -p 8080:8080 -v ${PWD}:/app zeroscope-server
