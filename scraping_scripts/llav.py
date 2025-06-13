from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    LLAVA_MODEL_PATH = "liuhaotian/llava-v1.5-7b-quantized",
    model_base=None,
    model_name="llava-v1.5-7b",
    device="cpu"  # or "cuda" if you have GPU
)
print("✅ LLaVA model loaded.")
