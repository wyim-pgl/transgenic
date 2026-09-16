# src/transgenic/utils/huggingface_integration.py

- getModel · function · L39-L148 — def getModel(config, safetensors_model=None, device="cpu", mode="predict")
- getLargeDecoderModel · function · L151-L222 — def getLargeDecoderModel(config, safetensors_model=None, device="cpu", mode="predict")
- getPeftModel · function · L225-L392 — def getPeftModel(decoder_checkpoint, segment_checkpoint, config=None, unlink=False, safetensors_model=None, device="cpu", mode="predict")
- registerModel · function · L395-L433 — def registerModel(hub_name, model)
