"""Cache the core DeiT-S model and its pretrained preprocessing metadata."""

from mita_probe.runtime import configure_runtime

configure_runtime()

import timm


if __name__ == "__main__":
    model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=True)
    print(f"Cached DeiT-S: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(timm.data.resolve_model_data_config(model))
