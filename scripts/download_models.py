"""Cache the core DeiT-S model and its pretrained preprocessing metadata."""

from mita_probe.runtime import configure_runtime

configure_runtime()

import timm
from mita_probe.assets import MODEL_REPO, MODEL_REVISION
from mita_probe.models import load_model


if __name__ == "__main__":
    model = load_model()
    print(f"Pinned model: {MODEL_REPO}@{MODEL_REVISION}")
    print(f"Cached DeiT-S: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(timm.data.resolve_model_data_config(model))
