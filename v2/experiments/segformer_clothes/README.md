# SegFormer clothes experiment

This experiment uses `mattmdjaga/segformer_b2_clothes` to segment a clothing
category from an input image URL, then saves a binary mask and a highlighted
overlay.

Model card: https://huggingface.co/mattmdjaga/segformer_b2_clothes

Key notes from the model card:

- Fine-tuned SegFormer B2 on the ATR human parsing dataset.
- Label set includes: Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt,
  Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg,
  Left-arm, Right-arm, Bag, Scarf.

## Run

From repo root:

```bash
uv sync --project v2
uv run --project v2 python v2/experiments/segformer_clothes/run.py \
  --image-url "https://example.com/image.jpg" \
  --category tshirt \
  --mask-out /absolute/path/to/mask.png \
  --overlay-out /absolute/path/to/overlay.png
```

Category aliases:

- `tshirt`, `t-shirt`, `tee`, `top`, `shirt`, `blouse` -> `upper-clothes`
- `topwear` -> `upper-clothes`
- `jeans`, `trousers` -> `pants`
- `bottomwear` -> `pants` + `skirt` + `dress`
- `shoe` -> `left-shoe` + `right-shoe`
- `arm` -> `left-arm` + `right-arm`
- `leg` -> `left-leg` + `right-leg`

## Output

- `mask-out`: single-channel PNG mask (255 = target category).
- `overlay-out`: original image dimmed with the target category highlighted.
