# Deep Research Report on Hugging Face Models for Clothing Color Identification

## Executive summary

Identifying clothing colors at both **coarse** (primary colors) and **fine** (very specific shades) granularity is rarely solved by a single “color classifier.” In practice, the most reliable approach is a **three-stage pipeline**: (a) localize garment pixels, (b) estimate dominant garment color(s) from those pixels, and (c) map raw colors into a chosen naming system (CSS/HTML, ISCC–NBS, Pantone-like palettes, or your own). This pipeline is also the only approach that scales naturally from “red” to “deep oxblood / burgundy / #4B1F2A” without retraining. citeturn53view0turn22view0turn36view0turn35view0

**Most practical recommended pipelines (prioritized):**

- **Production-friendly, open-world garments (best overall):**  
  **`IDEA-Research/grounding-dino-base`** (open-vocabulary detection) → **`facebook/sam-vit-base`** (mask generation from boxes/points) → **pixel-based color extraction + palette mapping**.  
  This combination gives scalable multi-garment, multi-object handling and keeps licensing permissive (Apache-2.0 for both models). citeturn33view0turn35view0turn36view0turn38view0

- **High-quality fashion human parsing (best when “single person fashion photo” holds):**  
  **`fashn-ai/fashn-human-parser`** → pixel-based color extraction + palette mapping.  
  Very convenient because it directly returns semantic masks like `top`, `pants`, `dress`, etc., but it is explicitly optimized for **single-person images** and is resized to a fixed shape; also it inherits the SegFormer license. citeturn22view0turn15search2

- **Fast-ish fixed taxonomy fashion detection (best for “detect clothing items” with known classes):**  
  **`valentinafeve/yolos-fashionpedia`** (46 fashion categories, boxes) → optional refinement with SAM → pixel-based color extraction.  
  This can be simpler than open-vocabulary flows when your garment taxonomy matches Fashionpedia-style labels (e.g., `shirt, blouse`, `pants`, `skirt`, accessories). citeturn41view0turn43view1turn7view0turn36view0

For “very specific shades,” learned color labels (even CLIP-style) generally require heavy prompt engineering and still tend to output **language-like** rather than **colorimetrically grounded** answers; they are best used as a _semantic naming layer_, not as the ground truth color estimator. citeturn29view0turn23view0turn26view0

## Problem framing and selection criteria

Clothing color identification is hard because a “shirt color” is not a single pixel value: garments may be textured, multi-colored, glossy (specular highlights), shaded by folds, and illuminated by mixed lighting. Backgrounds and skin/hair also contaminate naive global color extraction. The most critical capability is therefore **accurate garment pixel isolation** plus a **robust summary statistic** of garment pixels (median/trimmed mean in Lab space, multi-cluster dominant colors) rather than a single global RGB average. citeturn22view0turn11view0turn53view0turn52view0

This report evaluates Hugging Face-hosted models and repos against the attributes you requested:

- **Model type:** classification / detection / segmentation (semantic or mask generation). citeturn22view0turn41view0turn33view0turn36view0
- **Architecture + input size:** from model cards and `preprocessor_config.json` where available. citeturn35view0turn38view0turn31view0turn25view0turn22view0
- **Pretrained data:** declared in model cards or upstream papers/cards where explicitly stated. citeturn29view0turn23view0turn36view0turn22view0
- **Label taxonomy:** garment classes vs free-form prompts; color label sets if present. citeturn22view0turn11view0turn43view1turn32view0
- **Color extraction method:** pixel-based (mask/box → pixels → clusters) vs learned color labels (prompt-based). citeturn53view0turn23view0turn26view0turn29view0
- **Multiple garments per image:** detection + instance masks generally strongest; semantic parsing is best for “one person.” citeturn22view0turn33view0turn36view0turn43view1
- **Lighting/skin/background robustness:** depends strongly on segmentation fidelity and pixel-statistics used. citeturn22view0turn11view0turn36view0
- **Multilingual label support:** primarily relevant to prompt-based models; M-CLIP offers a multilingual text encoder for CLIP-like similarity. citeturn32view0turn29view0
- **Inference speed/latency:** rarely specified; where absent, this report treats it as unspecified and uses parameter counts + input sizes as proxies. citeturn22view0turn36view0turn33view0turn29view0
- **License:** prioritized because SegFormer-derived models are often restricted. citeturn15search2turn22view0turn11view0turn36view0
- **Sample outputs:** based on the model cards’ documented pipeline outputs and configs. citeturn22view0turn33view0turn36view0turn53view0

## Shortlisted Hugging Face models and comparative tables

### Garment localization models (segmentation, mask generation, detection)

| HF repo (shortlisted)               | Model type                                    | Architecture                                                                      | Input size / preprocessing                                                                             | Pretrained / finetune data                                                                           | Label taxonomy (garment granularity)                                                                             | Multiple garments in one image                                           | Robustness notes                                                                                                         | Model size / compute                                  | License                                           | Sample output                                                                                                                          |
| ----------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `fashn-ai/fashn-human-parser`       | Semantic segmentation (human parsing)         | SegFormer-B4 (MIT-B4 encoder + MLP decoder)                                       | Fixed resize to **384×576** (w×h) per card; returns 18-class mask                                      | Finetuned on **proprietary** dataset curated by entity["company","FASHN AI","fashion ai company"] | 18 semantic classes including `top`, `dress`, `skirt`, `pants`, plus body/accessories (IDs 0–17 listed)          | Good for multiple regions on a **single person**; not instance-separated | Explicitly optimized for **single-person fashion/e-commerce photos**; warns small subjects may lose detail due to resize | **64M params**                                        | `nvidia-segformer` and inherits SegFormer license | HF pipeline returns per-class masks (list of dict with `label`, `score`, `mask`) (documented in card) citeturn22view0turn15search2 |
| `mattmdjaga/segformer_b2_clothes`   | Semantic segmentation (human/clothes parsing) | SegFormer-B2                                                                      | `preprocessor_config.json` uses resize `size: 512`; model card upsamples logits to original resolution | Fine-tuned on **ATR dataset** (stated)                                                               | 18 labels (includes `Upper-clothes`, `Pants`, `Dress`, etc.) listed explicitly; evaluation reports per-class IoU | Works well for single person; semantic, not per-instance                 | Robustness not explicitly characterized; typical parsing caveats apply; metrics reported (mIoU 0.69)                     | **27.4M params**                                      | “other” on HF; links to SegFormer license         | Per-pixel class map (argmaxed logits) + label IDs 0–17 listed                                                                          | citeturn11view0turn12view1turn15search2          |
| `valentinafeve/yolos-fashionpedia`  | Object detection                              | YOLOS (`YolosForObjectDetection`)                                                 | Feature extractor resizes to `size=512`, `max_size=1333`; config lists `image_size: [512, 864]`        | Finetuned on `detection-datasets/fashionpedia` (HF dataset)                                          | 46 categories (config `id2label`, includes `shirt, blouse`, `top, t-shirt, sweatshirt`, `pants`, etc.)           | Yes (detections are per instance)                                        | Detection-only (boxes, not masks) unless paired with a segmenter; taxonomy includes many apparel parts/accessories       | Params unspecified on model page; weights repo ~123MB | **MIT**                                           | Boxes + labels + scores (standard object detection output)                                                                             | citeturn41view0turn43view0turn43view1turn7view0 |
| `IDEA-Research/grounding-dino-base` | Zero-shot object detection (open vocabulary)  | Grounding DINO with Swin backbone (config shows Swin stages/heads) + text encoder | Image processor resizes to shortest_edge=800, longest_edge=1333; COCO-format postprocess               | Pretraining data not specified in HF card (see paper); card reports **52.5 AP on COCO zero-shot**    | Open-vocabulary via text prompts (e.g., “t-shirt.” “shirt.”)                                                     | Strong: detects multiple prompted concepts and multiple instances        | Note: prompts should be lowercased and end with a dot (documented)                                                       | **0.2B params**                                       | **Apache-2.0**                                    | Output is list of detected boxes + phrase alignment scores via processor postprocess                                                   | citeturn33view0turn35view0turn35view1            |
| `facebook/sam-vit-base`             | Mask generation (promptable segmentation)     | Segment Anything Model (ViT image encoder + prompt encoder + mask decoder)        | Processor resizes longest edge to 1024 and pads to 1024×1024                                           | Trained on SA-1B: **11M images, 1.1B masks** (stated)                                                | Not a garment taxonomy; segments “objects” from prompts                                                          | Yes: prompt per garment (box/points) or automatic mask generation        | High-quality masks; official model does not support text prompts directly (per card); commonly paired with detectors     | **93.7M params**                                      | **Apache-2.0**                                    | `pred_masks` + `iou_scores`, post-processed to original image size                                                                     | citeturn36view0turn38view0turn38view1            |

**Key observation:** the only models above that immediately produce **clothing-part masks** without extra prompting are the SegFormer human parsers; the most scalable “any garments anywhere” solution is **Grounding DINO + SAM**, but it is heavier and requires prompt design + box/mask postprocessing. citeturn22view0turn33view0turn36view0

### Color inference and naming models (pixel-based extraction vs prompt-based labels)

| HF repo (shortlisted)               | Model type                                              | Architecture                                                             | Input size / preprocessing                                                       | Pretrained / finetune data                                                                                                                                             | Color label taxonomy / granularity                                                             | Color extraction method                                                                                     | Multilingual support                                               | Model size / compute                                       | License                                                    | Sample output                                                                      |
| ----------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------ |
| `adikuma/Colour-Extract`            | Color extraction pipeline                               | KMeans clustering (documented)                                           | Takes a PIL image                                                                | Not applicable (algorithmic)                                                                                                                                           | No fixed names; returns dominant colors as HEX                                                 | Pixel-based KMeans over all image pixels (or over masked crop you provide); also returns “closest to white” | Not applicable (outputs HEX)                                       | Compute depends on image size and K; model itself is light | **Unknown**                                                | Returns `(colors, closest_to_white)` where `colors` is list of HEX                 | citeturn53view0                         |
| `openai/clip-vit-large-patch14`     | Zero-shot image classification / similarity             | ViT-L/14 image encoder + transformer text encoder                        | CLIP feature extractor uses `size=224` (crop/resize)                             | Trained on large-scale public image-caption data; English-oriented; discussed as research model                                                                        | Open vocabulary via prompts (“a burgundy shirt”, “deep teal t-shirt”)                          | Learned label selection via prompt set; best for semantic names, weaker for colorimetry                     | Model card warns language usage should be limited to English       | **0.4B params**                                            | Unspecified on HF model page; model card derived from repo | Similarity scores (`logits_per_image`) over provided prompts                       | citeturn29view0turn31view0             |
| `patrickjohncyh/fashion-clip`       | Zero-shot image classification / embeddings for fashion | CLIPModel (ViT-B/32 style)                                               | `preprocessor_config.json` shows `size=224`                                      | Fine-tuned on an English fashion product dataset (>800K products) described in card                                                                                    | Open vocabulary; tuned to fashion domain (product imagery)                                     | Learned label selection via a curated prompt list of garment+color strings                                  | English (training data and card focus)                             | **0.2B params**                                            | **MIT**                                                    | Similarity scores across labels; intended for product/fashion retrieval            | citeturn23view0turn25view0turn25view1 |
| `Marqo/marqo-fashionSigLIP`         | Zero-shot image classification / embeddings             | SigLIP-based (OpenCLIP-compatible); config indicates ViT-B/16 SigLIP 224 | Processor resizes to 224×224                                                     | Fine-tuned from ViT-B-16-SigLIP; trained with fashion signals incl. categories, style, colors, materials (stated), and benchmarked on multiple public fashion datasets | Open vocabulary via prompts; not a fixed color label set                                       | Learned prompt ranking; can be used to pick best textual color name among many candidates                   | English (model card examples are English; multilingual not stated) | **0.2B params**                                            | **Apache-2.0**                                             | Example computes softmax over prompt set (label probabilities)                     | citeturn26view0turn28view0turn28view2 |
| `M-CLIP/XLM-Roberta-Large-Vit-B-32` | Multilingual text encoder for CLIP-like retrieval       | XLM-RoBERTa-large text encoder aligned to CLIP image encoder             | Works with OpenAI CLIP ViT-B/32 image model (separately loaded per instructions) | Training details deferred to GitHub card; HF card: “extends OpenAI’s English text encoders to multiple languages”                                                      | Open vocabulary in **48 languages** (text side); color names can be provided in many languages | Learned prompt ranking; requires pairing with an image encoder                                              | Explicitly supports 48 languages                                   | Model size not stated on HF card page                      | Unspecified on HF card page                                | Returns multilingual text embeddings; combine with image embeddings for similarity | citeturn32view0turn29view0             |

**Key observation:** For **very specific shades**, the only truly scalable “taxonomy” is a _palette you control_ (e.g., tens of thousands of named shades, or continuous Lab space) and therefore the most rigorous approach is to compute color(s) from garment pixels and then map them into your chosen palette using a perceptual distance metric (e.g., CIEDE2000). citeturn52view0turn53view0turn49search1turn49search4

## Recommended pipelines and integration notes

### Canonical integration flow

```mermaid
flowchart LR
  A[Input image (RGB/sRGB)] --> B[Garment localization]
  B -->|Option 1: Semantic parsing| C1[Human parser mask per class]
  B -->|Option 2: Detection| C2[Boxes per garment]
  C2 --> D[SAM segmentation prompted by boxes/points]
  C1 --> E[Mask cleanup: erode, remove edges, drop skin/hair if needed]
  D --> E
  E --> F[Color sampling in mask: Lab space]
  F --> G[Dominant colors: k-means / GMM / quantiles]
  G --> H[Map to palette: nearest neighbor by ΔE00 or custom rules]
  H --> I[Output: {garment, color_names, hex, proportions, confidence}]
```

This design matches how the shortlisted models are intended to be used: detectors provide boxes, SAM produces masks from prompts, and human parsers produce semantic clothing regions directly. citeturn33view0turn36view0turn22view0turn53view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["human parsing segmentation clothing mask example","Grounding DINO SAM clothing segmentation example","dominant color extraction kmeans palette example"],"num_per_query":1}

### Pipeline recipe A: Human parser → pixel colors (best when single-person fashion images dominate)

Use `fashn-ai/fashn-human-parser` if you want a clean 18-class fashion-oriented mask schema (top/dress/skirt/pants/belt/scarf, etc.) and you can accept its constraints: it is optimized for single-person images, uses fixed resizing, and the authors recommend their own package for exact preprocessing parity. citeturn22view0

**Preprocessing notes:**

- Match the training resize method if you want maximum mask accuracy: their package uses `cv2.INTER_AREA` while the generic Hugging Face pipeline uses PIL LANCZOS (documented). citeturn22view0
- Be explicit about color space: convert any BGR sources (OpenCV) to RGB before feeding to processors; keep everything in sRGB unless you have calibrated camera workflows.

**Postprocessing notes for accurate garment color pixels:**

- **Erode the mask** slightly (e.g., 2–6 px depending on resolution) to avoid boundary bleed from background and skin edges.
- Remove specular highlights with a simple rule in HSV (very high V and low S) or by trimming the top few percent of L\* in Lab.

The output of the model is an 18-class segmentation; you then compute color(s) from each class mask region. citeturn22view0turn52view0

### Pipeline recipe B: Grounding DINO → SAM → pixel colors (best open-world, production-friendly)

This is the most flexible approach for real-world photos with multiple people, complex backgrounds, and arbitrary garment types.

1. **Detect garments using text prompts** with `IDEA-Research/grounding-dino-base`. The model card emphasizes that text queries should be lowercased and end with a dot. citeturn33view0turn35view2
2. **Segment each detected garment** using `facebook/sam-vit-base` by passing the detection box as a prompt (or points for refinement). citeturn36view0turn38view0
3. **Extract color(s)** from each garment mask:
   - Use KMeans (like `adikuma/Colour-Extract`) on masked pixels, or a more stable approach: compute a trimmed-mean/median in Lab and optionally a secondary cluster for multi-color garments. citeturn53view0turn52view0

**Why this works well for multiple garments:** Grounding DINO outputs multiple boxes (instances) for each concept, and SAM can generate a mask per instance; this is closer to instance segmentation than semantic parsing. citeturn33view0turn36view0

### Pipeline recipe C: YOLOS Fashionpedia detection → optional SAM → pixel colors (best fixed taxonomy)

`valentinafeve/yolos-fashionpedia` provides a predefined fashion label set of 46 categories, exposed directly in the configuration (`id2label`) and also described in the model card. citeturn41view0turn43view1

The fine-tune dataset on Hugging Face (`detection-datasets/fashionpedia`) contains **46,781 images** and **342,182 bounding boxes**, and is licensed under **CC BY 4.0** (dataset card). citeturn7view0

Use this pipeline when:

- You want consistent category labels like `shirt, blouse` vs `top, t-shirt, sweatshirt`.
- You want fast, structured outputs with fewer prompt-engineering variables than open-vocabulary detection.

Limitations:

- It is **box-only**; for precise color extraction you either (a) crop within the box and erode the crop margins, or (b) send the box into SAM to obtain a mask. citeturn41view0turn36view0

### Color mapping to standard palettes (CSS/HTML, ISCC–NBS, Pantone-like)

**CSS/HTML named colors.** CSS defines a `<named-color>` concept and a standardized set of named colors (e.g., `red`, `lightseagreen`), which is convenient for web-facing outputs. citeturn49search4turn49search1

**ISCC–NBS-style naming.** If you want a middle ground between “basic colors” and “thousands of proprietary swatches,” ISCC–NBS style systems offer multi-level naming (coarse → more specific) and are commonly explained as a structured naming framework. citeturn49search2turn49search5  
Practical use: map your extracted Lab values to a fixed table of centroids (Level 1/2/3) and return both a coarse and fine name.

**Pantone-like mapping.** entity["company","Pantone","color matching system"] palettes are widely used in design and manufacturing, but they are typically distributed under proprietary licensing; the safest engineering pattern is to treat them as a “user-supplied palette file” (a list of Lab/XYZ/RGB centroids + names) rather than hardcoding. (Licensing terms vary by palette source; treat as user-provided data.)

**Recommended mapping metric:** use perceptual distance in Lab with **CIEDE2000 (ΔE00)** rather than Euclidean RGB. CIEDE2000 is widely used to compute industrial color differences and has well-known implementation pitfalls; Sharma et al. provide implementation notes and test data. citeturn52view0

## Evaluation protocol, metrics, and a suggested test dataset

### What to evaluate

A rigorous evaluation should separate the three error sources:

- **Localization error:** wrong pixels (segmentation/detection mistakes).
- **Color estimation error:** correct pixels but unstable extraction (highlights, folds).
- **Naming/mapping error:** correct extracted color but mapped to the “wrong name” due to palette boundaries or human disagreement.

### Metrics

Use task-appropriate metrics per stage:

- **Detection:** mAP (IoU thresholds) for garment boxes. (Standard for object detection.)
- **Segmentation:** mIoU / per-class IoU for semantic parsing; mask AP for instance segmentation.
- **Color (continuous):** ΔE00 between predicted Lab centroid and ground truth Lab reference per garment. citeturn52view0
- **Color (discrete names):**
  - Top-1 / Top-k accuracy on a fixed label set (e.g., CSS names subset).
  - Hierarchical accuracy if you use a multi-level system (e.g., Level-1 correct but Level-3 off).
- **Multi-color garments:** treat as mixture prediction:
  - Earth Mover’s Distance (EMD) between predicted color distribution and annotated distribution (if you annotate proportions).
  - Intersection-over-union over sets of named colors (if you annotate 2–3 dominant names per garment).

### Suggested test dataset strategy

No mainstream HF dataset simultaneously provides (a) high-quality garment instance masks and (b) fine-grained shade labels. A practical compromise is to **reuse an existing fashion mask dataset** and add your own color annotations.

Strong base datasets to build on:

- **DeepFashion2:** The official repository describes it as containing **491K images** across **13 clothing categories**, with **801K clothing items** and rich per-item labels including bounding boxes and **per-pixel masks** (plus landmarks and other metadata). citeturn54search0
- **Look into Person (LIP):** introduced as a human parsing benchmark with **>50,000 images** and **19 semantic part labels**, designed for unconstrained viewpoints/occlusions/backgrounds. citeturn54search1turn54search5
- **ATR (Active Template Regression) human parsing dataset:** HF dataset cards describe it with **18 semantic category labels** and **17,700 images** (with a suggested train/test split), and other references describe it as a fashion-oriented single-person parsing dataset. citeturn54search2turn54search6

For category diversity with fashion-specific labels (boxes, not masks on HF):

- **Fashionpedia on HF (`detection-datasets/fashionpedia`):** provides a 46-class taxonomy and bounding boxes, with dataset scale and license clearly stated on the HF card. citeturn7view0  
  Note: the original Fashionpedia paper includes fine-grained attributes and segmentation masks in its full form, but the HF-hosted detection version is a bounding-box-centric conversion. citeturn6search3turn7view0

### Annotation format recommendation

Use **COCO instance segmentation JSON** as the base and extend each annotation with a structured color block.

Minimal recommended fields per garment instance:

- `category_id` (garment type)
- `segmentation` (polygon or RLE)
- `bbox`
- `color`:
  - `dominant_hex`: `["#RRGGBB", ...]`
  - `dominant_lab`: `[[L,a,b], ...]` (numeric)
  - `proportions`: `[0.62, 0.24, 0.14]` (optional but very useful)
  - `name_css`: `["darkslateblue"]` (optional)
  - `name_custom`: `["deep teal"]` (optional)
  - `annotation_method`: `"measured_from_mask"` vs `"human_named"` vs `"both"`

To support “primary → very specific shade,” store **both**:

- a **continuous** representation (Lab, hex), and
- one or more **names** at different granularity levels (e.g., coarse bucket + fine name).

This allows you to evaluate (a) numeric color accuracy and (b) naming consistency separately. citeturn52view0turn49search1turn49search4

## Licensing, deployment, and practical pitfalls

### Licensing realities

If you intend commercial deployment, license screening is not optional:

- Several parsing models fine-tuned from SegFormer are tagged “other” or explicitly “nvidia-segformer,” and the upstream SegFormer repository states it may be used **non-commercially** (research/evaluation) and points to NVIDIA research licensing for business use. citeturn15search2turn22view0turn11view0
- In contrast, the open-vocabulary production stack of **Grounding DINO (Apache-2.0)** and **SAM (Apache-2.0)** is much easier to operationalize from a licensing standpoint. citeturn33view0turn36view0
- Some helpful utilities like `adikuma/Colour-Extract` have **unknown license** in the model card; treat such components as “prototype only” unless the license is clarified. citeturn53view0

### Technical pitfalls that most strongly affect shade accuracy

- **Mask contamination dominates:** a 5–10 pixel bleed from background can shift a garment’s dominant color materially—especially with pastel garments against colored walls. This is why erosion and boundary trimming are essential.
- **Specular highlights and shadows:** a glossy black leather jacket can produce large white highlight clusters; treat highlights explicitly (trim at high L\* and low saturation) and use robust statistics in Lab. citeturn52view0
- **Patterns vs “color”:** striped shirts are intrinsically multi-color. Your output schema should support 2–3 dominant colors plus proportions, not a single name.
- **Prompt-based naming is not measurement:** CLIP-like models can be excellent semantic rankers (“is this closer to ‘olive’ or ‘khaki’?”) but are not anchored to a calibrated color space; always prefer pixel-based measurement for shade precision. citeturn29view0turn23view0turn26view0

### Recommended “hybrid” strategy for best quality

1. Use **pixel-based extraction** for the numeric color (hex/Lab + proportions).
2. Use a **palette mapping** layer for naming:
   - CSS name for web UI (standard set) citeturn49search4turn49search1
   - ISCC–NBS-like multi-level naming for “human readable but rich” citeturn49search2turn49search5
3. Optionally, use a fashion-domain embedding model (e.g., `Marqo/marqo-fashionSigLIP` or `patrickjohncyh/fashion-clip`) as a **tie-breaker** for ambiguous names by comparing “{shade name} {garment}” prompts. citeturn26view0turn23view0turn32view0
