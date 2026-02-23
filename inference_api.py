from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from starlette.concurrency import run_in_threadpool
import torch

from src.color_pipeline.localize import (
    CombinedMaskProvider,
    HeuristicMaskProvider,
    SegformerMaskProvider,
)
from src.color_pipeline.pipeline import ColorExtractionPipeline


class ExtractRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="HTTP(S) image URL")
    title: str = Field(..., description="Product title used for garment detection")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum colors to return")
    palette_path: str | None = Field(
        default=None,
        description="Optional local path to a user palette (.csv/.json)",
    )
    external_id: str | None = Field(
        default=None,
        description="Optional external id for raw result storage",
    )
    debug_mask_out: str | None = Field(
        default=None,
        description="Optional explicit path to save the masked output image",
    )
    solid_garment_mode: bool = Field(
        default=False,
        description=(
            "Enable conservative mask tightening and secondary-shade pruning for "
            "mostly single-color garments."
        ),
    )


class ColorItem(BaseModel):
    name: str | None
    code: str | None
    hex: str
    percentage: float


class ExtractResponse(BaseModel):
    colors: list[ColorItem]
    palette_source: str
    mask_coverage: float
    warnings: list[str]
    debug_mask_out: str | None = None


class SegformerRuntime:
    def __init__(self, model_id: str = "mattmdjaga/segformer_b2_clothes") -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load(self) -> None:
        provider = SegformerMaskProvider(title="warmup", model_id=self.model_id)
        provider._ensure_model(self.device)
        self.model = provider.model
        self.processor = provider.processor

    def build_provider(self, title: str) -> SegformerMaskProvider:
        provider = SegformerMaskProvider(title=title, model_id=self.model_id)
        provider.device = self.device
        provider.model = self.model
        provider.processor = self.processor
        return provider


app = FastAPI(
    title="Color Extractor Inference API",
    version="1.0.0",
    description=(
        "High-throughput color extraction API that keeps SegFormer B2 loaded in memory."
    ),
)

runtime = SegformerRuntime()


@app.on_event("startup")
async def _startup() -> None:
    await run_in_threadpool(runtime.load)


def _save_raw_result(external_id: str, result: Any) -> None:
    root_dir = Path(__file__).resolve().parent
    raw_values_dir = root_dir / "raw_values"
    raw_values_dir.mkdir(parents=True, exist_ok=True)
    safe_id = external_id.replace("/", "_").replace("\\", "_")
    output_path = raw_values_dir / f"{safe_id}.json"
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def _run_pipeline(payload: ExtractRequest):
    segformer = runtime.build_provider(payload.title)
    mask_provider = CombinedMaskProvider(
        title=payload.title,
        heuristic=HeuristicMaskProvider(),
        segformer=segformer,
    )
    pipeline = ColorExtractionPipeline(mask_provider=mask_provider)
    # Debug mask output is opt-in only.
    debug_mask_out = payload.debug_mask_out
    result = pipeline.run(
        image_path=str(payload.image_url),
        title=payload.title,
        palette_path=payload.palette_path,
        top_k=payload.top_k,
        debug_mask_out=debug_mask_out,
        solid_garment_mode=payload.solid_garment_mode,
    )
    if payload.external_id:
        _save_raw_result(payload.external_id, result)
    return result, debug_mask_out


@app.post("/extract", response_model=ExtractResponse)
async def extract_colors(payload: ExtractRequest) -> ExtractResponse:
    try:
        result, debug_mask_out = await run_in_threadpool(_run_pipeline, payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed_to_extract_colors: {exc}"
        ) from exc

    colors = [
        ColorItem(
            name=color.matched_name,
            code=color.matched_code,
            hex=color.hex,
            percentage=float(color.proportion * 100.0),
        )
        for color in result.dominant_colors
    ]
    return ExtractResponse(
        colors=colors,
        palette_source=result.palette_source,
        mask_coverage=float(result.mask_coverage),
        warnings=result.warnings,
        debug_mask_out=debug_mask_out,
    )
