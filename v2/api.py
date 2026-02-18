from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from v2.src.color_pipeline.localize import CombinedMaskProvider, HeuristicMaskProvider
from v2.src.color_pipeline.pipeline import ColorExtractionPipeline


class ExtractRequest(BaseModel):
    image_url: str = Field(..., description="HTTP(S) image URL")
    title: str = Field(..., description="Product title used for garment detection")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum colors to return")
    palette_path: str | None = Field(
        default=None,
        description="Optional local path to a user palette (.csv/.json)",
    )
    use_skin_mask: bool = Field(
        default=False,
        description="Enable skin suppression masking",
    )
    external_id: str | None = Field(
        default=None,
        description="Optional external id for masked image naming",
    )


class ColorItem(BaseModel):
    hex: str
    matched_name: str | None
    matched_palette_name: str | None
    matched_code: str | None
    proportion: float
    percentage: float


class ExtractResponse(BaseModel):
    colors: list[ColorItem]
    palette_source: str
    warnings: list[str]
    mask_coverage: float


app = FastAPI(
    title="Color Extractor v2 API",
    version="1.0.0",
    description="Extract dominant apparel colors from an image URL using the v2 pipeline.",
)


def _build_pipeline(use_skin_mask: bool, title: str) -> ColorExtractionPipeline:
    return ColorExtractionPipeline(
        mask_provider=CombinedMaskProvider(
            title=title,
            heuristic=HeuristicMaskProvider(use_skin_mask=use_skin_mask),
        )
    )


def _mask_output_path(external_id: str) -> str:
    root_dir = Path(__file__).resolve().parents[1]
    safe_id = external_id.replace("/", "_").replace("\\", "_")
    return str(root_dir / "masked_images" / f"{safe_id}.jpg")


@app.post("/extract", response_model=ExtractResponse)
async def extract_colors(payload: ExtractRequest) -> ExtractResponse:
    pipeline = _build_pipeline(use_skin_mask=payload.use_skin_mask, title=payload.title)
    debug_mask_out = (
        _mask_output_path(payload.external_id) if payload.external_id else None
    )
    try:
        result = await run_in_threadpool(
            pipeline.run,
            payload.image_url,
            payload.title,
            payload.palette_path,
            payload.top_k,
            debug_mask_out,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed_to_extract_colors: {exc}"
        ) from exc

    colors = [
        ColorItem(
            hex=color.hex,
            matched_name=color.matched_name,
            matched_palette_name=color.matched_palette_name,
            matched_code=color.matched_code,
            proportion=float(color.proportion),
            percentage=float(color.proportion * 100.0),
        )
        for color in result.dominant_colors
    ]
    return ExtractResponse(
        colors=colors,
        palette_source=result.palette_source,
        warnings=result.warnings,
        mask_coverage=float(result.mask_coverage),
    )
