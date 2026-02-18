from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from v2 import api


def test_extract_endpoint_returns_color_fields(monkeypatch):
    class FakePipeline:
        def run(self, image_path: str, palette_path: str | None, top_k: int, debug_mask_out: str | None):
            assert image_path == "https://example.com/image.jpg"
            assert palette_path is None
            assert top_k == 2
            assert debug_mask_out is None
            return SimpleNamespace(
                dominant_colors=[
                    SimpleNamespace(
                        hex="#123456",
                        matched_name="Blue",
                        matched_palette_name="Deep Blue",
                        matched_code="PMS-123",
                        proportion=0.6,
                    ),
                    SimpleNamespace(
                        hex="#222222",
                        matched_name="Black",
                        matched_palette_name="Black",
                        matched_code="PMS-000",
                        proportion=0.4,
                    ),
                ],
                palette_source="pantone_user",
                warnings=[],
                mask_coverage=0.7,
            )

    monkeypatch.setattr(api, "_build_pipeline", lambda use_skin_mask: FakePipeline())

    client = TestClient(api.app)
    response = client.post("/extract", json={"image_url": "https://example.com/image.jpg", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["palette_source"] == "pantone_user"
    assert payload["mask_coverage"] == 0.7
    assert len(payload["colors"]) == 2
    assert payload["colors"][0]["matched_name"] == "Blue"
    assert payload["colors"][0]["matched_palette_name"] == "Deep Blue"
    assert payload["colors"][0]["matched_code"] == "PMS-123"
    assert payload["colors"][0]["proportion"] == 0.6
    assert payload["colors"][0]["percentage"] == 60.0

