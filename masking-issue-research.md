Here’s what I found in the v2 pipeline that explains the “Segmentation/Masking Errors (mask leakage)” and why they show up ~19/100 items, plus a concrete plan to reduce them to “almost irrelevant”.
What’s causing leakage in the current pipeline

- Heuristic mask relies on corner color similarity (corner_lab_threshold=18) and edge-connected background. If garment and background are similar (light neutrals, studio backdrops), the background isn’t removed and remains foreground.
- Heuristic fallback is aggressive: when coverage is <0.20 or >0.98 it uses a center box; if the box includes background/skin/legs, leakage happens. This is frequent for model shots.
- SegFormer depends on NLP title category. If title parsing fails or is ambiguous, SegFormer returns empty → CombinedMaskProvider falls back to heuristic.
- If the mask is empty, extract_dominant_colors uses the full image (mask_empty_using_full_image). That guarantees leakage.
- Skin removal is enabled by default, but skin-colored regions can still leak if the heuristic thresholding misses them or SegFormer includes partial arms.
- CombinedMaskProvider returns heuristic even if it’s bad when combined mask coverage is low and segformer coverage is low. There’s no refinement step to “clean up” either mask.
  Why this matches your eval remarks
- Many remarks mention neutral backgrounds, skin, pants, or studio walls. Those are exactly the failure modes of the heuristic + center-box fallback.
- The SegFormer category path can silently fail (title not parsed) so you get heuristic-only behavior even though SegFormer is available.
- Low-contrast garments (off-white, beige, grey) are especially prone to background leakage because the heuristic’s corner-lab threshold can’t distinguish them.
  Plan to eliminate or minimize leakage
  This is ordered from highest impact with lowest complexity → more advanced.

1. Fix mask-fallback behavior

- Change the fallback sequence so a “bad” heuristic mask isn’t used when SegFormer is available.
- If combined mask coverage is low but SegFormer coverage is reasonable, prefer SegFormer even if combined is tiny.
- Remove the “empty mask → full image” fallback. Instead: return a warning and use the best available mask (even if small).
  Impact: prevents the worst leakage cases immediately.

2. Make SegFormer resilient to title parsing failure

- If NLP yields no category, default to upper-clothes (or dress if title contains dress-like terms) instead of zero-mask.
- Optionally include a lightweight keyword fallback (tshirt/tee/top/polo) before giving up.
  Impact: reduces SegFormer-empty → heuristic-only cases.

3. Post-process masks to remove background/skin

- Keep skin masking enabled by default and refine thresholds/morphology if needed for stubborn skin tones.
- Add a post-processing step to remove components that touch the image border and are color-similar to corners.
- Keep only the largest centered component and discard small off-center components.
  Impact: removes arms, pants, and studio wall leakage in model shots.

4. Add a refinement step that blends SegFormer + heuristic

- Use SegFormer mask as “sure foreground”, heuristic as “possible foreground”, then run a quick refinement (GrabCut or trimap).
- This keeps garment boundaries while rejecting background.
  Impact: best for low-contrast garments with soft edges.

5. Geometric priors by garment type

- For upper-clothes: constrain mask to upper 65–75% of person bbox (or image if no bbox).
- For bottoms: lower 50–60%.
- For dresses/long tops: allow full height.
  Impact: eliminates pants/shoes leakage for tops.

6. Adaptive corner thresholding

- Make corner_lab_threshold adaptive based on corner variance; if corners are uniform, use tighter threshold; if varied, loosen a bit.
  Impact: improves segmentation when background is uniform and low contrast.

7. Add a confidence gate before clustering

- If mask coverage is too low or too high, mark as “mask_fail” and skip clustering or reduce top_k.
- Prevents KMeans from amplifying leakage artifacts.
  Impact: avoids false secondary colors from tiny leaks.
  Suggested next step (concrete)
  If you want, I can:

1. Run a quick analysis of eval_results.csv to count which remark strings map to each failure type.
2. Draft the exact code changes for steps 1–3 (fallback logic, default SegFormer category, post-processing).
3. Provide a targeted “mask debug set” list (the 19 leakage items) to validate improvements.
   Which step do you want me to implement first?
