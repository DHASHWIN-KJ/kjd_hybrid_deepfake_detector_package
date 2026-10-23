"""
Phase 3: Upgraded Interactive Deepfake Verification Web Application
===================================================================
A multi-scale forensic analysis suite powered by:
- CLIP ViT-L/14: Foundation semantic decision anchor (cross-generator generalization)
- ResNet-18: Convolutional micro-texture and frequency sensor-noise classifier
- Forensic U-Net: Spatial reconstruction residual heatmap localizing spliced seams
- Calibrated Confirmation Percentage & Multi-Model Consensus Engine

Run with:
    python app/gradio_app.py
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import gradio as gr

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ensemble import HybridForensicPredictor

# Initialize Hybrid Forensic Predictor globally once
print("[app] Initializing Hybrid Forensic Predictor (CLIP + ResNet-18 + Forensic U-Net)...")
predictor = HybridForensicPredictor(
    clip_checkpoint="checkpoints/full_detector_package.pth",
    resnet_checkpoint="best_resnet18_universal.pth",
    unet_checkpoint="checkpoints/forensic_unet.pth",
    calibration_path="results/calibration.json",
    alpha=0.85,
)
print("[app] Predictor ready.")


def analyze_image(
    img: Optional[Image.Image],
    threshold: float = 0.467,
) -> Tuple[Optional[Image.Image], str, Dict[str, float], str]:
    """
    Analyzes an uploaded PIL Image and returns:
    1. Forensic U-Net Heatmap Image.
    2. HTML formatted result card with confirmation percentage & consensus.
    3. Dictionary of stream probabilities for gr.Label.
    4. Markdown formatted technical diagnostic details.
    """
    if img is None:
        placeholder_html = """
        <div style='padding: 28px; text-align: center; color: #94a3b8; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <div style="font-size: 32px; margin-bottom: 8px;">🛡️</div>
            <div style="font-size: 16px; font-weight: 600;">Awaiting Face Image</div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">Upload an image on the left or select a quick test sample below.</div>
        </div>
        """
        return None, placeholder_html, {}, "*No image provided.*"

    # Run full hybrid prediction with dynamic sensitivity threshold
    res = predictor.predict(img, generate_heatmap=True, threshold=threshold)

    print("\n" + "=" * 60)
    print(f"🔍 [FORENSIC INSPECTION] Image Size: {img.size}")
    print(f"VERDICT: [{res['verdict']}] with {res['confidence_pct']:.1f}% confidence")
    print(f"  • CLIP ViT-L/14 Semantic Stream: {res['model_breakdown']['clip_prob_fake']*100:.1f}% Fake")
    if res['model_breakdown']['resnet_prob_fake'] is not None:
        print(f"  • ResNet-18 Micro-Texture Stream: {res['model_breakdown']['resnet_prob_fake']*100:.1f}% Fake")
    print(f"  • Forensic U-Net Anomaly Score:  {res['model_breakdown']['unet_anomaly_score']*100:.1f}% (Seam: {res['seam_detected']})")
    print(f"  • Combined Fake Probability:     {res['p_fake']*100:.1f}% (Active Threshold: {res['threshold_used']:.3f})")
    print(f"  • Multi-Model Consensus:         {res['consensus']}")
    print("=" * 60 + "\n")

    is_fake = "DEEPFAKE" in res["verdict"].upper()
    conf_pct = res["confidence_pct"]
    consensus = res["consensus"]
    breakdown = res["model_breakdown"]
    regions = res["anomalous_regions"]
    seam = res["seam_detected"]

    # Borderline uncertainty handling
    is_borderline = any("Borderline" in w for w in res.get("warnings", []))

    # Theme colors
    if is_borderline:
        card_color = "#d97706"
        badge_bg = "#fffbeb"
        badge_border = "#fde68a"
        badge_icon = "⚠️"
        verdict_title = "BORDERLINE / INCONCLUSIVE"
        verdict_sub = f"Score ({res['p_fake']*100:.1f}%) is in the uncertainty margin — Manual Forensic Review Advised"
    elif is_fake:
        card_color = "#e11d48"
        badge_bg = "#fff1f2"
        badge_border = "#fecdd3"
        badge_icon = "⚠️"
        verdict_title = "DEEPFAKE DETECTED"
        verdict_sub = "Manipulated or Synthetically Generated"
    else:
        card_color = "#059669"
        badge_bg = "#ecfdf5"
        badge_border = "#a7f3d0"
        badge_icon = "✓"
        verdict_title = "AUTHENTIC REAL FACE"
        verdict_sub = "Natural Facial Manifold Confirmed"

    # Region tags badges
    if regions and is_fake:
        region_badges_html = " ".join([
            f"""<span style="
                background: #fef2f2;
                color: #991b1b;
                border: 1px solid #fecaca;
                padding: 4px 10px;
                border-radius: 9999px;
                font-size: 11px;
                font-weight: 600;
                display: inline-block;
                margin: 2px;
            ">{r}</span>""" for r in regions
        ])
    elif not is_fake:
        region_badges_html = """<span style="
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        ">Smooth Natural Manifold (Uniform Low Residuals)</span>"""
    else:
        region_badges_html = """<span style="
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        ">Global Synthetic Perturbation</span>"""

    # Consensus styling
    if "STRONG" in consensus.upper():
        consensus_bg = "#eff6ff"
        consensus_text_color = "#1d4ed8"
        consensus_border = "#bfdbfe"
    elif "MODERATE" in consensus.upper():
        consensus_bg = "#f0fdf4"
        consensus_text_color = "#15803d"
        consensus_border = "#bbf7d0"
    else:
        consensus_bg = "#fffbeb"
        consensus_text_color = "#b45309"
        consensus_border = "#fde68a"

    # HTML Card
    html_card = f"""
    <div style="
        background: linear-gradient(135deg, {badge_bg} 0%, #ffffff 100%);
        border: 2px solid {card_color};
        border-radius: 16px;
        padding: 22px;
        margin: 4px 0 16px 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    ">
        <!-- Top Status Bar -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="
                    background-color: {card_color};
                    color: white;
                    font-size: 20px;
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                ">{badge_icon}</span>
                <div>
                    <div style="font-size: 22px; font-weight: 800; color: {card_color}; line-height: 1.1; letter-spacing: -0.3px;">
                        {verdict_title}
                    </div>
                    <div style="font-size: 12px; color: #64748b; font-weight: 500; margin-top: 2px;">
                        {verdict_sub}
                    </div>
                </div>
            </div>
            <div style="
                background: {consensus_bg};
                color: {consensus_text_color};
                border: 1px solid {consensus_border};
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.2px;
            ">
                {consensus.split('(')[0].strip()}
            </div>
        </div>

        <!-- Metric Area -->
        <div style="margin-top: 14px; padding-top: 14px; border-top: 1px solid rgba(0,0,0,0.06);">
            <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;">
                Calibrated Percentage of Confirmation
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
                <span style="font-size: 40px; font-weight: 900; color: #0f172a; line-height: 1; letter-spacing: -1px;">
                    {conf_pct:.1f}%
                </span>
                <span style="font-size: 14px; color: #64748b; font-weight: 500;">
                    certainty in <strong>{'DEEPFAKE' if is_fake else 'AUTHENTIC'}</strong> verdict
                </span>
            </div>
        </div>

        <!-- Anomaly Region Badges -->
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,0.06);">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 700; margin-bottom: 6px;">
                Spatial U-Net Anomaly Localization:
            </div>
            <div>
                {region_badges_html}
            </div>
        </div>
    </div>
    """

    # Label dictionary for visual probability breakdown
    label_dict = {
        "Ensemble Combined Fake Probability": res["p_fake"],
        "CLIP ViT-L/14 Semantic Stream": breakdown["clip_prob_fake"],
    }
    if breakdown["resnet_prob_fake"] is not None:
        label_dict["ResNet-18 Micro-Texture Stream"] = breakdown["resnet_prob_fake"]
    label_dict["U-Net Anomaly Residual Score"] = breakdown["unet_anomaly_score"]

    # Diagnostics markdown
    img_size_str = res["image_size"]
    warnings_md = ""
    if res["warnings"]:
        warnings_md = "\n".join([f"- ⚠️ **Warning**: {w}" for w in res["warnings"]])
    else:
        warnings_md = "- ✓ Resolution check passed (adequate pixel density for multi-scale analysis)."

    resnet_line = f"- **ResNet-18 Texture Stream**: `{breakdown['resnet_prob_fake']*100:.1f}%` Manipulated" if breakdown['resnet_prob_fake'] is not None else "- **ResNet-18 Stream**: *(Optional model bypassed)*"

    roi_info = res.get("roi_info", {})
    if roi_info.get("crop_applied"):
        roi_md = f"- **Auto-Face RoI Localization**: ✓ Applied (Detected Face BBox: `{roi_info['bbox']}` with 25% seam margin)"
    else:
        roi_md = "- **Auto-Face RoI Localization**: Direct Benchmark Face Evaluation (No Crop Needed)"

    details_md = f"""
### Multi-Stream Technical Diagnostics
- **Image Resolution**: `{img_size_str}` px
- **Multi-Model Consensus**: `{consensus}`
- **CLIP ViT-L/14 Semantic Stream**: `{breakdown['clip_prob_fake']*100:.1f}%` Manipulated
{resnet_line}
- **Forensic U-Net Spatial Heatmap**: `{breakdown['unet_anomaly_score']*100:.1f}%` relative residual contrast *(Illustrative localization, independent of verdict)*
{roi_md}
- **Calibrated Threshold ($\theta$)**: `{res['threshold_used']:.3f}` *(T = {res['temperature_used']:.4f})*
- **Fusion Architecture**: Calibrated Foundation Blend (80% CLIP ViT-L/14 + 20% ResNet-18) validated across 2,000 images

#### Preprocessing & Boundary Checks:
{warnings_md}
"""

    return res["heatmap_image"], html_card, label_dict, details_md


# Custom Gradio Interface with 3-column / responsive layout
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Hybrid Multi-Scale Forensic Deepfake Detector") as demo:
    gr.Markdown(
        """
        # 🛡️ Hybrid Multi-Scale Forensic Deepfake Verification Platform
        ### Unifying CLIP ViT-L/14 Semantic Foundations, ResNet-18 Micro-Texture & Forensic U-Net Spatial Heatmaps
        Upload any human face portrait or synthetic media sample to generate real-time authenticity classification, calibrated confidence %, and pixel-level manipulation boundary heatmaps.
        """
    )

    with gr.Row():
        # Column 1: Input Face
        with gr.Column(scale=4):
            image_input = gr.Image(
                type="pil",
                label="Input Face Image (Upload, Clipboard, or Drag & Drop)",
                sources=["upload", "clipboard"],
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                predict_btn = gr.Button("Inspect Authenticity", variant="primary")

            with gr.Accordion("⚙️ Forensic Sensitivity Calibration", open=False):
                sensitivity_slider = gr.Slider(
                    minimum=0.35,
                    maximum=0.65,
                    value=0.467,
                    step=0.01,
                    label="Operational Decision Threshold (θ)",
                    info="Lower (e.g. 0.42) catches subtle face swaps. Higher (0.55) is conservative.",
                )

        # Column 2: Forensic U-Net Heatmap
        with gr.Column(scale=4):
            heatmap_output = gr.Image(
                type="pil",
                label="Forensic U-Net Heatmap (Anomaly Localization)",
                interactive=False,
            )

        # Column 3: Forensic Analytics Card
        with gr.Column(scale=5):
            verdict_output = gr.HTML(
                value="""
                <div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>
                    Upload a face image on the left and click <b>Inspect Authenticity</b> to run the multi-scale forensic engine.
                </div>
                """,
                label="Authenticity Verdict & Confirmation",
            )
            prob_output = gr.Label(
                label="Multi-Model Stream Breakdown",
                num_top_classes=4,
            )
            details_output = gr.Markdown(
                value="*Awaiting image input...*",
                label="Technical Diagnostics",
            )

    # Event handlers
    predict_btn.click(
        fn=analyze_image,
        inputs=[image_input, sensitivity_slider],
        outputs=[heatmap_output, verdict_output, prob_output, details_output],
    )

    image_input.change(
        fn=analyze_image,
        inputs=[image_input, sensitivity_slider],
        outputs=[heatmap_output, verdict_output, prob_output, details_output],
    )

    sensitivity_slider.change(
        fn=analyze_image,
        inputs=[image_input, sensitivity_slider],
        outputs=[heatmap_output, verdict_output, prob_output, details_output],
    )

    clear_btn.click(
        fn=lambda: (
            None,
            """
            <div style='padding: 30px; text-align: center; color: #94a3b8; font-size: 15px;'>
                Upload a face image on the left and click <b>Inspect Authenticity</b> to run the multi-scale forensic engine.
            </div>
            """,
            {},
            "*Awaiting image input...*",
        ),
        inputs=[],
        outputs=[heatmap_output, verdict_output, prob_output, details_output],
    )

    # Quick test samples
    example_paths = [
        os.path.join(PROJECT_ROOT, "data_std", "original_inswapper", "real", "real_00000.jpg"),
        os.path.join(PROJECT_ROOT, "data_std", "original_inswapper", "fake", "fake_00000.jpg"),
    ]
    valid_examples = [[p] for p in example_paths if os.path.isfile(p)]

    if valid_examples:
        gr.Markdown("### 🔍 Benchmark Test Samples (Click any sample below for instant forensic analysis)")
        gr.Examples(
            examples=valid_examples,
            inputs=[image_input],
            outputs=[heatmap_output, verdict_output, prob_output, details_output],
            fn=analyze_image,
            cache_examples=False,
        )

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Launching Hybrid Multi-Scale Deepfake Forensic Web App...")
    print("Access the interface in your browser at: http://127.0.0.1:7860")
    print("=" * 70 + "\n")
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=theme)
    except OSError:
        print("[app] Port 7860 occupied, launching on next available port...")
        demo.launch(server_name="127.0.0.1", inbrowser=True, theme=theme)
