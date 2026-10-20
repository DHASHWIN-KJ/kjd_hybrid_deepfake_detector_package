#!/usr/bin/env python
import os
import sys
import argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ensemble import HybridForensicPredictor

def main():
    parser = argparse.ArgumentParser(description="Tri-Stream Forensic Deepfake Prediction Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to face image to inspect.")
    parser.add_argument("--save_heatmap", type=str, default="heatmap_output.jpg", help="Path to save heatmap overlay.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    predictor = HybridForensicPredictor()
    res = predictor.predict(args.image)
    print(predictor.render_summary(res))

    if res.get("heatmap_image") and args.save_heatmap:
        res["heatmap_image"].save(args.save_heatmap)
        print(f"[OK] Saved visual heatmap overlay to: {args.save_heatmap}\n")

if __name__ == '__main__':
    main()
