"""
Interactive Demo Launcher for Industrial Laser Metrology Engine.
Runs an industrial inspection web server supporting direct test photo uploads (JPG, PNG, TIFF, BMP, WebP),
pinpoint sub-pixel measurement, and automated terminal demonstrations.

Usage:
    python demo.py          # Launches interactive browser metrology station
    python demo.py --cli    # Runs instant terminal metrology demo
"""

import argparse
import base64
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from laser_measure import (
    CameraCalibration,
    LaserDetector,
    MeasurementResult,
    measure_laser_distance,
    measure_points_with_refinement,
    render_inspection_overlay,
)


def run_web_demo(port: int = 8080):
    repo_dir = Path(__file__).resolve().parent
    html_file = repo_dir / "demo.html"
    calib_file = repo_dir / "calibration.json"
    if not html_file.exists():
        sys.exit(f"Error: {html_file} not found.")

    os.chdir(repo_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable cache for local demo iterations & allow CORS
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            if self.path == "/api/measure":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    raw_b64 = payload.get("image_base64", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    
                    img_bytes = base64.b64decode(raw_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        raise ValueError("Failed to decode uploaded image. Supported formats: JPG, PNG, WEBP, BMP, TIFF.")

                    pt_a = payload.get("pt_a")
                    pt_b = payload.get("pt_b")
                    laser_color = payload.get("color", "red")
                    mode = payload.get("mode", "spots")
                    refine = payload.get("refine_peaks", True)
                    source = payload.get("source", "auto")

                    # Calibration resolution (supporting 1-click reference calibration)
                    calibrate_to_cm = payload.get("calibrate_to_cm")
                    scale_cm = payload.get("scale_cm_per_px")
                    scale_mm = payload.get("scale_mm_per_px")

                    if calibrate_to_cm and pt_a and pt_b:
                        px_dist = math.hypot(float(pt_b[0]) - float(pt_a[0]), float(pt_b[1]) - float(pt_a[1]))
                        s = (float(calibrate_to_cm) * 10.0) / max(1e-6, px_dist)
                        H = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                        calib = CameraCalibration(homography=H, mean_reprojection_error_mm=0.001)
                    elif scale_cm and float(scale_cm) > 0:
                        s = float(scale_cm) * 10.0
                        H = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                        calib = CameraCalibration(homography=H, mean_reprojection_error_mm=0.002)
                    elif scale_mm and float(scale_mm) > 0:
                        s = float(scale_mm)
                        H = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                        calib = CameraCalibration(homography=H, mean_reprojection_error_mm=0.002)
                    elif calib_file.exists():
                        calib = CameraCalibration.load(str(calib_file))
                    else:
                        # Fallback default
                        H = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                        calib = CameraCalibration(homography=H, mean_reprojection_error_mm=0.005)

                    if pt_a and pt_b:
                        pt_a_hint = (float(pt_a[0]), float(pt_a[1]))
                        pt_b_hint = (float(pt_b[0]), float(pt_b[1]))
                        result = measure_points_with_refinement(
                            bgr_image=img,
                            calibration=calib,
                            pt_a_hint=pt_a_hint,
                            pt_b_hint=pt_b_hint,
                            laser_color=laser_color,
                            refine_peaks=refine,
                            source=source,
                        )
                    else:
                        result = measure_laser_distance(
                            bgr_image=img,
                            calibration=calib,
                            mode=mode,
                            laser_color=laser_color,
                            source=source,
                        )

                    overlay = render_inspection_overlay(img, result)
                    _, buf = cv2.imencode(".png", overlay)
                    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

                    resp_data = {
                        "success": True,
                        "result": {
                            "point_a_px": [round(result.point_a_px[0], 2), round(result.point_a_px[1], 2)],
                            "point_b_px": [round(result.point_b_px[0], 2), round(result.point_b_px[1], 2)],
                            "point_a_cm": [result.point_a_cm[0], result.point_a_cm[1]],
                            "point_b_cm": [result.point_b_cm[0], result.point_b_cm[1]],
                            "distance_cm": result.distance_cm,
                            "uncertainty_cm": result.uncertainty_cm,
                            "point_a_mm": [result.point_a_mm[0], result.point_a_mm[1]],
                            "point_b_mm": [result.point_b_mm[0], result.point_b_mm[1]],
                            "distance_mm": result.distance_mm,
                            "uncertainty_mm": result.uncertainty_mm,
                            "gsd_mm_per_px": result.gsd_mm_per_px,
                        },
                        "overlay_base64": overlay_b64,
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp_data).encode("utf-8"))

                except Exception as e:
                    resp_data = {"success": False, "error": str(e)}
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            else:
                super().do_POST()

        def log_message(self, format, *args):
            # Silent logging for clean terminal output
            pass

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            url = f"http://localhost:{port}/demo.html"
            print("================================================================")
            print("  HRL INDUSTRIAL LASER METROLOGY STATION - LIVE INSPECTION SERVER")
            print("================================================================")
            print(f"  Live Demo Station URL: {url}")
            print("  Photo Upload API:      POST /api/measure (JPG, PNG, WEBP, TIFF, BMP)")
            print("  Opening browser automatically...")
            print("  Press Ctrl+C to stop the demo server.")
            print("================================================================")

            threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo station stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {port} busy, trying port {port + 1}...")
            run_web_demo(port + 1)
        else:
            raise e


def run_cli_demo():
    print("Running terminal metrology demo using synthetic optical test frame...")
    repo_dir = Path(__file__).resolve().parent
    img_path = repo_dir / "sample_laser_test.png"
    calib_path = repo_dir / "calibration.json"

    if not img_path.exists() or not calib_path.exists():
        print("Generating test fixtures...")
        from check import test_subpixel_spot_measurement
        test_subpixel_spot_measurement()

    calib = CameraCalibration.load(str(calib_path))
    img = cv2.imread(str(img_path))
    res = measure_laser_distance(img, calib, mode="spots", laser_color="red")

    print("================================================================")
    print("  LIVE INSPECTION READOUT (A -> B)")
    print("================================================================")
    print(f"  Point A Image (Sub-pixel px): ({res.point_a_px[0]:.2f}, {res.point_a_px[1]:.2f})")
    print(f"  Point B Image (Sub-pixel px): ({res.point_b_px[0]:.2f}, {res.point_b_px[1]:.2f})")
    print(f"  Point A Physical (cm):        ({res.point_a_cm[0]:.3f}, {res.point_a_cm[1]:.3f}) cm")
    print(f"  Point B Physical (cm):        ({res.point_b_cm[0]:.3f}, {res.point_b_cm[1]:.3f}) cm")
    print("----------------------------------------------------------------")
    print(f"  TRUE MEASURED DISTANCE:       {res.distance_cm:.3f} cm  ({res.distance_mm:.3f} mm)")
    print(f"  UNCERTAINTY BUDGET (95% CI):  +/- {res.uncertainty_cm:.4f} cm  (+/- {res.uncertainty_mm:.3f} mm)")
    print(f"  SENSOR RESOLUTION (GSD):      {res.gsd_mm_per_px:.4f} mm/pixel")
    print("================================================================")

    out_path = repo_dir / "demo_inspection_result.png"
    overlay = render_inspection_overlay(img, res)
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved annotated inspection HUD overlay: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Laser Distance Measurement Demo")
    parser.add_argument("--cli", action="store_true", help="Run terminal inspection demo instead of web UI")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo()
    else:
        run_web_demo(port=args.port)


if __name__ == "__main__":
    main()
