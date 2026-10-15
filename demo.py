"""
Interactive Demo Launcher for Industrial Laser Metrology Engine.
Runs a zero-dependency local web server to interact with the visual station in your browser,
or executes an automated command-line demonstration.

Usage:
    python demo.py          # Launches interactive browser demo station
    python demo.py --cli    # Runs instant terminal metrology demo
"""

import argparse
import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path


def run_web_demo(port: int = 8080):
    repo_dir = Path(__file__).resolve().parent
    html_file = repo_dir / "demo.html"
    if not html_file.exists():
        sys.exit(f"Error: {html_file} not found.")

    os.chdir(repo_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable cache for local demo iterations
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def log_message(self, format, *args):
            # Silent logging for clean terminal output
            pass

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            url = f"http://localhost:{port}/demo.html"
            print("================================================================")
            print("  INDUSTRIAL LASER METROLOGY STATION - INTERACTIVE DEMO")
            print("================================================================")
            print(f"  Live Demo Station URL: {url}")
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
    from laser_measure import CameraCalibration, measure_laser_distance, render_inspection_overlay
    import cv2

    repo_dir = Path(__file__).resolve().parent
    img_path = repo_dir / "sample_laser_test.png"
    calib_path = repo_dir / "calibration.json"

    if not img_path.exists() or not calib_path.exists():
        print("Generating test fixtures...")
        from check import test_subpixel_spot_measurement
        test_subpixel_spot_measurement()

    calib = CameraCalibration.load(calib_path)
    img = cv2.imread(str(img_path))
    res = measure_laser_distance(img, calib, mode="spots", laser_color="red")

    print("================================================================")
    print("  LIVE INSPECTION READOUT (A -> B)")
    print("================================================================")
    print(f"  Point A Image (Sub-pixel px): ({res.point_a_px[0]:.2f}, {res.point_a_px[1]:.2f})")
    print(f"  Point B Image (Sub-pixel px): ({res.point_b_px[0]:.2f}, {res.point_b_px[1]:.2f})")
    print(f"  Point A Physical (mm):        ({res.point_a_mm[0]:.3f}, {res.point_a_mm[1]:.3f})")
    print(f"  Point B Physical (mm):        ({res.point_b_mm[0]:.3f}, {res.point_b_mm[1]:.3f})")
    print("----------------------------------------------------------------")
    print(f"  TRUE MEASURED DISTANCE:       {res.distance_mm:.3f} mm")
    print(f"  UNCERTAINTY BUDGET (95% CI):  +/- {res.uncertainty_mm:.3f} mm")
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
