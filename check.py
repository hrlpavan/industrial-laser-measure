"""
Self-verifying test suite for industrial-laser-measure (Ponytail check).
No test frameworks required: run with `python check.py`.
Synthesizes perspective-tilted optical scenes with sub-pixel laser profiles,
calibrates from fiducials, and asserts sub-millimeter measurement accuracy.
"""

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from laser_measure import (
    CameraCalibration,
    LaserDetector,
    measure_laser_distance,
    render_inspection_overlay,
)


def synthesize_optical_scene(
    width_px: int = 1280,
    height_px: int = 960,
    camera_tilt_factor: float = 0.2,
):
    """
    Simulates an industrial camera viewing a 200x200 mm workpiece with perspective tilt.
    Returns:
        image: synthesized BGR image with 4 fiducials and 2 laser spots
        fiducials_px: 4 image coordinates
        fiducials_mm: 4 world coordinates
        true_dist_mm: exact physical distance between laser spots
    """
    img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    # 4 World fiducials on table (in mm)
    fiducials_mm = [
        (20.0, 20.0),
        (180.0, 20.0),
        (180.0, 180.0),
        (20.0, 180.0),
    ]

    # Target pixel coordinates with realistic perspective trapezoid
    fiducials_px = [
        (200.0, 200.0),
        (1080.0, 200.0),
        (1080.0 + camera_tilt_factor * 200.0, 800.0),
        (200.0 - camera_tilt_factor * 200.0, 800.0),
    ]

    # Compute ground truth World-to-Image homography H_w2i
    pts_mm = np.array(fiducials_mm, dtype=np.float32)
    pts_px = np.array(fiducials_px, dtype=np.float32)
    H_w2i, _ = cv2.findHomography(pts_mm, pts_px)

    # Helper: project world mm point to pixel
    def w2px(x_mm, y_mm):
        pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H_w2i)[0, 0]
        return float(res[0]), float(res[1])

    # Draw fiducials on image (white ring with dark center)
    for u, v in fiducials_px:
        cv2.circle(img, (int(round(u)), int(round(v))), 8, (200, 200, 200), -1)
        cv2.circle(img, (int(round(u)), int(round(v))), 2, (30, 30, 30), -1)

    # Laser spots: Point A at (40.0, 100.0) mm, Point B at (160.0, 100.0) mm
    pt_a_mm = (40.0, 100.0)
    pt_b_mm = (160.0, 100.0)
    true_dist_mm = math.hypot(pt_b_mm[0] - pt_a_mm[0], pt_b_mm[1] - pt_a_mm[1])  # 120.00 mm

    u_a, v_a = w2px(*pt_a_mm)
    u_b, v_b = w2px(*pt_b_mm)

    # Render Gaussian sub-pixel laser spots (Red laser: high R, low G/B)
    def render_gaussian_spot(image, cx, cy, radius_sigma=2.5, peak_val=250):
        r = int(math.ceil(radius_sigma * 3))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px = int(round(cx + dx))
                py = int(round(cy + dy))
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    dist_sq = (px - cx) ** 2 + (py - cy) ** 2
                    intensity = peak_val * math.exp(-dist_sq / (2.0 * radius_sigma ** 2))
                    # Red laser wavelength: strong Red, slight orange saturation core
                    image[py, px, 2] = min(255, int(image[py, px, 2] + intensity))
                    image[py, px, 1] = min(255, int(image[py, px, 1] + intensity * 0.2))
                    image[py, px, 0] = min(255, int(image[py, px, 0] + intensity * 0.1))

    render_gaussian_spot(img, u_a, v_a)
    render_gaussian_spot(img, u_b, v_b)

    return img, fiducials_px, fiducials_mm, true_dist_mm


def test_subpixel_spot_measurement():
    print("[1/3] Testing two-spot distance measurement under perspective distortion...")
    img, fid_px, fid_mm, true_dist_mm = synthesize_optical_scene()

    # 1. Calibrate using fiducials
    calib = CameraCalibration.from_fiducials(fid_px, fid_mm)
    assert calib.mean_reprojection_error_mm < 0.01, f"High calibration error: {calib.mean_reprojection_error_mm}"

    # Save raw test frame for CLI testing
    cv2.imwrite("sample_laser_test.png", img)

    # 2. Run distance measurement
    res = measure_laser_distance(img, calib, mode="spots", laser_color="red")
    error_mm = abs(res.distance_mm - true_dist_mm)

    print(f"      True Distance:     {true_dist_mm:.3f} mm")
    print(f"      Measured Distance: {res.distance_mm:.3f} mm")
    print(f"      Absolute Error:    {error_mm:.4f} mm")
    print(f"      Uncertainty Bound: +/- {res.uncertainty_mm:.4f} mm")

    # Assert sub-millimeter industrial accuracy (< 0.05 mm error on 120 mm baseline)
    assert error_mm < 0.05, f"Measurement error too high! {error_mm:.4f} mm exceeds 0.05 mm threshold"
    assert res.uncertainty_mm > 0.0, "Uncertainty bound must be positive"

    # Test inspection overlay rendering
    overlay = render_inspection_overlay(img, res)
    assert overlay.shape == img.shape
    cv2.imwrite("check_overlay.png", overlay)
    print("      Inspection overlay saved to check_overlay.png")


def test_2pt_perpendicular_calibration():
    print("[2/3] Testing 2-point perpendicular scale calibration...")
    # Known 100 mm scale between (100, 100) and (500, 100) px (400 px = 100 mm -> 0.25 mm/px)
    calib = CameraCalibration.from_two_points((100.0, 100.0), (500.0, 100.0), 100.0)
    x1, y1 = calib.pixel_to_world(100.0, 100.0)
    x2, y2 = calib.pixel_to_world(500.0, 100.0)
    d = math.hypot(x2 - x1, y2 - y1)
    assert abs(d - 100.0) < 1e-6, f"2-point scale mismatch: {d}"

    # Test serialization round-trip
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        calib.save(tmp.name)
        loaded = CameraCalibration.load(tmp.name)
        lx1, ly1 = loaded.pixel_to_world(100.0, 100.0)
        lx2, ly2 = loaded.pixel_to_world(500.0, 100.0)
        assert abs(math.hypot(lx2 - lx1, ly2 - ly1) - 100.0) < 1e-6
    print("      2-point calibration and JSON serialization verified.")


def test_continuous_line_endpoint_detection():
    print("[3/3] Testing continuous laser line endpoint extraction...")
    # Create black canvas and draw a laser line segment
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    pt1_px = (150, 200)
    pt2_px = (650, 400)
    cv2.line(img, pt1_px, pt2_px, (20, 40, 255), 3, cv2.LINE_AA)

    detector = LaserDetector(wavelength_color="red")
    sub_a, sub_b = detector.detect_line_endpoints(img)

    expected_len = math.hypot(pt2_px[0] - pt1_px[0], pt2_px[1] - pt1_px[1])
    detected_len = math.hypot(sub_b[0] - sub_a[0], sub_b[1] - sub_a[1])
    diff_px = abs(detected_len - expected_len)

    print(f"      Expected Line Length: {expected_len:.2f} px")
    print(f"      Detected Line Length: {detected_len:.2f} px")
    print(f"      Delta:                {diff_px:.2f} px")
    assert diff_px < 3.0, f"Line endpoint detection error too large: {diff_px} px"
    print("      Continuous laser line detection verified.")


if __name__ == "__main__":
    print("=== INDUSTRIAL LASER METROLOGY SELF-CHECK ===")
    test_subpixel_spot_measurement()
    test_2pt_perpendicular_calibration()
    test_continuous_line_endpoint_detection()
    print("=== ALL CHECKS PASSED (Industrial tolerance verified) ===")
