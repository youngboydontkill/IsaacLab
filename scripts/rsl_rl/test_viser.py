#!/usr/bin/env python3
"""Test script to verify viser visualization functionality."""

import argparse
import threading
import time
from collections import deque

import numpy as np
import viser

def main():
    parser = argparse.ArgumentParser(description="Test viser visualization.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8891, help="Server port.")
    args = parser.parse_args()

    print(f"[INFO] Starting viser test server at http://{args.host}:{args.port}")

    server = viser.ViserServer(host=args.host, port=args.port)

    # Add UI elements
    with server.gui.add_folder("Test Status"):
        iteration_text = server.gui.add_text("# Iteration", initial_value="0")
        value_text = server.gui.add_text("Sine Value", initial_value="0.00")
        cosine_text = server.gui.add_text("Cosine Value", initial_value="0.00")

    with server.gui.add_folder("Test Curves"):
        # Sine and cosine chart
        chart = server.gui.add_uplot(
            data=(np.array([0.0]), np.array([0.0]), np.array([0.0])),
            series=(
                {},  # x-axis
                {"stroke": "rgba(0, 255, 0, 1)", "width": 2},  # sine
                {"stroke": "rgba(0, 0, 255, 1)", "width": 2},  # cosine
            ),
            title="Sine and Cosine Waves",
            scales={"x": {"auto": True}, "y": {"auto": True}},
            axes=[
                {"label": "Time"},
                {"label": "Value", "stroke": "#888"},
            ],
        )

    with server.gui.add_folder("3D Test"):
        test_frame = server.scene.add_frame(name="test_frame")
        server.scene.add_box(
            "/test_frame/box",
            dimensions=(0.5, 0.5, 0.5),
            color=(1, 0, 0),
        )
        server.scene.add_icosphere(
            "/test_frame/sphere",
            radius=0.2,
            color=(0, 1, 0),
            position=(1.0, 0, 0),
        )
        server.scene.add_cylinder(
            "/test_frame/arrow",
            height=0.5,
            radius=0.05,
            color=(0, 0, 1),
            wxyz=(0.7071, 0.7071, 0, 0),  # Rotate 90 degrees around X
            position=(0, 1.25, 0),
        )

    print("[INFO] Server started. Open your browser to test the visualization.")
    print("[INFO] Press Ctrl+C to stop the server.")

    # Data storage
    times = deque(maxlen=100)
    sines = deque(maxlen=100)
    cosines = deque(maxlen=100)

    # Start time
    start_time = time.time()
    iteration = 0

    # Update loop
    try:
        while True:
            # Update values
            t = time.time() - start_time
            sine_val = np.sin(t * 2.0)
            cosine_val = np.cos(t * 2.0)

            # Update text
            iteration += 1
            iteration_text.value = str(iteration)
            value_text.value = f"{sine_val:.4f}"
            cosine_text.value = f"{cosine_val:.4f}"

            # Update data storage
            times.append(t)
            sines.append(sine_val)
            cosines.append(cosine_val)

            # Update chart
            chart.data = (np.array(times), np.array(sines), np.array(cosines))

            # Update 3D objects (rotation)
            import math
            test_frame.wxyz = (math.cos(t * 0.15), 0, 0, math.sin(t * 0.15))  # Rotation around X axis

            # Small delay
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()