import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blur_faces as bf


class SizeTests(unittest.TestCase):
    def test_wheel_steps_for_mac_and_windows(self):
        self.assertEqual(bf.wheel_steps(1), -1)
        self.assertEqual(bf.wheel_steps(-1), 1)
        self.assertEqual(bf.wheel_steps(3), -3)
        self.assertEqual(bf.wheel_steps(120), -1)
        self.assertEqual(bf.wheel_steps(-240), 2)
        self.assertEqual(bf.wheel_steps(0, button=4), -1)
        self.assertEqual(bf.wheel_steps(0, button=5), 1)
        self.assertEqual(bf.wheel_steps(0), 0)

    def test_format_size(self):
        self.assertEqual(bf.format_size(None), "—")
        self.assertEqual(bf.format_size(512), "512 B")
        self.assertEqual(bf.format_size(1536), "1.5 KB")
        self.assertEqual(bf.format_size(5 * 1024 * 1024), "5.0 MB")

    def test_fit_dimensions_are_even(self):
        self.assertEqual(bf.fit_dimensions(1919, 1080, 0), (1918, 1080))
        self.assertEqual(bf.fit_dimensions(3840, 2160, 1920), (1920, 1080))
        self.assertEqual(bf.fit_dimensions(1920, 1080, 1280), (1280, 720))
        self.assertEqual(bf.fit_dimensions(1920, 1080, 854), (854, 480))
        self.assertEqual(bf.fit_dimensions(1080, 1920, 854), (480, 854))

    def test_estimate_shrinks_when_compressed(self):
        sharp = bf.estimate_output_bytes(1920, 1080, 300, 30, 18, 0)
        compact = bf.estimate_output_bytes(1920, 1080, 300, 30, 28, 0)
        small = bf.estimate_output_bytes(1920, 1080, 300, 30, 34, 854)
        self.assertGreater(sharp, compact)
        self.assertGreater(compact, small)

    def test_comparison_sentence(self):
        text = bf.comparison_sentence(20, 100, estimated=True)
        self.assertIn("20%", text)
        self.assertIn("80%", text)
        larger = bf.comparison_sentence(150, 100, estimated=False)
        self.assertIn("150%", larger)
        self.assertIn("Small", larger)


class BlurTests(unittest.TestCase):
    def test_clip_and_iou(self):
        self.assertIsNone(bf.clip_box((0, 0, 1, 1, 0), 100, 100))
        self.assertEqual(bf.clip_box((10, 10, 40, 40, 0.9), 100, 100), (10, 10, 40, 40))
        self.assertAlmostEqual(bf.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_blur_styles_change_the_face(self):
        frame = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
        boxed = frame.copy()
        bf.blur_box(boxed, (8, 8, 24, 24), "Black box", 0.2)
        self.assertEqual(int(boxed[20, 20, 0]), 0)
        pixel = frame.copy()
        bf.blur_box(pixel, (8, 8, 24, 24), "Pixelate", 0.2)
        self.assertFalse(np.array_equal(pixel, frame))


class RenderTests(unittest.TestCase):
    def test_export_compresses_and_scales(self):
        if not bf.find_ffmpeg():
            self.skipTest("ffmpeg is not available")
        tmp = tempfile.mkdtemp(prefix="blurface_test_")
        try:
            src = os.path.join(tmp, "in.mp4")
            writer = bf.cv2.VideoWriter(src, bf.cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
            if not writer.isOpened():
                self.skipTest("OpenCV could not write a sample video")
            rng = np.random.default_rng(0)
            for _ in range(16):
                writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
            writer.release()
            self.assertGreater(os.path.getsize(src), 1000)

            meta = bf.probe_video(src)
            self.assertEqual(meta["width"], 320)
            self.assertEqual(meta["height"], 240)
            self.assertGreater(meta["bytes"], 0)

            cancel = threading.Event()
            boxes = {i: [(40, 40, 80, 80)] for i in range(16)}

            def progress(_frac, _text):
                return None

            sharp = os.path.join(tmp, "sharp.mp4")
            small = os.path.join(tmp, "small.mp4")
            scaled = os.path.join(tmp, "scaled.mp4")
            sharp_res = bf.render_video(src, sharp, boxes, "Strong blur", 0.2, 25,
                                         progress, cancel, 16, crf=18, max_edge=0)
            small_res = bf.render_video(src, small, boxes, "Strong blur", 0.2, 25,
                                         progress, cancel, 16, crf=36, max_edge=0)
            scaled_res = bf.render_video(src, scaled, boxes, "Black box", 0.2, 25,
                                          progress, cancel, 16, crf=36, max_edge=160)
            self.assertEqual(sharp_res["frames"], 16)
            self.assertEqual(sharp_res["blurred_frames"], 16)
            self.assertEqual(sharp_res["output_bytes"], os.path.getsize(sharp))
            self.assertGreater(os.path.getsize(sharp), os.path.getsize(small))
            self.assertGreater(os.path.getsize(small), os.path.getsize(scaled))

            cap = bf.cv2.VideoCapture(scaled)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertLessEqual(max(frame.shape[:2]), 160)
            self.assertEqual(frame.shape[1] % 2, 0)
            self.assertLess(int(frame[40, 40, 0]), 12)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class WindowTests(unittest.TestCase):
    def test_window_shows_input_and_output_size(self):
        if bf.tk is None:
            self.skipTest("tkinter is not available")
        try:
            root = bf.tk.Tk()
        except bf.tk.TclError:
            self.skipTest("no display")
        root.withdraw()
        app = None
        try:
            app = bf.App(root)
            root.withdraw()
            self.assertGreater(float(app.quality_scale.cget("from")), float(app.quality_scale.cget("to")))
            app.meta = {
                "path": "/tmp/clip.mp4",
                "width": 1920,
                "height": 1080,
                "frames": 300,
                "fps": 30.0,
                "bytes": 50 * 1024 * 1024,
                "duration": 10.0,
            }
            app.refresh_sizes()
            root.update_idletasks()
            self.assertEqual(app.in_size.cget("text"), "50.0 MB")
            self.assertTrue(app.out_size.cget("text").startswith("~"))
            self.assertEqual(app.out_kind.cget("text"), "Estimated")
            app.crf.set(34)
            small = app._planned_output()[0]
            app.crf.set(18)
            sharp = app._planned_output()[0]
            self.assertGreater(sharp, small)
            app.output_actual = 8 * 1024 * 1024
            app.refresh_sizes()
            self.assertEqual(app.out_kind.cget("text"), "Saved")
            self.assertEqual(app.out_size.cget("text"), "8.0 MB")
            self.assertIn("MB", app.in_size.cget("text"))
        finally:
            if app is not None:
                app.on_close()
            elif root.winfo_exists():
                root.destroy()

    def test_page_scrolls_with_trackpad_delta(self):
        if bf.tk is None:
            self.skipTest("tkinter is not available")
        try:
            root = bf.tk.Tk()
        except bf.tk.TclError:
            self.skipTest("no display")
        root.withdraw()
        app = None
        try:
            app = bf.App(root)
            root.geometry("900x480")
            root.update_idletasks()
            root.update()
            app._sync_scrollregion()
            content_h = int(float(app.canvas.cget("scrollregion").split()[3]))
            self.assertGreater(content_h, app.canvas.winfo_height())
            app.canvas.yview_moveto(0)
            app._wheel(type("E", (), {"delta": -1, "num": 0})())
            self.assertGreater(app.canvas.yview()[0], 0)
            app._wheel(type("E", (), {"delta": 1, "num": 0})())
            self.assertEqual(app.canvas.yview()[0], 0)
        finally:
            if app is not None:
                app.on_close()
            elif root.winfo_exists():
                root.destroy()


if __name__ == "__main__":
    unittest.main()
