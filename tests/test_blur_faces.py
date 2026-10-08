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

    def test_rejects_objects_that_are_not_faces(self):
        def row(x, y, w, h, eyes_below=False):
            eye_y = 0.72 if eyes_below else 0.33
            mouth_y = 0.34 if eyes_below else 0.75
            return [
                x, y, w, h,
                x + 0.32 * w, y + eye_y * h,
                x + 0.68 * w, y + (eye_y + 0.02) * h,
                x + 0.50 * w, y + 0.52 * h,
                x + 0.38 * w, y + mouth_y * h,
                x + 0.62 * w, y + (mouth_y + 0.02) * h,
                0.9,
            ]

        good_box = (10, 10, 80, 100)
        self.assertTrue(bf.face_is_plausible(row(*good_box), good_box))
        wide = (0, 0, 220, 50)
        self.assertFalse(bf.face_is_plausible(row(*wide), wide))
        self.assertFalse(bf.face_is_plausible(row(10, 10, 80, 100, eyes_below=True), (10, 10, 80, 100)))
        self.assertEqual(bf.resolve_shape("Auto", (0, 0, 40, 80)), "Ellipse")
        self.assertEqual(bf.resolve_shape("Auto", (0, 0, 50, 50)), "Circle")
        self.assertEqual(bf.resolve_shape("Rectangle", (0, 0, 50, 50)), "Rectangle")

    def test_hand_under_a_face_and_a_teal_object_are_rejected(self):
        face = (100, 40, 80, 100, 0.93)
        hand = (110, 180, 90, 70)
        self.assertTrue(bf.body_false_positive(hand, 0.74, [face]))
        beside = (320, 50, 80, 100)
        self.assertFalse(bf.body_false_positive(beside, 0.8, [face]))
        skin = np.zeros((80, 80, 3), np.uint8)
        skin[:, :, 0] = 150
        skin[:, :, 1] = 155
        skin[:, :, 2] = 110
        skin = bf.cv2.cvtColor(skin, bf.cv2.COLOR_YCrCb2BGR)
        self.assertTrue(bf.looks_like_skin(skin, (8, 8, 64, 64)))
        teal = np.zeros((80, 80, 3), np.uint8)
        teal[:] = (170, 130, 30)
        self.assertFalse(bf.looks_like_skin(teal, (8, 8, 64, 64)))

    def test_weak_object_track_is_not_blurred(self):
        track = type("T", (), {})()
        track.dets = [type("D", (), {"score": score})() for score in (0.62, 0.64, 0.63)]
        self.assertFalse(bf.track_should_blur(track, 30))
        strong = type("T", (), {})()
        strong.dets = [type("D", (), {"score": 0.95})()]
        self.assertTrue(bf.track_should_blur(strong, 30))

    def test_blur_fades_outside_the_face(self):
        frame = np.full((90, 90, 3), 240, np.uint8)
        before = frame.copy()
        bf.blur_box(frame, (28, 22, 34, 46), "Black box", 0.2, "Ellipse")
        self.assertEqual(int(frame[1, 1, 0]), 240)
        self.assertLess(int(frame[45, 45, 0]), 30)
        faded = ((frame[:, :, 0] > 30) & (frame[:, :, 0] < 230)).any()
        self.assertTrue(faded)
        self.assertFalse(np.array_equal(frame, before))

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
            binding = app.root.tk.call("bind", "all", "<MouseWheel>")
            self.assertIn("%D", binding)
            self.assertIn(app.canvas._w, binding)
            if app.root.tk.call("info", "commands", "::tk::PreciseScrollDeltas"):
                touch = app.root.tk.call("bind", "all", "<TouchpadScroll>")
                self.assertIn("PreciseScrollDeltas", touch)
                self.assertIn(app.canvas._w, touch)
                app.canvas.yview_moveto(0.2)
                before_touch = app.canvas.yview()[0]
                packed = (-24) & 0xFFFF
                app.root.tk.eval(touch.replace("%D", str(packed)))
                self.assertGreater(app.canvas.yview()[0], before_touch)
            app.canvas.yview_moveto(0.2)
            before = app.canvas.yview()[0]
            # A fractional mouse-wheel delta. Python's event.delta would become 0.
            script = binding.replace("%D", "-2.4")
            app.root.tk.eval(script)
            app.root.tk.eval(script)
            self.assertGreater(app.canvas.yview()[0], before)
        finally:
            if app is not None:
                app.on_close()
            elif root.winfo_exists():
                root.destroy()


if __name__ == "__main__":
    unittest.main()
