import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_LOG = "_watch_review.json"
REJECTED_DIR = "_filtered_out"


class WatchReviewer:
    def __init__(self, root, dataset_root):
        self.root = root
        self.root.title("Analog Watch Dataset Reviewer")
        self.root.geometry("1100x800")
        self.root.configure(bg="#202020")

        self.dataset_root = Path(dataset_root)
        self.log_path = self.dataset_root / REVIEW_LOG
        self.rejected_root = self.dataset_root / REJECTED_DIR

        self.reviewed = self.load_log()
        self.items = self.collect_images()
        self.index = 0
        self.photo = None

        self.info = tk.Label(root, text="", fg="white", bg="#202020",
                             font=("Segoe UI", 12))
        self.info.pack(pady=(10, 5))

        self.image_label = tk.Label(root, bg="#202020")
        self.image_label.pack(expand=True, fill="both", padx=20, pady=10)

        self.help_label = tk.Label(
            root,
            text="A = KEEP    D = REMOVE (move to _filtered_out)    S = SKIP    Q = QUIT",
            fg="white", bg="#202020", font=("Segoe UI", 14, "bold")
        )
        self.help_label.pack(pady=(5, 15))

        for key, action in [
            ("a", "keep"), ("A", "keep"),
            ("d", "remove"), ("D", "remove"),
            ("s", "skip"), ("S", "skip")
        ]:
            root.bind(f"<KeyPress-{key}>", lambda e, a=action: self.decide(a))
        root.bind("<KeyPress-q>", lambda e: self.quit())
        root.bind("<KeyPress-Q>", lambda e: self.quit())

        if not self.items:
            messagebox.showinfo(
                "No images",
                "No unreviewed images were found in train/images, valid/images, or test/images."
            )
            root.destroy()
            return

        self.show_current()

    def load_log(self):
        if not self.log_path.exists():
            return {}
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_log(self):
        temp = self.log_path.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(self.reviewed, f, indent=2)
        temp.replace(self.log_path)

    def collect_images(self):
        items = []
        for split in ("train", "valid", "test"):
            image_dir = self.dataset_root / split / "images"
            if not image_dir.exists():
                continue

            for p in sorted(image_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    key = str(p.relative_to(self.dataset_root)).replace("\\", "/")
                    if key not in self.reviewed:
                        items.append((split, p))
        return items

    def show_current(self):
        if self.index >= len(self.items):
            self.info.config(text="Review complete!")
            self.image_label.config(
                image="",
                text="All currently unreviewed images have been processed.",
                fg="white", font=("Segoe UI", 18, "bold")
            )
            return

        split, path = self.items[self.index]
        self.info.config(
            text=f"{self.index + 1} / {len(self.items)}    {split}    {path.name}"
        )

        try:
            img = Image.open(path).convert("RGB")
            max_w, max_h = 1000, 650
            scale = min(max_w / img.width, max_h / img.height, 1.0)
            size = (max(1, int(img.width * scale)),
                    max(1, int(img.height * scale)))
            img = img.resize(size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            self.image_label.config(
                image="", text=f"Could not open image:\n{e}",
                fg="red", font=("Segoe UI", 14)
            )

    def decide(self, decision):
        if self.index >= len(self.items):
            return

        split, path = self.items[self.index]
        key = str(path.relative_to(self.dataset_root)).replace("\\", "/")

        if decision == "remove":
            label = self.dataset_root / split / "labels" / (path.stem + ".txt")
            dest_img_dir = self.rejected_root / split / "images"
            dest_lbl_dir = self.rejected_root / split / "labels"
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            shutil.move(str(path), str(dest_img_dir / path.name))

            if label.exists():
                shutil.move(str(label), str(dest_lbl_dir / label.name))

        self.reviewed[key] = decision
        self.save_log()
        self.index += 1
        self.show_current()

    def quit(self):
        self.save_log()
        self.root.destroy()


def choose_dataset():
    temp = tk.Tk()
    temp.withdraw()
    folder = filedialog.askdirectory(
        title="Select wrist-watch.v4i.yolov8 dataset folder"
    )
    temp.destroy()
    return folder


if __name__ == "__main__":
    dataset = choose_dataset()
    if not dataset:
        raise SystemExit("No dataset folder selected.")

    root = tk.Tk()
    WatchReviewer(root, dataset)
    root.mainloop()
