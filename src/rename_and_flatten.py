import os, shutil

SRC_ROOT = "dataset/raw"       
DST_ROOT = "dataset/flat"      

CLASSES = ["Bacterial", "Fungal", "Healthy", "Mould"]

def sanitize(name):
    return name.strip().replace(" - ", "_").replace(" ", "_").replace("-", "_")

def rename_and_flatten():
    for cls in CLASSES:
        src_cls = os.path.join(SRC_ROOT, cls)
        dst_cls = os.path.join(DST_ROOT, cls.lower())
        os.makedirs(dst_cls, exist_ok=True)

        if not os.path.exists(src_cls):
            print(f"  WARNING: {src_cls} not found — skipping")
            continue

        print(f"\n── {cls} ──────────────────────────────────")
        total_copied = 0

        subfolders = sorted([
            s for s in os.listdir(src_cls)
            if os.path.isdir(os.path.join(src_cls, s))
        ])

        for sub in subfolders:
            sub_path = os.path.join(src_cls, sub)
            prefix   = sanitize(sub)

            images = sorted([
                f for f in os.listdir(sub_path)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
            ])

            sub_count = 0
            for serial, fname in enumerate(images, start=1):
                ext      = os.path.splitext(fname)[1].lower()
                new_name = f"{prefix}_{serial:04d}{ext}"
                src_file = os.path.join(sub_path, fname)
                dst_file = os.path.join(dst_cls, new_name)
                shutil.copy2(src_file, dst_file)
                sub_count += 1

            print(f"   {sub:<45s} → {sub_count:>4} images  [{prefix}_XXXX]")
            total_copied += sub_count

        print(f"   {'─'*55}")
        print(f"   Total in flat/{cls.lower()}/: {total_copied} images")

    print("\n\n── Final class counts ──────────────────────────────")
    grand_total = 0
    for cls in CLASSES:
        dst_cls = os.path.join(DST_ROOT, cls.lower())
        n = len(os.listdir(dst_cls)) if os.path.exists(dst_cls) else 0
        bar = "█" * (n // 20)
        print(f"  {cls:<12s}  {n:>5} images  {bar}")
        grand_total += n
    print(f"  {'─'*45}")
    print(f"  {'Total':<12}  {grand_total:>5} images")
    print("────────────────────────────────────────────────────")
    print(f"\nOutput saved to → {DST_ROOT}/")

if __name__ == "__main__":
    rename_and_flatten()