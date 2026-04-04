import cv2
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry

# ── config ──────────────────────────────────────────────────────────
SAM_CKPT  = "checkpoints/sam_vit_b.pth"
RAW_ROOT  = "dataset/flat"       # ← flattened images
MASK_ROOT = "dataset/masks"      # ← masks will be saved here
CLASSES   = ["bacterial", "fungal", "healthy", "mould"]   # ← 4 classes
IMG_SIZE  = 512
WIN_NAME  = "annotate  |  L-click=lesion  R=reset  U=undo  S=save  N=skip  Q=quit"
# ────────────────────────────────────────────────────────────────────

sam       = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
predictor = SamPredictor(sam)

click_points = []
current_img  = None
current_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# ── mouse callback ───────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global click_points, current_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x, y])
        run_sam()
    elif event == cv2.EVENT_RBUTTONDOWN:   # right-click = reset
        click_points = []
        current_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        show_overlay()

# ── SAM inference ────────────────────────────────────────────────────
def run_sam():
    global current_mask
    if not click_points:
        return
    pts    = np.array(click_points)
    labels = np.ones(len(pts), dtype=int)    # all clicks = foreground

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=True
    )
    best_mask    = masks[np.argmax(scores)]
    current_mask = (best_mask * 255).astype(np.uint8)
    show_overlay()

# ── overlay display ──────────────────────────────────────────────────
def show_overlay():
    overlay = current_img.copy()

    # green highlight on lesion region
    overlay[current_mask > 0] = (0, 200, 100)
    blended = cv2.addWeighted(current_img, 0.55, overlay, 0.45, 0)

    # draw each click point as a small dot
    for pt in click_points:
        cv2.circle(blended, tuple(pt), 5, (0, 0, 255), -1)   # red dot

    # lesion pixel count in top-left corner
    px_count = int(current_mask.sum() // 255)
    cv2.putText(blended, f"lesion px: {px_count}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    cv2.imshow(WIN_NAME, blended)

# ── per-class annotation loop ────────────────────────────────────────
def annotate_class(cls):
    global click_points, current_mask, current_img

    raw_dir  = os.path.join(RAW_ROOT,  cls)
    mask_dir = os.path.join(MASK_ROOT, cls)
    os.makedirs(mask_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"  WARNING: {raw_dir} not found — skipping")
        return

    # only process image files
    all_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    ])

    total   = len(all_files)
    done    = 0
    skipped = 0

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)

    for idx, fname in enumerate(all_files, start=1):
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

        # resume — skip already annotated
        if os.path.exists(mask_path):
            print(f"  [{idx}/{total}] skip (done): {fname}")
            done += 1
            continue

        img_path = os.path.join(raw_dir, fname)
        img_bgr  = cv2.imread(img_path)

        if img_bgr is None:
            print(f"  [{idx}/{total}] WARNING: cannot open {fname}")
            continue

        img_bgr  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        current_img  = img_bgr.copy()
        current_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        click_points = []

        print(f"\n  [{idx}/{total}]  {fname}")

        # ── healthy class: blank mask, no SAM needed ─────────────────
        if cls == "healthy":
            cv2.imwrite(mask_path, current_mask)
            print(f"  saved blank mask (healthy): {fname}")
            done += 1
            continue

        # ── diseased classes: interactive SAM annotation ─────────────
        predictor.set_image(img_rgb)
        show_overlay()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):                      # SAVE
                cv2.imwrite(mask_path, current_mask)
                px = int(current_mask.sum() // 255)
                print(f"  saved: {fname}  |  lesion px: {px}")
                done += 1
                break

            elif key == ord('r') or key == 2:        # RESET (r or right-click)
                click_points = []
                current_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                show_overlay()
                print("  reset — click again")

            elif key == ord('u'):                    # UNDO last point
                if click_points:
                    click_points.pop()
                    run_sam() if click_points else show_overlay()
                    print(f"  undo — {len(click_points)} points remaining")

            elif key == ord('n'):                    # SKIP this image
                print(f"  skipped: {fname}")
                skipped += 1
                break

            elif key == ord('q'):                    # QUIT entire session
                print(f"\n  Quit at {fname}. Progress saved.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\n  {cls} done — annotated: {done}  skipped: {skipped}  total: {total}")

# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SAM Annotation Tool")
    print("  Controls:")
    print("    L-click  → add foreground point on lesion")
    print("    R-click  → reset all points")
    print("    U        → undo last point")
    print("    S        → save mask and go to next image")
    print("    N        → skip image (annotate later)")
    print("    Q        → quit and resume later")
    print("=" * 55)

    # create mask folder structure upfront
    for cls in CLASSES:
        os.makedirs(os.path.join(MASK_ROOT, cls), exist_ok=True)

    for cls in CLASSES:
        print(f"\n{'='*55}")
        print(f"  Class: {cls.upper()}")
        print(f"{'='*55}")
        annotate_class(cls)

    print("\n\nAnnotation complete.")
    print(f"Masks saved to → {MASK_ROOT}/")

    # ── final mask count summary ──────────────────────────────────────
    print("\n── Mask counts ─────────────────────────────────────")
    for cls in CLASSES:
        mask_dir = os.path.join(MASK_ROOT, cls)
        n = len(os.listdir(mask_dir)) if os.path.exists(mask_dir) else 0
        raw_dir  = os.path.join(RAW_ROOT, cls)
        total    = len([f for f in os.listdir(raw_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
                       ]) if os.path.exists(raw_dir) else 0
        pct = (n / total * 100) if total > 0 else 0
        bar = "█" * (n // 20)
        print(f"  {cls:<12s}  {n:>4}/{total:<4} ({pct:5.1f}%)  {bar}")
    print("────────────────────────────────────────────────────")