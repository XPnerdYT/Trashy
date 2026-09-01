import cv2
import torch
import numpy as np
import open_clip

from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

trashy_yolo = YOLO("yolo11n.pt")

trashy_clip, _, trashy_clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
trashy_clip = trashy_clip.to(DEVICE).eval()
trashy_tokenizer = open_clip.get_tokenizer("ViT-B-32")

CATEGORY_PROMPTS = {
    "Hazardous": [
        "a photo of a lithium battery",
        "a photo of a 18650 battery",
        "a photo of a AA or AAA battery",
        "a photo of a used battery",
        "a photo of electronic waste",
        "a photo of a smartphone",
        "a photo of a laptop computer",
    ],
    "Recycling": [
        "a photo of a sheet of paper",
        "a photo of a cardboard box",
        "a photo of a plastic bottle",
        "a photo of a glass bottle",
        "a photo of an aluminum can",
        "a photo of a paper cup",
        "a photo of a book or magazine",
    ],
    "Compost": [
        "a photo of a banana peel",
        "a photo of an apple core",
        "a photo of food scraps",
        "a photo of a half-eaten sandwich",
        "a photo of an orange peel",
        "a photo of coffee grounds",
    ],
    "Garbage": [
        "a photo of a used tissue",
        "a photo of a broken plastic object",
        "a photo of a plastic wrapper",
        "a photo of scissors",
        "a photo of general non-recyclable trash",
    ],
}

MATERIAL_PROMPTS = {
    "Recycling": [
        "a photo of paper material",
        "a photo of cardboard material",
        "a photo of plastic material",
        "a photo of metal material",
        "a photo of glass material",
        "a photo of an aluminum surface",
    ],
    "Compost": [
        "a photo of organic food waste",
        "a photo of a fruit or vegetable scrap",
        "a photo of biodegradable plant material",
    ],
    "Hazardous": [
        "a photo of an electronic device",
        "a photo of a battery-shaped object",
    ],
    "Garbage": [
        "a photo of fabric or textile material",
        "a photo of styrofoam material",
        "a photo of mixed non-recyclable material",
    ],
}


def _flatten(prompt_dict):
    prompts, categories = [], []
    for cat, plist in prompt_dict.items():
        for p in plist:
            prompts.append(p)
            categories.append(cat)
    return prompts, categories


SPECIFIC_PROMPTS, SPECIFIC_CATEGORY = _flatten(CATEGORY_PROMPTS)
MATERIAL_PROMPT_LIST, MATERIAL_CATEGORY = _flatten(MATERIAL_PROMPTS)


def _encode_text(prompts):
    with torch.no_grad():
        tokens = trashy_tokenizer(prompts).to(DEVICE)
        feats = trashy_clip.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


SPECIFIC_TEXT_FEATURES = _encode_text(SPECIFIC_PROMPTS)
MATERIAL_TEXT_FEATURES = _encode_text(MATERIAL_PROMPT_LIST)

SPECIFIC_CONFIDENCE_THRESHOLD = 0.28
MATERIAL_CONFIDENCE_THRESHOLD = 0.20

COLOR_MAPPING = {
    "Compost": (0, 230, 0),
    "Recycling": (255, 200, 0),
    "Hazardous": (200, 0, 200),
    "Garbage": (50, 80, 240),
    "Unsure": (160, 160, 160),
}


def _best_match(image_features, text_features, prompts, categories):
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    best_idx = int(similarity.argmax().item())
    confidence = float(similarity[0, best_idx].item())
    return categories[best_idx], prompts[best_idx], confidence


def classify_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = _to_pil(rgb)
    image_input = trashy_clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = trashy_clip.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        category, prompt, confidence = _best_match(
            image_features, SPECIFIC_TEXT_FEATURES, SPECIFIC_PROMPTS, SPECIFIC_CATEGORY
        )
        if confidence >= SPECIFIC_CONFIDENCE_THRESHOLD:
            return category, prompt, confidence, "specific"

        m_category, m_prompt, m_confidence = _best_match(
            image_features, MATERIAL_TEXT_FEATURES, MATERIAL_PROMPT_LIST, MATERIAL_CATEGORY
        )
        if m_confidence >= MATERIAL_CONFIDENCE_THRESHOLD:
            return m_category, m_prompt, m_confidence, "material"

    return "Unsure", prompt, confidence, "unsure"


def _to_pil(rgb_array):
    from PIL import Image
    return Image.fromarray(rgb_array)


def _format_label(category, prompt, confidence, tier):
    label_text = (
        prompt.replace("a photo of ", "")
        .replace("an ", "", 1)
        .replace("a ", "", 1)
    )
    if tier == "specific":
        return f"{category} ({label_text}) {confidence * 100:.0f}%"
    if tier == "material":
        return f"{category}? ({label_text}, unsure) {confidence * 100:.0f}%"
    return f"Unsure {confidence * 100:.0f}%"


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open any webcam. Check camera index / drivers.")

FRAME_SKIP = 2
frame_count = 0
last_results_cache = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_h, frame_w = frame.shape[:2]
    total_area = frame_h * frame_w
    frame_count += 1

    run_detection_this_frame = (frame_count % FRAME_SKIP == 0) or not last_results_cache

    if run_detection_this_frame:
        results = trashy_yolo(frame, conf=0.35, iou=0.35, verbose=False)

        new_cache = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_label = trashy_yolo.names[int(box.cls[0])].lower()

                if raw_label == "person":
                    continue

                box_w, box_h = x2 - x1, y2 - y1
                if box_w <= 0 or box_h <= 0:
                    continue
                if (box_w * box_h) > (total_area * 0.60):
                    continue

                crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                if crop.size == 0:
                    continue

                category, prompt, confidence, tier = classify_crop(crop)
                new_cache.append((x1, y1, x2, y2, category, prompt, confidence, tier))

        last_results_cache = new_cache

    for (x1, y1, x2, y2, category, prompt, confidence, tier) in last_results_cache:
        color = COLOR_MAPPING.get(category, COLOR_MAPPING["Unsure"])
        display_text = _format_label(category, prompt, confidence, tier)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(y1 - 10, 20)
        cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("Trashy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
