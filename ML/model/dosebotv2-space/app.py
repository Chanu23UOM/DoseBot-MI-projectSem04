import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model and label map
model = tf.keras.models.load_model("prescription_model.h5")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

with open("model_info.json", "r") as f:
    model_info = json.load(f)

IMG_W = model_info["img_w"]
IMG_H = model_info["img_h"]


def resize_with_padding(img, target_w, target_h, pad_color=(255, 255, 255)):
    """Match the training preprocessing: scale to fit, then pad (no warping)."""
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def predict_medicine(image):
    """Predict medicine name from handwritten prescription image."""
    if image is None:
        return {}, "Please upload an image"

    # Preprocess (aspect-ratio preserved, same as training)
    img = Image.fromarray(image).convert("RGB")
    img = resize_with_padding(img, IMG_W, IMG_H)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    top_k = 5
    top_indices = predictions.argsort()[-top_k:][::-1]

    results = {}
    details = []
    for i, idx in enumerate(top_indices):
        label = label_map[str(idx)]
        conf = float(predictions[idx])
        results[label] = conf
        details.append(f"{i+1}. {label}: {conf*100:.1f}%")

    top_medicine = label_map[str(top_indices[0])]
    top_conf = predictions[top_indices[0]] * 100
    summary = f"**Predicted Medicine:** {top_medicine} ({top_conf:.1f}% confidence)"

    return results, summary

# Build Gradio interface
with gr.Blocks(
    title="DoseBotV2 - Prescription Recognition",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
) as demo:
    gr.Markdown("""
    # 🤖💊 DoseBotV2 - Prescription Medicine Recognition
    Upload a handwritten prescription image to identify the medicine name.
    Trained on 78 medicine classes from the Doctor\'s Handwritten Prescription BD Dataset.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Prescription Image", type="numpy")
            predict_btn = gr.Button("🔍 Identify Medicine", variant="primary", size="lg")
            gr.Examples(
                examples=[],
                inputs=image_input,
                label="Try these examples"
            )

        with gr.Column(scale=1):
            label_output = gr.Label(label="Prediction Confidence", num_top_classes=5)
            text_output = gr.Markdown(label="Result")

    predict_btn.click(
        fn=predict_medicine,
        inputs=image_input,
        outputs=[label_output, text_output]
    )

    gr.Markdown(f"""
    ---
    **Model Info:** CNN trained on {model_info['num_classes']} medicine classes |
    Test Accuracy: {model_info['test_accuracy']*100:.1f}%
    """)

demo.launch()
