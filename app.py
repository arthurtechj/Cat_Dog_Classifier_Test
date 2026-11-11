import gradio as gr

from predict import load_model, predict_pil_image
from utils.device import get_device


def classify_image(image, checkpoint_path):
    if image is None:
        return "Add an image first."

    if not checkpoint_path:
        return "Add the path to a saved model.pt file."

    try:
        device = get_device()
        model, class_names, transform = load_model(checkpoint_path, device)
        result = predict_pil_image(image, model, class_names, transform, device)
        return f"{result['prediction']} ({result['confidence']:.2%})"
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return f"Could not classify image: {error}"


with gr.Blocks() as demo:
    gr.Markdown("# Cat or Dog Classifier")
    gr.Markdown("Upload an image and give the path to a saved `model.pt` checkpoint.")

    image_input = gr.Image(type="pil", label="Image")
    checkpoint_input = gr.Textbox(label="Checkpoint path")
    output = gr.Textbox(label="Prediction")
    predict_button = gr.Button("Classify")

    predict_button.click(
        classify_image,
        inputs=[image_input, checkpoint_input],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch()
