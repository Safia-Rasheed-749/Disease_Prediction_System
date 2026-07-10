import gradio as gr
import joblib
import pickle
import numpy as np
import os
# import sys
PORT = int(os.environ.get("PORT", 7860))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("Loading models...")
model = joblib.load(os.path.join(MODELS_DIR, 'disease_model.joblib'))
print("Model loaded")

le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
print("Label Encoder loaded")

with open(os.path.join(MODELS_DIR, 'all_symptoms.pkl'), 'rb') as f:
    loaded_symptoms = pickle.load(f)

# Sanitize and normalize loaded symptoms to match training-time token format
# (remove surrounding whitespace; drop non-strings/empty)
FEATURE_NAMES = sorted(
    {
        str(s).strip()
        for s in loaded_symptoms
        if isinstance(s, str) and s not in [0, None] and str(s).strip()
    }
)

print(f"{len(FEATURE_NAMES)} symptoms loaded")
print("All models loaded successfully!")

def _build_input_vector(selected_symptoms_set: set[str]):
    # Create binary vector aligned with FEATURE_NAMES order
    input_vector = np.zeros(len(FEATURE_NAMES), dtype=np.uint8)
    for i, sym in enumerate(FEATURE_NAMES):
        if sym in selected_symptoms_set:
            input_vector[i] = 1
    return input_vector


def predict_disease(selected_symptoms):
    selected_symptoms_set = {str(s).strip() for s in (selected_symptoms or []) if isinstance(s, str) and str(s).strip()}
    # Ignore any symptom not in FEATURE_NAMES (prevents Gradio/choice mismatch issues)
    selected_symptoms_set = {s for s in selected_symptoms_set if s in set(FEATURE_NAMES)}

    input_vector = _build_input_vector(selected_symptoms_set).reshape(1, -1)

    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]

    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = le.inverse_transform(top_3_idx)
    top_3_probs = probabilities[top_3_idx]

    disease_name = le.inverse_transform([prediction])[0]
    confidence = float(probabilities[prediction]) * 100

    result = f"## 🏥 Predicted Disease: **{disease_name}**\n"
    result += f"### Confidence: **{confidence:.1f}%**\n\n"
    result += "---\n"
    result += "### 🔍 Top 3 Possible Diseases:\n\n"

    for i, (disease, prob) in enumerate(zip(top_3_diseases, top_3_probs), 1):
        bar = "█" * int(prob * 50)
        result += f"{i}. **{disease}** - {prob * 100:.1f}%\n"
        result += f"   `{bar}`\n\n"

    return result


def suggest_symptoms(query: str, limit: int = 40):
    q = (query or "").strip().lower()
    if not q:
        return []
    # Only suggest values that are guaranteed to exist in FEATURE_NAMES
    matches = [s for s in FEATURE_NAMES if q in s.lower()]
    return matches[:limit]


with gr.Blocks(title="Disease Prediction System") as demo:
    gr.Markdown(
        """# 🏥 Disease Prediction System
**Search symptoms (type keyword) → click suggestions to select → Predict**
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            search = gr.Textbox(
                label="Search symptom",
                placeholder="Type: cough, fever, chest pain, breathlessness...",
            )

            search_btn = gr.Button("🔎 Show matches")

            # Suggestions must be subset of choices; choices are set dynamically via update(choices=...)
            suggestions = gr.CheckboxGroup(
                label="Suggestions",
                choices=[],
                interactive=True,
            )

            # Selected must have choices that include any value the user can select.
            # We keep selected choices in sync with suggestions merging.
            selected = gr.CheckboxGroup(
                label="Selected symptoms",
                choices=[],
                interactive=True,
            )

            clear_btn = gr.Button("Clear selected")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Prediction Result")
            predict_btn = gr.Button("🔮 Predict Disease", variant="primary")

    def _update_suggestions(q):
        return gr.update(choices=suggest_symptoms(q), value=[])  # reset any invalid previously checked values

    search_btn.click(fn=_update_suggestions, inputs=[search], outputs=[suggestions])

    # Merge suggestions into selected safely.
    def _merge_selected(prev_selected, new_suggestions):
        prev_selected = prev_selected or []
        new_suggestions = new_suggestions or []

        # Only keep values that exist in FEATURE_NAMES
        prev_selected = [s for s in prev_selected if s in FEATURE_NAMES]
        new_suggestions = [s for s in new_suggestions if s in FEATURE_NAMES]

        merged = sorted(set(prev_selected).union(set(new_suggestions)))
        # Set both choices and value to the same merged list.
        return gr.update(choices=merged, value=merged)

    # suggestions.change provides selected, suggestions.checked values
    suggestions.change(
        fn=_merge_selected,
        inputs=[selected, suggestions],
        outputs=[selected],
    )

    def _clear():
        return gr.update(choices=[], value=[])

    clear_btn.click(fn=_clear, inputs=[], outputs=[selected])

    predict_btn.click(fn=predict_disease, inputs=[selected], outputs=[output])
if __name__ == "__main__":
    demo.launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            share=True
        )
