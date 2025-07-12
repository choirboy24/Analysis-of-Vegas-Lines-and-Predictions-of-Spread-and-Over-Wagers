import gradio as gr
from predict import pipeline_spread, pipeline_ou, pipeline_w
from joblib import load


# Replace these with actual pipeline objects
def predict_outcomes(champ_spread, champ_ou, champ_underdog):
    # Spread prediction
    pred_spread = pipeline_spread.predict(champ_spread)[0]
    pred_spread_text = 'Favorite covered the spread.' if pred_spread else 'Underdog covered the spread.'
    proba_spread = pipeline_spread.predict_proba(champ_spread)[0]
    spread_confidence = f"Favorite: {proba_spread[1]:.2%} | Underdog: {proba_spread[0]:.2%}"

    # Over/Under prediction
    pred_ou = pipeline_ou.predict(champ_ou)[0]
    pred_ou_text = 'The total went over.' if pred_ou else 'The total stayed under.'
    proba_ou = pipeline_ou.predict_proba(champ_ou)[0]
    ou_confidence = f"Over: {proba_ou[1]:.2%} | Under: {proba_ou[0]:.2%}"

    # Underdog win prediction
    pred_dog = pipeline_w.predict(champ_underdog)[0]
    pred_dog_text = 'Underdog won outright.' if pred_dog else 'Favorite secured the win.'
    proba_dog = pipeline_w.predict_proba(champ_underdog)[0]
    dog_confidence = f"Underdog: {proba_dog[1]:.2%} | Favorite: {proba_dog[0]:.2%}"

    # Return formatted output
    return (
        f"🏈 Spread Prediction:\n➡️ {pred_spread_text}\nConfidence — {spread_confidence}",
        f"📈 Total Points Prediction:\n➡️ {pred_ou_text}\nConfidence — {ou_confidence}",
        f"🐶 Underdog Win Prediction:\n➡️ {pred_dog_text}\nConfidence — {dog_confidence}"
    )


# Define interface (dummy arrays for input; replace with proper preprocessing if needed)
iface = gr.Interface(
    fn=predict_outcomes,
    inputs=["dataframe", "dataframe", "dataframe"],  # or whatever format your inputs take
    outputs=["text", "text", "text"],
    title="College Football Game Outcome Predictor",
    description="Submit game data and get predictions for spread, total, and outright winner.",
    theme="default"
)

iface.launch()