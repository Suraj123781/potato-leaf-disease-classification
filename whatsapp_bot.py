import os
import io
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
load_dotenv()
app = Flask(__name__)

model = tf.keras.models.load_model("potato_disease_model.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
SUGGESTIONS = {
    "Early Blight": "🛡 Use fungicides like chlorothalonil or mancozeb. Remove infected leaves and rotate crops.",
    "Late Blight": "🧪 Apply copper-based fungicides. Avoid overhead watering and improve air circulation.",
    "Healthy": "✅ No action needed. Maintain regular monitoring and good soil health."
}

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100
        return predicted_class, confidence
    except Exception as e:
        print("Error processing image:", e)
        return None, None

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    try:
        incoming_msg = request.values.get("Body", "").strip().lower()
        num_media = int(request.values.get("NumMedia", 0))
        resp = MessagingResponse()
        msg = resp.message()

        if num_media > 0:
            media_url = request.values.get("MediaUrl0")
            headers = {"User-Agent": "TwilioBot/1.0"}
            image_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), headers=headers)

            if image_response.status_code == 200:
                predicted_class, confidence = predict_image(image_response.content)
                if predicted_class:
                    recommendation = SUGGESTIONS[predicted_class]
                    msg.body(f"✅ The leaf appears to be: *{predicted_class}* ({confidence:.2f}% confidence)\n\n{recommendation}")
                else:
                    msg.body("⚠ Error: Could not process the image. Please try another one.")
            else:
                msg.body("⚠ Error downloading image. Please resend.")
            return str(resp)

        if "hi" in incoming_msg or "hello" in incoming_msg:
            msg.body("👋 Hello! Send me a *potato leaf image*, and I'll tell you if it's *Early Blight*, *Late Blight*, or *Healthy*. 🌿")
        elif "predict" in incoming_msg:
            msg.body("📸 Please send a potato leaf image to predict its health.")
        else:
            msg.body("🤖 I didn't understand that. Send a leaf image or say 'hi'.")
        return str(resp)

    except Exception as e:
        print("WhatsApp bot error:", e)
        return "Error", 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)