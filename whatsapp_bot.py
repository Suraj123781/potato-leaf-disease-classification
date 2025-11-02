import os
import io
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

model = tf.keras.models.load_model("potato_disease_model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((128, 128))  # ✅ Match training size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100
        return f"{predicted_class} ({confidence:.2f}% confidence)"
    except Exception as e:
        print("Error processing image:", e)
        return None

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
                prediction = predict_image(image_response.content)
                if prediction:
                    msg.body(f"✅ The leaf appears to be: *{prediction}* 🍃")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)