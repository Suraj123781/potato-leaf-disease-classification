import os
import io
import requests   # <-- critical import
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from dotenv import load_dotenv

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load environment variables
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

print("🔑 SID:", TWILIO_ACCOUNT_SID)
print("🔑 TOKEN:", "Loaded" if TWILIO_AUTH_TOKEN else "Missing")

app = Flask(__name__)

# -----------------------------
# Model Configuration
# -----------------------------
IMG_SIZE = (224, 224)  # Must match training configuration

# -----------------------------
# Safe Model Load
# -----------------------------
try:
    # Get model path from environment variable or use default
    model_path = os.getenv('MODEL_PATH', 'models/potato_model.h5')
    print(f"🔍 Loading model from: {model_path}")
    print(f"🔍 Absolute path: {os.path.abspath(model_path)}")
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
    
    # Load the model with custom objects if needed
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Set to evaluation mode
    model.trainable = False
    
    # Print model information
    print("✅ Model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
except Exception as e:
    print("❌ Model load failed:", str(e))
    import traceback
    traceback.print_exc()
    model = None

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

SUGGESTIONS = {
    "Early Blight": (
        "🛡 Use fungicides like chlorothalonil or mancozeb. Remove infected leaves and rotate crops.\n\n"
        "🛒 Buy online:\n"
        "- AgriBegri: https://agribegri.com/products/shivalik-zee-l-fungicide.php\n"
        "- BigHaat: https://www.bighaat.com/collections/management-of-early-blight-in-tomato-and-potato\n"
        "- Amazon: https://www.amazon.in/Blitox-RALLIS-Copper-Oxychloride-Fungicide/dp/B0CKW9LGL1\n"
        "- UPL Blitox via AgriBegri: https://agribegri.com/products/blitox-fungicide.php"
    ),
    "Late Blight": (
        "🧪 Apply copper-based fungicides. Avoid overhead watering and improve air circulation.\n\n"
        "🛒 Buy online:\n"
        "- BharatAgri: https://krushidukan.bharatagri.com/en/collections/late-blight-disease-products-online\n"
        "- BigHaat: https://www.bighaat.com/collections/late-blight-disease-management-in-tomato-and-potato-crops\n"
        "- AgriBegri: https://agribegri.com/en/products/buy-sumitomo-kemoxyl-metalaxyl-8--mancozeb-64-wp-fungicide-online.php\n"
        "- Amazon: https://www.amazon.in/Katyayani-Blight-Metalaxyl-M-Chlorothalonil-Fast-Acting/dp/B0FT3TQX58"
    ),
    "Healthy": "✅ No action needed. Maintain regular monitoring and good soil health."
}

# Store last prediction per user
last_prediction = {}

def preprocess_image(image_bytes):
    """Preprocess the image to match training configuration"""
    # 1. Open and convert image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"✅ Loaded image with size: {img.size} and mode: {img.mode}")
    
    # 2. Resize to expected input shape
    print(f"🔄 Resizing to: {IMG_SIZE}")
    img_resized = img.resize(IMG_SIZE)
    
    # 3. Convert to array and apply EfficientNet preprocessing
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_bytes):
    if model is None:
        print("❌ Model not available")
        return None, None
    try:
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]  # Get first (and only) batch
        print(f"Raw predictions: {predictions}")
        
        # Get class with highest probability
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Prepare results
        result = {
            "Early Blight": float(predictions[0]) * 100,
            "Late Blight": float(predictions[1]) * 100,
            "Healthy": float(predictions[2]) * 100
        }
        
        print(f"🎯 Prediction: {predicted_class} (Confidence: {confidence:.2%})")
        print(f"All confidences: {result}")
        
        return predicted_class, result
        
    except Exception as e:
        error_msg = f"❌ Error in predict_image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, {"Error": error_msg}
        
@app.route("/", methods=["GET"])
def home():
    return "✅ Potato Leaf Disease Classifier is running.", 200

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/whatsapp/status", methods=["POST"])
def status_callback():
    """Handle status callbacks for WhatsApp message delivery"""
    try:
        message_sid = request.values.get('MessageSid', '')
        message_status = request.values.get('MessageStatus', '')
        error_code = request.values.get('ErrorCode', '')
        error_message = request.values.get('ErrorMessage', '')
        
        print(f"📤 Message SID: {message_sid}")
        print(f"📤 Status: {message_status}")
        if error_code:
            print(f"❌ Error {error_code}: {error_message}")
            
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"❌ Status callback error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    try:
        # Get request data
        sender = request.values.get("From", "")
        incoming_msg = request.values.get("Body", "").strip().lower()
        num_media = int(request.values.get("NumMedia", 0))
        
        # Log incoming request
        print(f"\n📨 New request from: {sender}")
        print(f"💬 Message: {incoming_msg}")
        print(f"📷 Media count: {num_media}")
        
        # Create new response object
        resp = MessagingResponse()

        print(f"📨 From: {sender}")
        print(f"💬 Message: {incoming_msg}")
        print(f"📷 Media count: {num_media}")

        # Step 1: User uploads image
        if num_media > 0:
            media_url = request.values.get("MediaUrl0")
            print(f"📥 Downloading image: {media_url}")

            try:
                image_response = requests.get(
                    media_url,
                    auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                    timeout=10
                )
                print(f"📦 Image download status: {image_response.status_code}")

                if image_response.status_code == 200:
                    predicted_class, results = predict_image(image_response.content)
                    if predicted_class and results and not isinstance(results, dict) or 'Error' not in results:
                        # Format the response message for WhatsApp
                        message = "*Prediction Result*\n\n"
                        message += f"🌿 *Condition*: {predicted_class}\n\n"
                        message += "*Confidence Levels*:\n"
                        for disease, confidence in results.items():
                            message += f"- {disease}: {confidence:.2f}%\n"
                        message += "\n*What would you like to do next?*\n"
                        message += "1. Type 'prevention' for prevention tips\n"
                        message += "2. Send another image for analysis"
                        
                        # Create a clean response
                        response = MessagingResponse()
                        response.message(message)
                        last_prediction[sender] = {"class": predicted_class, "results": results}
                        print("📤 Sending prediction to WhatsApp...")
                        return str(response)
                    else:
                        error_msg = results.get('Error', 'Unknown error') if isinstance(results, dict) else 'Invalid prediction format'
                        print(f"❌ Prediction error: {error_msg}")
                        resp.message("⚠ Oops! I couldn't analyze that image. Please try with a clearer photo of a potato leaf.")
                else:
                    error_msg = f"Failed to download image. Status: {image_response.status_code}"
                    print(f"❌ {error_msg}")
                    resp.message("⚠ Sorry, I couldn't download that image. Please try again or send a different image.")
                
                # Ensure the response is properly formatted
                response_str = str(resp)
                print("🔧 Final TwiML response:", response_str)
                return response_str

            except Exception as e:
                print("❌ Exception while downloading:", e)
                resp.message(f"⚠ Error downloading image: {e}")
                return str(resp)


        # Step 2: User replies "prevention"
        if incoming_msg == "prevention" and sender in last_prediction:
            disease = last_prediction[sender]["class"]
            try:
                message = f"*{disease} - Prevention Tips*\n\n"
                message += SUGGESTIONS[disease]
                message += "\n\n🔍 *Need more help?* Send another image or type 'hi' for options."
                resp.message(message)
                print("📤 Prevention tips sent to WhatsApp")
                return str(resp)
            except Exception as e:
                print(f"❌ Error sending prevention tips: {e}")
                resp.message("⚠ Sorry, I couldn't fetch prevention tips. Please try again later.")
                return str(resp)

        # Step 3: User replies "confidence"
        if incoming_msg == "confidence" and sender in last_prediction:
            results = last_prediction[sender]["results"]
            msg_text = (
                "*Confidence levels*:\n"
                f"- Early Blight: {results['Early Blight']:.2f}%\n"
                f"- Late Blight: {results['Late Blight']:.2f}%\n"
                f"- Healthy: {results['Healthy']:.2f}%"
            )
            resp.message(msg_text)
            print("📤 Confidence levels sent")
            print("🔧 TwiML response:", str(resp))
            return str(resp)

        # Step 4: Greetings and fallback
        if "hi" in incoming_msg or "hello" in incoming_msg:
            message = ("👋 *Hello!* Send me a clear photo of a potato leaf, and I'll tell you "
                     "if it has *Early Blight*, *Late Blight*, or if it's *Healthy*. 🌿")
            response = MessagingResponse()
            response.message(message)
            print("📤 Sent greeting message")
            return str(response)

        # Fallback for unknown messages
        response = MessagingResponse()
        response.message("🤖 I didn't understand that. Please send a clear photo of a potato leaf or say 'hi'.")
        print("📤 Sent fallback message")
        return str(response)

    except Exception as e:
        print("❌ WhatsApp bot error:", e)
        return "Error", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)