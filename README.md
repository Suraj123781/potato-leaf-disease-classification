
## 🥔 Potato Leaf Disease Classification with WhatsApp Bot

This project uses deep learning to classify potato leaf diseases from images and provides instant feedback via WhatsApp. It empowers farmers and agricultural workers with an accessible, mobile-friendly tool for early disease detection.

---

### 🚀 Features

- 📷 Image-based disease classification using a trained CNN model
- 🤖 WhatsApp bot interface powered by Twilio
- 🧠 Supports multi-image predictions
- ☁️ Cloud-ready with `Procfile` and `requirements.txt`
- 🔒 Secure `.env` integration for API keys and credentials

---

### 🧠 Model Overview

- Framework: TensorFlow / Keras
- Architecture: Custom CNN
- Classes: Healthy, Early Blight, Late Blight
- Input size: 256x256 RGB images
- Output: Predicted class label + confidence score

---

### 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Suraj123781/potato-leaf-disease-classification.git
cd potato-leaf-disease-classification

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 🤖 Running the WhatsApp Bot

1. Set up your `.env` file with:
   ```
   TWILIO_ACCOUNT_SID=your_sid
   TWILIO_AUTH_TOKEN=your_token
   TWILIO_PHONE_NUMBER=your_twilio_number
   USER_PHONE_NUMBER=your_verified_number
   ```

2. Start the bot:
   ```bash
   python whatsapp_bot.py
   ```

3. Send leaf images to your Twilio WhatsApp number and receive predictions instantly.

---

### 🧪 Training the Model (Optional)

```bash
python train_potato.py
```

- Training and validation data should be placed in `train/` and `val/` folders respectively.
- Model will be saved as `potato_disease_model.keras`

---

### 🌍 Deployment (Render or Railway)

- Add `Procfile`:
  ```
  web: python whatsapp_bot.py
  ```

- Ensure `requirements.txt` is complete
- Set environment variables in your cloud dashboard

---

### 📸 Sample Prediction

| Input Image | Prediction |
|-------------|------------|
| ![leaf](sample.jpg) | Late Blight (92.3%) |


