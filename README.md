
Potato Leaf Disease Detection using Deep Learning & WhatsApp Bot


---

1. Project Title

Potato Leaf Disease Detection using Deep Learning with WhatsApp Chatbot


---

2. Project Description

This project presents an intelligent agricultural support system that detects potato leaf diseases using Deep Learning (CNN) and delivers results through a WhatsApp chatbot. Farmers or users can simply send an image of a potato leaf via WhatsApp, and the system classifies it as Early Blight, Late Blight, or Healthy, along with confidence scores and preventive suggestions.

The system eliminates the need for expert diagnosis, mobile applications, or special equipment. It leverages TensorFlow, Flask, and Twilio WhatsApp API, and can be deployed online using platforms such as Railway for real-time access.


---

3. Features

🌿 Automatic potato leaf disease detection

📷 Image-based prediction using CNN

💬 WhatsApp chatbot interface

📊 Confidence score for predictions

🧠 Preventive tips

☁️ Cloud deployable backend





---

4. Technology Stack

Programming Language:

Python 3.10+


Libraries & Frameworks:

TensorFlow / Keras

NumPy 

Flask

Pillow (PIL)

Requests

Python-dotenv


APIs & Tools:

Twilio WhatsApp API

Railway (Deployment)

GitHub (Version Control)



---

5. Dataset

The model is trained using a potato leaf dataset containing three classes:

Early Blight

Late Blight

Healthy


Dataset structure:

train/
 ├── Early_Blight/
 ├── Late_Blight/
 └── Healthy/

val/
 ├── Early_Blight/
 ├── Late_Blight/
 └── Healthy/

Images are resized, normalized, and augmented before training.


---

6. Model Architecture

The CNN model consists of:

Convolutional Layers (feature extraction)

Max Pooling Layers

Flatten Layer

Dense Fully Connected Layers

Softmax Output Layer


The trained model is saved as:

potato_disease_model.keras


---

7. Project Folder Structure

major_project/
│
├── train/
├── val/
├── static/
│   └── temp.jpg
│
├── whatsapp_bot.py
├── train_potato.py
├── requirements.txt
├── .env
├── potato_disease_model.keras
├── README.txt


---

8. Environment Setup (Local)

Step 1: Create Virtual Environment

python -m venv venv

Step 2: Activate Virtual Environment

Windows:

venv\Scripts\activate

Step 3: Install Dependencies

pip install -r requirements.txt


---

9. Environment Variables (.env file)

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
MY_WHATSAPP_NUMBER=whatsapp:+91XXXXXXXXXX


---

10. Train the Model (Optional)

If you want to retrain the model:

python train_potato.py

This will generate a new potato_disease_model.keras.


---

11. Run the WhatsApp Bot (Local)

python whatsapp_bot.py

The Flask server will start at:

http://127.0.0.1:5000

Use ngrok to expose the local server and configure the Twilio webhook.


---

12. WhatsApp Bot Workflow

1. User sends leaf image via WhatsApp


2. Twilio forwards request to Flask webhook


3. Image is downloaded and preprocessed


4. CNN model predicts disease


5. Confidence score is calculated


6. Prevention tips are added


7. Response is sent back to WhatsApp




---

13. Deployment on Railway

Steps:

1. Push project to GitHub


2. Login to https://railway.app


3. Create new project → Deploy from GitHub


4. Add environment variables in Railway dashboard


5. Railway auto-builds and deploys the Flask app


6. Use deployed URL as Twilio webhook



Note: Railway free plan may sleep after inactivity.


---

14. Output Example

🧠 Disease Detected: Early Blight
📊 Confidence: 94.3%
💡 Advice: Remove infected leaves and improve air circulation.


---

15. Applications

Smart agriculture

Disease monitoring for farmers

Educational demonstrations

Research and academic projects



---

16. Advantages

No mobile app required

Easy WhatsApp-based interface

Cost-effective solution

Real-time prediction

Scalable and deployable



---

17. Limitations

Depends on image quality

Internet required

Limited to potato leaves



---

18. Conclusion

This project demonstrates an effective integration of deep learning and messaging platforms to solve real-world agricultural problems. By enabling disease detection through WhatsApp, it provides an accessible, user-friendly, and scalable solution that can assist farmers in early diagnosis and prevention, improving crop productivity and sustainability.