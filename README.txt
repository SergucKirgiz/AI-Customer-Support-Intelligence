# 🤖 AI Customer Support Intelligence

A deep learning-powered customer support chatbot built with **TensorFlow** and **Streamlit**. This assistant can classify 27 different customer intents and provide appropriate automated responses.

## 🚀 Features
- **Intent Classification:** Uses a Neural Network to identify user needs.
- **Natural Language Processing:** Powered by TF-IDF Vectorization.
- **Real-time Chat UI:** Interactive interface built with Streamlit.
- **Automated Responses:** Smart mapping from intent to professional support messages.

## 📂 Project Structure
- `app.py`: The Streamlit web application.
- `customer.py`: Training script for the Neural Network.
- `models/`: Saved model, TF-IDF vectorizer, and label encoder.
- `data/`: Dataset files for training and validation.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/AI-Customer-Support-Intelligence.git](https://github.com/yourusername/AI-Customer-Support-Intelligence.git)
   cd AI-Customer-Support-Intelligence

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3.Run the application:

Bash
streamlit run app.py

🧠 Model Details
The model is a Sequential Neural Network with:

Input Layer: TF-IDF Features (Max 2000)

Hidden Layers: 512 & 256 neurons with ReLU activation.

Output Layer: 27 neurons with Softmax activation for multi-class classification.