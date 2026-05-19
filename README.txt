# 🚀 Vigil.AI — Intelligent Network Traffic Analyzer  
Machine Learning + ChatGPT Integration

Vigil.AI is an intelligent traffic analysis assistant that combines:
- A trained XGBoost machine learning model (32-feature CIC-IDS dataset)
- A custom Python prediction engine
- A Node.js backend for API routing
- A frontend chatbot UI powered by ChatGPT + ML
- Real-time traffic classification via `/predict`

---

# 📁 Project Structure

```
chatbot_project/
│
├── server.js                 # Node.js backend (ChatGPT + ML routes)
├── model.py                  # Python model loader + prediction script
├── xgboost_model.pkl         # Trained ML model (from Google Colab)
│
├── chatbot-index.html        # Frontend UI
├── chatbot-style.css         # Frontend styling
├── script.js                 # Frontend logic + ML routing
│
├── .env                      # Stores OpenAI API key
└── README.md                 # Documentation
```

---

# 🧠 Requirements

## 💻 Software
- Node.js (v18+)
- Python 3.11+
- VS Code (recommended)
- Browser (Chrome/Edge)

## 📦 Python Packages
Install these on your system:

```
pip install numpy joblib xgboost==3.1.1
```

> ⚠ IMPORTANT:  
> Your model was trained in Google Colab using **XGBoost 3.1.1**,  
> so your local machine MUST use the *same version*.

---

# 🔑 Environment Setup

Create a `.env` file inside your project folder:

```
OPENAI_API_KEY=your_api_key_here
```

Make sure there are **no quotes** around the key.

---

# 🐍 Running the Python Model Manually (Optional Test)

Test that your model loads correctly:

```
python model.py
```

If it prints nothing → GOOD  
If you get errors, fix Python dependencies

---

# ⚙️ Running the Backend (Node.js)

1. Install dependencies:

```
npm install express node-fetch cors dotenv
```

2. Start the server:

```
node server.js
```

You should see:

```
Loaded API Key: ✓ Found
Server running on http://localhost:3000
```

---

# 💬 Testing the ChatGPT Route

POST:

```
http://localhost:3000/api/chat
```

Body:

```json
{
  "userMessage": "Hello"
}
```

Should return a normal ChatGPT response.

---

# 🧠 Testing the ML Prediction Route

POST:

```
http://localhost:3000/predict
```

Body:

```json
{
  "features": [1,2,3,4,5,6]
}
```

Your backend + Python model pads missing features to 32 automatically.

Expected response:

```json
{
  "prediction": 0
}
```

---

# 🌐 Running the Frontend

Open `chatbot-index.html` in your browser.

OR install the VS Code extension:

```
Live Server
```

Then right-click > **Open with Live Server**

This loads the chatbot UI.

---

# 🧪 Using the Chatbot

Type:

```
analyze 1,2,3,4,5,6
```

The system will:

1. Detect "analyze"
2. Extract the numbers
3. Send them to: `POST /predict`
4. Receive ML prediction
5. Convert it to a readable label:
   - Benign  
   - DDoS Attack  
   - DoS Attack  
   - Port Scan  
   - Botnet  
   - Infiltration  
   - Web Attack

Example output:

```
🧠 Traffic Classification: Benign (Normal Traffic)
```

---

# 🧯 Troubleshooting

### ❌ Issue: “Unknown Class (undefined)”
Cause: Python crashed → returned no JSON  
Fix: Check VS Code terminal for Python errors.

### ❌ Issue: XGBoost version mismatch
Fix:

```
pip install xgboost==3.1.1
```

### ❌ Issue: Python not running
Try alternate commands:

```
py model.py
python3 model.py
```

### ❌ Issue: Node server prints no Python errors
Your `/predict` route isn’t reached → frontend URL wrong  
Fix frontend code or check CORS.

---

# 🧩 Next Steps
You now can choose between:

### Option A — Build a UI to input all 32 features  
(Full model accuracy, no retraining)

### Option B — Retrain your model using only 6 features  
(Simpler UX, lower accuracy)

---

# 🎓 Credits
Developed for a capstone project integrating:
- Machine Learning (XGBoost)
- Python Backend
- Node.js API Gateway
- OpenAI ChatGPT
- Modern Web Frontend

