# Vigil AI - Final Prototype Submission

Vigil AI is a CST499 capstone prototype for network traffic analysis. The project combines a Node.js web server, a browser-based dashboard/chatbot interface, a Python machine learning prediction pipeline, and a CICDDoS2019-based training workflow.

The prototype demonstrates:

- User registration, login, and protected app pages
- Live traffic monitoring through `tshark` and Server-Sent Events
- Manual 32-feature traffic classification through a trained Python model
- Gemini-powered chatbot support for user questions and alert explanations
- Model comparison results for XGBoost and MLP
- A model dashboard that reads saved training statistics

## Project Structure

```text
chatbot_project/
├── server.js                     # Main Node.js server and API routes
├── routes/
│   └── auth.js                   # Register, login, and logout routes
├── db/
│   └── database.js               # SQLite setup for local users
├── public/
│   ├── chatbot-index.html        # Main chatbot page
│   ├── script.js                 # Chatbot and model request logic
│   ├── traffic-analysis.html     # Live traffic and manual ML page
│   ├── traffic-analysis.js       # Live capture, alert, and feature form logic
│   ├── traffic-analytics.html    # Chart-based traffic analytics page
│   ├── traffic-analytics.js      # Live chart updates
│   └── model_dashboard.html      # Model metrics dashboard
├── model.py                      # Loads the trained model and predicts one request
├── train_vigil_ai_model.py       # Builds XGBoost and MLP models from the dataset
├── fix_cm.py                     # Regenerates the confusion matrix image
├── inspect_labels.py             # Checks dataset files for label columns
├── extract_model_stats.py        # Exports scaler/model metadata
├── cicddos2019/                  # Dataset files used for training/evaluation
├── vigil_pipeline.pkl            # Saved XGBoost production pipeline
├── vigil_mlp_pipeline.pkl        # Saved MLP comparison pipeline
├── model_stats.json              # Saved metrics used by the dashboard
├── confusion_matrix.png          # Confusion matrix artifact
├── DEMO_SCRIPT.md                # Suggested group demo plan
└── package.json                  # Node dependencies and start script
```

## Requirements

- Node.js 18 or newer
- Python 3.11 or newer
- Wireshark/tshark for live packet capture
- Browser such as Chrome or Edge

Python packages used by the ML scripts:

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost joblib matplotlib pyarrow
```

Node packages are listed in `package.json`.

## Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SESSION_SECRET=replace_with_a_local_secret
TSHARK_PATH=C:\Program Files\Wireshark\tshark.exe
```

`TSHARK_PATH` is optional if `tshark` is already available from the system path.

## Running the Prototype

Install Node dependencies:

```bash
npm install
```

Start the server:

```bash
node server.js --localhost
```

The app runs at:

```text
http://localhost:3000
```

The default route redirects to the login page. After registration/login, use the app pages for chatbot support, manual prediction, live traffic analysis, and model results.

## Dataset

The prototype uses CICDDoS2019 parquet files stored in `cicddos2019/`.

Dataset summary from the latest training run:

- Total raw samples loaded: `431,371`
- Features selected for the web/model pipeline: `32`
- Target classes used by the app: `7`
- Training samples after SMOTE balancing: `849,254`
- Testing samples after SMOTE balancing: `212,314`

The selected features are numeric network-flow measurements such as packet counts, packet lengths, byte/packet rates, inter-arrival timing, header lengths, and TCP flag counts.

The label categories used by Vigil AI are:

- Benign / normal traffic
- DDoS attack
- DoS attack
- Port scan activity
- Botnet / malware behavior
- Infiltration attempt
- Web attack

## Preprocessing

The raw CICDDoS files contain several specific attack names, so `train_vigil_ai_model.py` groups them into the seven broader categories used by the app. For example, LDAP, MSSQL, NetBIOS, NTP, TFTP, and similar distributed attack labels are grouped under DDoS.

Only numeric columns are used for model input. The script then reindexes the dataframe to the exact 32 selected features. This keeps feature order consistent between training, the web form, and prediction. If a selected column is missing from a dataset file, it is filled with `0` so the pipeline can still run.

SMOTE is applied before the train/test split to reduce class imbalance. This gives smaller attack categories more representation during model training and evaluation.

Standard scaling is included inside each saved sklearn pipeline. During prediction, `model.py` checks the scaler to determine the expected feature count, replaces invalid values like NaN or infinity with `0`, and pads or trims incoming feature arrays when needed. This makes the demo more forgiving if the frontend sends fewer or extra values.

## Model Decisions

The production model is XGBoost, saved as `vigil_pipeline.pkl`. XGBoost was chosen because it performs well on tabular flow-based network data and trains quickly enough for prototype iteration.

The comparison model is an MLP neural network, saved as `vigil_mlp_pipeline.pkl`. It is included for the capstone comparison but is not the default `/predict` model.

Latest saved metrics:

| Model | Accuracy | Precision | Recall | F1 | Train Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 93.44% | 94.01% | 93.44% | 93.41% | 21.4s |
| MLP | 75.23% | 84.77% | 75.23% | 74.57% | 608.2s |

## Important Routes

| Route | Purpose |
| --- | --- |
| `/` | Redirects to login |
| `/auth/register` | Creates a local user |
| `/auth/login` | Starts a session |
| `/auth/logout` | Ends a session |
| `/app` | Protected app entry point |
| `/predict` | Sends features to `model.py` for ML prediction |
| `/api/chat` | Sends chatbot questions to Gemini |
| `/api/classify` | Uses Gemini for traffic classification comparison |
| `/api/interfaces` | Lists tshark interfaces |
| `/api/start` | Starts live capture |
| `/api/stop` | Stops live capture |
| `/api/live` | Streams live traffic events through SSE |
| `/models` | Opens the model dashboard |
| `/model_stats.json` | Serves saved model metrics |

## Testing and Verification

Syntax checks used before packaging:

```bash
node --check server.js
node --check public/script.js
node --check public/traffic-analysis.js
node --check public/traffic-analytics.js
python -m py_compile model.py train_vigil_ai_model.py fix_cm.py inspect_labels.py extract_model_stats.py
```

Manual prototype checks:

- Register or log in through the browser
- Open the chatbot page and ask a basic question
- Submit 32 manual feature values to `/predict`
- Open `/models` and confirm metrics load
- If Wireshark is installed, start live traffic capture and confirm events stream into the table

## Submission Notes

The final zip should include source code, public frontend files, ML scripts, saved model artifacts, dataset files or an accessible dataset link, `model_stats.json`, `confusion_matrix.png`, and this documentation.

Do not include private `.env` values in public submissions. Use `.env.example` for expected environment variable names.
