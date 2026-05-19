# Vigil AI Demo Script

This script is designed for an 8-12 minute group prototype video. Each member should explain a clear section of the system and show the code briefly while also demonstrating the running app.

## Suggested Timing

| Time | Section | Speaker |
| --- | --- | --- |
| 0:00-1:00 | Project overview and problem statement | Member 1 |
| 1:00-3:00 | Backend, authentication, and API flow | Member 1 |
| 3:00-5:00 | Frontend chatbot, dashboards, and live traffic UI | Member 2 |
| 5:00-7:30 | Dataset, preprocessing, feature handling, and training | Member 3 |
| 7:30-9:30 | Model results, predictions, and comparison | Member 4 |
| 9:30-11:00 | Full system demo and wrap-up | Whole group |

## Member 1: Backend and Authentication

Files to show:

- `server.js`
- `routes/auth.js`
- `db/database.js`

Main points:

- `server.js` is the entry point for the prototype.
- Express serves the static frontend and provides API routes.
- Sessions protect the app pages so users must log in first.
- `routes/auth.js` handles register, login, and logout.
- Passwords are hashed with bcrypt before being stored.
- SQLite is used because it is lightweight and easy to package for a prototype.

Demo actions:

- Start the server with `node server.js --localhost`.
- Open `http://localhost:3000`.
- Register or log in.
- Show that protected pages redirect properly.

## Member 2: Frontend, Chatbot, and Traffic UI

Files to show:

- `public/script.js`
- `public/traffic-analysis.js`
- `public/traffic-analytics.js`
- `public/chatbot-index.html`
- `public/traffic-analysis.html`

Main points:

- The chatbot page lets a user ask security-related questions.
- Messages are sent to `/api/chat`, which forwards them to Gemini.
- The traffic page can manually submit 32 model features.
- The live monitor uses Server-Sent Events from `/api/live`.
- The alert system looks for repeated high-risk events instead of reacting to one packet.

Demo actions:

- Ask the chatbot a basic question.
- Open the traffic analysis page.
- Submit sample feature values for a model prediction.
- If tshark is available, start a live capture and show the table updating.

## Member 3: Dataset and Preprocessing

Files to show:

- `train_vigil_ai_model.py`
- `inspect_labels.py`
- `fix_cm.py`

Main points:

- The dataset comes from CICDDoS2019 parquet files.
- The training script loads all parquet files into one dataframe.
- Different raw attack names are grouped into seven app categories.
- Only numeric network-flow features are used.
- The project uses 32 selected features to match the web input form.
- Missing selected columns are filled with `0`.
- Feature order is kept consistent because sklearn models depend on column position.
- SMOTE balances the dataset so minority attack categories are represented better.

Demo actions:

- Show the `selected_features` list in `train_vigil_ai_model.py`.
- Show the label normalization dictionary.
- Show the SMOTE and train/test split section.
- Show `fix_cm.py` as the script used to regenerate `confusion_matrix.png`.

## Member 4: Model Results and Prediction Pipeline

Files to show:

- `model.py`
- `train_vigil_ai_model.py`
- `model_stats.json`
- `public/model_dashboard.html`

Main points:

- XGBoost is the production model because it performed better and trains faster on tabular flow features.
- MLP is included as a deep learning comparison model.
- `model.py` loads the saved pipeline and handles prediction requests from Node.
- The prediction path cleans invalid values and handles feature-count mismatch.
- `model_stats.json` stores results for the model dashboard.

Latest saved results:

- XGBoost accuracy: 93.44%
- XGBoost F1: 93.41%
- MLP accuracy: 75.23%
- MLP F1: 74.57%

Demo actions:

- Open `/models` and show the dashboard.
- Explain why XGBoost is used by `/predict`.
- Run or describe a sample prediction request.

## Full Demo Flow

1. Start the server:

   ```bash
   node server.js --localhost
   ```

2. Open the app:

   ```text
   http://localhost:3000
   ```

3. Log in or register.

4. Show the chatbot page and ask a question.

5. Show the traffic analysis page.

6. Submit manual features to the ML model.

7. Open `/models` and explain the saved results.

8. Briefly show the code sections for each member.

## Short Closing Statement

Vigil AI demonstrates a working prototype that combines web development, authentication, live network monitoring, machine learning classification, dataset preprocessing, and AI-assisted security explanations. The project is packaged as a capstone prototype with commented source code, dataset documentation, and a repeatable demo flow.
