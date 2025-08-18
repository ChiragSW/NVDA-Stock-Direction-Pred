# NVDA Stock Predictor 📈

This project uses an **LSTM-based Machine Learning model** to predict whether **NVIDIA's (NVDA) stock price** will go **up or down tomorrow**.  
The prediction leverages historical data from **NVDA** as well as correlated stocks (**AAPL, GOOGL, META**).  

After every run, the model also **sends an email** with the prediction and confidence score.  
The next step will be to **automate daily runs** so predictions are delivered automatically.

---

## 🚀 Features
- Uses **LSTM** for time-series prediction  
- Considers **correlation with AAPL, GOOGL, META** along with NVDA history  
- Outputs **prediction direction (Up/Down)** and a **confidence score**  
- Sends **email notifications** after each run  
- Future work: **Automation for daily scheduled runs**  

---

## 🛠️ Tech Stack
- **Python**  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **TensorFlow / Keras**  


---

## ⚡ How It Works
1. Collects historical stock prices for NVDA, AAPL, GOOGL, META  
2. Prepares time-series input for LSTM  
3. Trains the model to classify **Up or Down** for NVDA tomorrow  
4. Runs inference and outputs prediction with confidence  
5. Sends results via **email notification**  

---

## 📧 Email Notifications
- After each prediction run, you’ll receive a **mail** with:  
  - Predicted direction (Up/Down)  
  - Confidence score (%)  

---

## ▶️ Usage
1. Clone this repo:
   ```bash
   git clone https://github.com/NVDA-Stock-Direction-Pred.git
   cd nvda-pred
2. Start a Virtual Environment.
3. Install requirements
   ```bash
   pip install requirements.txt
4. Make a .env file and write this:
   ```bash
   EMAIL_USER=user@gmail.com
   EMAIL_PASS=your_app_generated_password
   EMAIL_TO=receiver@gmail.com
5. Run the py file
   ```bash
   py lstm-prediction.py



