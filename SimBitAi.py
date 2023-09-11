import yfinance as yf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from art import text2art
import datetime

today = datetime.date.today()

# Generate ASCII art as logo
ascii_art = text2art("SimBitAi")

print("============================================================")

# Print the generated ASCII art
print(ascii_art)
print("Simple Bitcoin price prediction artificial intelligence")
print("============================================================")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Download historical data
print("Downloading historical Bitcoin data for training...")
bitcoin = yf.download('BTC-USD', start='2010-07-17', end=today)
print("Downloaded.")
print("Training...")

# Prepare data for model
bitcoin['Prediction'] = bitcoin['Close'].shift(-1)
bitcoin.dropna(inplace=True)
X = np.array(bitcoin.drop(['Prediction'], axis=1))
Y = np.array(bitcoin['Prediction'])

# Split data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
bitcoin['Prediction'] = model.predict(np.array(bitcoin.drop(['Prediction'], axis=1)))
print("Training complete.")

# Print the predicted price for tomorrow, 7 days, 30 days and 1 year from now
print("Predicted price for tomorrow: ", bitcoin['Prediction'].iloc[-1])
print("Predicted price for 7 days: ", bitcoin['Prediction'].iloc[-7])
print("Predicted price for 30 days: ", bitcoin['Prediction'].iloc[-30])
print("Predicted price for 1 year: ", bitcoin['Prediction'].iloc[-365])
print("============================================================")
print("If you love this program, buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Prevent the window from closing immediately
input('Press ENTER to exit')
