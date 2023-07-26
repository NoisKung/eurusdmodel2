import pandas as pd
import numpy as np
import tensorflow as tf
import MetaTrader5 as mt5
import datetime
import os
import sortCSVdata
import time
import schedule

LAGS = 5
SPLIT_RATIO = 0.8


def calculate_position_size(balance, risk_percentage, entry_price, stop_loss_price):
    risk_amount = balance * risk_percentage
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    position_size = round(position_size, 1)
    return float(position_size)


def determine_trade_direction(predicted_direction):
    return 'Buy' if predicted_direction > 0.5 else 'Sell'


def execute_trade(symbol, trade_direction, entry_price, stop_loss_price, take_profit_price, position_size):
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if trade_direction == 'Buy' else mt5.ORDER_TYPE_SELL_LIMIT
    order_comment = "Buy Limit" if trade_direction == 'Buy' else "Sell Limit"
    tp_price = entry_price + take_profit_price if trade_direction == 'Buy' else entry_price - take_profit_price
    sl_price = entry_price - stop_loss_price if trade_direction == 'Buy' else entry_price + stop_loss_price

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "type": order_type,
        "volume": float(position_size),
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "magic": 123456,
        "comment": order_comment
    }

    print(f"Entry Price: {entry_price:.6f}")
    print(f"Position Size: {position_size}")
    print(f"Take Profit: {tp_price:.6f}")
    print(f"Stop Loss: {sl_price:.6f}")

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"{trade_direction}\nFailed to place {order_comment} order, error code = {result.retcode}")
    else:
        print(f"{trade_direction}\n{order_comment} order placed successfully")


def preprocess_data(df):
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['highs'] = np.log(df['high'] / df['high'].shift(1))
    df['lows'] = np.log(df['low'] / df['low'].shift(1))
    df['direction'] = np.where(df['returns'] > 0, 1, 0)

    for i in range(1, LAGS + 1):
        col_prefixes = ['returns', 'highs', 'lows', 'direction']
        for prefix in col_prefixes:
            df[f'{prefix}_lag{i}'] = df[prefix].shift(i)

    df.dropna(inplace=True)
    return df


def train_model(X_train, y_train, namemodel):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, input_dim=X_train.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=2048, verbose=0)
    
    model.save(namemodel)
    return model


def load_latest_model(namemodel):
    if os.path.exists(namemodel):
        return tf.keras.models.load_model(namemodel)
    else:
        return None


def detect_inverse_head_and_shoulders(df):
    # Implement detection logic for Inverse Head & Shoulders pattern
    # Return True if pattern is detected, False otherwise

    if len(df) < 5:
        return False

    # Check if the current price is higher than the previous and next prices
    for i in range(2, len(df) - 2):
        prev_price = df['close'].iloc[i - 1]
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]

        if current_price > prev_price and current_price > next_price:
            # Check if the left shoulder is lower than the head
            if df['close'].iloc[i - 2] < current_price:
                # Check if the right shoulder is lower than the head
                if df['close'].iloc[i + 2] < current_price:
                    return True

    return False


def detect_head_and_shoulders(df):
    # Implement detection logic for Head & Shoulders pattern
    # Return True if pattern is detected, False otherwise

    if len(df) < 5:
        return False

    # Check if the current price is lower than the previous and next prices
    for i in range(2, len(df) - 2):
        prev_price = df['close'].iloc[i - 1]
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i + 1]

        if current_price < prev_price and current_price < next_price:
            # Check if the left shoulder is higher than the head
            if df['close'].iloc[i - 2] > current_price:
                # Check if the right shoulder is higher than the head
                if df['close'].iloc[i + 2] > current_price:
                    return True

    return False

def main(file_path, file_rates_sorted, timeframe, symbol, balance, risk_percentage, namemodel, train_new_model=True):
    mt5.initialize()
        
    rates = mt5.copy_rates_from(symbol, timeframe, datetime.datetime.now(), 86400)
    df_new = pd.DataFrame(rates)
    df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
    df_new.set_index('time', inplace=True)

    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path, index_col='time')

        if not existing_df.empty and df_new.index[-1] == existing_df.index[-1]:
            print("Data already exists in CSV file")
        else:
            new_df = df_new[df_new.index > existing_df.index[-1]]
            new_df.to_csv(file_path, mode='a', header=False)
            print(f"Added {len(new_df)} new rows to CSV file")
    else:
        df_new.to_csv(file_path)
        print(f"Saved {len(df_new)} rows to CSV file")

    sortCSVdata.sort_file(file_path, file_rates_sorted)
    df = pd.read_csv(file_rates_sorted)
    df = preprocess_data(df)

    input_cols = [col for col in df.columns if any(x in col for x in ['returns_lag', 'highs_log', 'lows_log', 'direction_lag'])]
    target_col = 'direction'
    X = df[input_cols].values
    y = df[target_col].values

    split_index = int(SPLIT_RATIO * len(X))
    X_train, X_valid = X[:split_index], X[split_index:]
    y_train, y_valid = y[:split_index], y[split_index:]

    if train_new_model:
        model = train_model(X_train, y_train, namemodel)
    else:
        model = load_latest_model(namemodel)

    if model is None:
        print("Model not found. Please set train_new_model=True to train a new model.")
        return

    y_pred = model.predict(X_valid)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    accuracy = np.mean(y_pred == y_valid)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    predictions = model.predict(X_valid)
    trade_directions = [determine_trade_direction(p) for p in predictions]

    mt5.symbol_select(symbol)
    last_bar = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0]
    entry_price = last_bar['close']
    #high_price = last_bar['high']
    #low_price = last_bar['low']
    stop_loss = entry_price * 0.000025  # Assuming a 1% stop loss
    take_profit = entry_price * 0.000025  # Assuming a 1% take profit

     # Detect chart patterns
    inverse_head_and_shoulders_count = 0
    head_and_shoulders_count = 0

    while inverse_head_and_shoulders_count < 30 or head_and_shoulders_count < 30:
        if detect_inverse_head_and_shoulders(df) and inverse_head_and_shoulders_count < 30:
            print("Inverse Head & Shoulders pattern detected.")
            inverse_head_and_shoulders_count += 1
            # Place trade according to your strategy

        if detect_head_and_shoulders(df) and head_and_shoulders_count < 30:
            print("Head & Shoulders pattern detected.")
            head_and_shoulders_count += 1
            # Place trade according to your strategy
    
    position_size = calculate_position_size(balance, risk_percentage, entry_price, stop_loss)

    execute_trade(symbol, trade_directions[-1], entry_price, stop_loss, take_profit, position_size)

    mt5.shutdown()


if __name__ == "__main__":
    # Set the file paths and other parameters
    file_path = "forex_xauusd_data.csv"
    file_rates_sorted = "sorted_data_file.csv"
    timeframe = mt5.TIMEFRAME_M1
    symbol = "XAUUSD"
    balance = 100000
    risk_percentage = 0.002
    namemodel = "xauusd_model.h5"
    train_new_model = True  # Set to False to use an existing model
    while True:
        main(file_path, file_rates_sorted, timeframe, symbol, balance, risk_percentage, namemodel, train_new_model)
        time.sleep(60)