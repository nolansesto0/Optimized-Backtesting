# Optimized-Backtesting
A lot more parameters. 83% win rate compared to the 80% win rate before.
--------------------------------
import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file with the correct column name for timestamps
data = pd.read_csv('QQQ_5min_data_April1_April6.csv', parse_dates=['timestamp'])
data.rename(columns={'timestamp': 'timestamp_5min'}, inplace=True)


# Function to calculate MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
   df['EMA_12'] = df['close'].ewm(span=short_window, adjust=False).mean()
   df['EMA_26'] = df['close'].ewm(span=long_window, adjust=False).mean()
   df['MACD'] = df['EMA_12'] - df['EMA_26']
   df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
   df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
   return df


# Function to calculate RSI
def calculate_rsi(df, period=14):
   delta = df['close'].diff(1)
   gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
   loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
   rs = gain / loss
   rsi = 100 - (100 / (1 + rs))
   df['RSI'] = rsi
   return df


# Apply the MACD and RSI calculations on the 5-minute data
data = calculate_macd(data)
data = calculate_rsi(data)


# Interpolate 1-minute data from 5-minute data
def interpolate_1min_data(df):
   rows = []
   for i in range(len(df) - 1):
       current_row = df.iloc[i]
       next_row = df.iloc[i + 1]
       for j in range(5):
           interpolated_row = current_row.copy()
           interpolated_row['timestamp_1min'] = current_row['timestamp_5min'] + pd.Timedelta(minutes=j)
           for column in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
               interpolated_row[column] = current_row[column] + (next_row[column] - current_row[column]) * (j / 5)
           rows.append(interpolated_row)
   df_1min = pd.DataFrame(rows)
   return df_1min


# Interpolate 30-minute data from 5-minute data
def interpolate_30min_data(df):
   rows = []
   for i in range(0, len(df) - 5, 6):
       current_row = df.iloc[i]
       next_row = df.iloc[i + 6]
       interpolated_row = current_row.copy()
       interpolated_row['timestamp_30min'] = current_row['timestamp_5min']
       for column in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
           interpolated_row[column] = current_row[column] + (next_row[column] - current_row[column]) * 6
       rows.append(interpolated_row)
   df_30min = pd.DataFrame(rows)
   return df_30min


# Interpolate 1-minute and 30-minute data
data_1min = interpolate_1min_data(data)
data_30min = interpolate_30min_data(data)


# Apply the MACD and RSI calculations on the 1-minute and 30-minute data
data_1min = calculate_macd(data_1min)
data_1min = calculate_rsi(data_1min)


data_30min = calculate_macd(data_30min)
data_30min = calculate_rsi(data_30min)


# Debugging: Check columns of 1-minute and 30-minute data
print("1-minute data columns:", data_1min.columns)
print("30-minute data columns:", data_30min.columns)


# Merge the 1-minute and 30-minute RSI and MACD back to the original 5-minute data
data = data.merge(data_1min[['timestamp_1min', 'MACD', 'Signal_Line', 'RSI']].rename(columns={
   'MACD': 'MACD_1min', 'Signal_Line': 'Signal_Line_1min', 'RSI': 'RSI_1min'}), left_on='timestamp_5min', right_on='timestamp_1min', suffixes=('', '_1min'))
data = data.merge(data_30min[['timestamp_30min', 'MACD', 'Signal_Line', 'RSI']].rename(columns={
   'MACD': 'MACD_30min', 'Signal_Line': 'Signal_Line_30min', 'RSI': 'RSI_30min'}), left_on='timestamp_5min', right_on='timestamp_30min', suffixes=('', '_30min'))


# Drop unnecessary columns
data.drop(columns=['timestamp_1min', 'timestamp_30min'], inplace=True)


# Debugging: Check final merged data columns
print("Final data columns:", data.columns)


# Function to calculate premarket move
def calculate_premarket_move(df):
   premarket_moves = []


   # Identify unique trading dates
   df['date'] = df['timestamp_5min'].dt.date
   unique_dates = df['date'].unique()


   for i in range(len(unique_dates) - 1):
       # Get the last row of the current day
       current_day = df[df['date'] == unique_dates[i]]
       last_row_current_day = current_day.iloc[-1]


       # Get the first row of the next day
       next_day = df[df['date'] == unique_dates[i + 1]]
       first_row_next_day = next_day.iloc[0]


       # Calculate the premarket move
       premarket_move = next_day['open'].iloc[0] - current_day['close'].iloc[-1]
       premarket_moves.append((unique_dates[i + 1], premarket_move))


   return pd.DataFrame(premarket_moves, columns=['date', 'premarket_move'])


# Calculate premarket moves
premarket_moves_df = calculate_premarket_move(data)


# Merge the premarket move data into the 5-minute data
data['date'] = data['timestamp_5min'].dt.date
data = data.merge(premarket_moves_df, on='date', how='left')


# Drop the helper column
data.drop(columns=['date'], inplace=True)


# Function to check for three consecutive higher highs and higher lows
def check_consecutive_days(daily_data):
   if len(daily_data) < 3:
       return False
   return (daily_data.iloc[-3]['high'] < daily_data.iloc[-2]['high'] < daily_data.iloc[-1]['high'] and
           daily_data.iloc[-3]['low'] < daily_data.iloc[-2]['low'] < daily_data.iloc[-1]['low'])


# Function to process each day's data
def process_day(day_data, account_balance, wait_for_lower_open, reference_open):
   trade_entries = []
   result = None
   entry_range = 0.05  # Define the range within which MACD and Signal Line are considered close


   # Check if the first 20 minutes go down by more than $2
   first_20_min = day_data.head(4)  # 4 rows of 5-minute intervals = 20 minutes
   if (first_20_min['open'].iloc[0] - first_20_min['close'].min()) > 2:
       return account_balance, trade_entries, 'No Trade - First 20 minutes down > $2'


   for i in range(1, len(day_data)):
       current_row = day_data.iloc[i]


       # Check if premarket move is within the acceptable range
       if current_row['premarket_move'] < -4 or current_row['premarket_move'] > 4:
           continue


       # Skip trading if RSI is too high or if waiting for a lower open
       if current_row['RSI'] > 50 or wait_for_lower_open:
           continue


       # Check for MACD and Signal Line being within a certain range below the zero line
       macd_condition_5min = current_row['MACD'] < 0 and abs(current_row['MACD'] - current_row['Signal_Line']) <= entry_range
       rsi_condition_30min = current_row['RSI_30min'] < 50
       macd_condition_30min = current_row['MACD_30min'] < 0
       macd_condition_1min = current_row['MACD_1min'] < 0 and abs(current_row['MACD_1min'] - current_row['Signal_Line_1min']) <= entry_range
       rsi_condition_1min = current_row['RSI_1min'] < 50


       # Debugging: Print current row to check column presence
       print("Current row during processing:", current_row)


       if macd_condition_5min and rsi_condition_30min and macd_condition_30min and macd_condition_1min and rsi_condition_1min:
           entry_price = current_row['close']
           entry_time = current_row.name


           # Track prices after entry
           after_entry_data = day_data[day_data.index > entry_time]


           # Determine win or loss
           min_price = after_entry_data['low'].min()
           if min_price <= entry_price - 3.5:
               account_balance *= 0.87  # Loss day
               result = 'Loss'
           else:
               account_balance *= 1.0412  # Win day
               result = 'Win'


           trade_entries.append(entry_time)
           break


   return account_balance, trade_entries, result


# Initial account balance
initial_balance = 5000
account_balance = initial_balance
equity_curve = []
dates = []
trade_entries_log = []
results_log = []


# Track daily highs and lows
daily_data = data.resample('D', on='timestamp_5min').agg({
   'open': 'first',
   'high': 'max',
   'low': 'min',
   'close': 'last',
   'volume': 'sum'
}).dropna()


# Process each day
wait_for_lower_open = False
reference_open = None
consecutive_highs_lows_counter = 0


for date in pd.date_range(start='2023-06-01', end='2024-05-24'):
   day_data = data[data['timestamp_5min'].dt.date == date.date()]
   if not day_data.empty:
       if wait_for_lower_open:
           if day_data.iloc[0]['open'] < reference_open - 5:
               wait_for_lower_open = False  # Condition met, resume trading
       else:
           if check_consecutive_days(daily_data):
               consecutive_highs_lows_counter += 1
               if consecutive_highs_lows_counter >= 3:
                   wait_for_lower_open = True
                   reference_open = day_data.iloc[0]['open']
                   consecutive_highs_lows_counter = 0
           else:
               consecutive_highs_lows_counter = 0


           account_balance, trade_entries, result = process_day(day_data, account_balance, wait_for_lower_open, reference_open)
           equity_curve.append((account_balance / initial_balance - 1) * 100)  # Calculate percent change
           dates.append(date)
           trade_entries_log.extend(trade_entries)
           results_log.append((date, result if result else 'None'))


# Create a DataFrame for the equity curve
equity_df = pd.DataFrame({'Date': dates, 'Percent Increase': equity_curve})


# Calculate maximum drawdown
equity_df['Drawdown'] = equity_df['Percent Increase'].cummax() - equity_df['Percent Increase']
max_drawdown = equity_df['Drawdown'].max()


# Function to calculate maximum advantage/gain
def calculate_max_gain(df):
   df['Max_Gain'] = df['Percent Increase'] - df['Percent Increase'].cummin()
   return df


# Calculate maximum advantage/gain
equity_df = calculate_max_gain(equity_df)
max_gain = equity_df['Max_Gain'].max()


# Plot the equity curve
plt.figure(figsize=(10, 6))
plt.plot(equity_df['Date'], equity_df['Percent Increase'], marker='o')
plt.title('Equity Curve (Percent Increase)')
plt.xlabel('Date')
plt.ylabel('Percent Increase')
plt.grid(True)
plt.show()


# Save the equity curve to a CSV file
equity_df.to_csv('Equity_Curve_April1_April6.csv', index=False)


# Print maximum drawdown and maximum gain
print(f'Maximum Drawdown: {max_drawdown:.2f}%')
print(f'Maximum Gain: {max_gain:.2f}%')


# Calculate and print the difference and ratio
difference = max_gain - max_drawdown
ratio = max_gain / max_drawdown if max_drawdown != 0 else float('inf')
print(f'Difference between Max Gain and Max Drawdown: {difference:.2f}%')
print(f'Ratio of Max Gain to Max Drawdown: {ratio:.2f}')


# Print trade entries and results
for date, result in results_log:
   print(f'{date.date()}: {result}')


for entry in trade_entries_log:
   print(f'Trade entry at: {entry}')


print(equity_df)

