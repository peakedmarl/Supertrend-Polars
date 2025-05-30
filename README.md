# Polars Supertrend Indicator

This repository contains a Python script (`polars_supertrend.py`) that implements the Supertrend technical indicator using the Polars library.

## `polars_supertrend.py`

The script provides a function `supertrend` to calculate the Supertrend line and direction.

### Function: `supertrend(df, period, multiplier)`

Calculates the Supertrend indicator based on input financial data.

**Parameters:**

*   `df` (polars.DataFrame): Input DataFrame containing financial data. Must include the following columns with `Float64` dtype:
    *   `'High'`: High prices for each period.
    *   `'Low'`: Low prices for each period.
    *   `'Close'`: Closing prices for each period.
*   `period` (int): The lookback period for calculating the Average True Range (ATR). A common value is 10 or 7.
*   `multiplier` (float): The multiplier for the ATR to determine the offset of the bands from the median price. A common value is 3.0.

**Returns:**

*   `polars.DataFrame`: A new DataFrame with the original data plus the following calculated columns:
    *   `'TR'` (Float64): True Range.
    *   `'ATR'` (Float64): Average True Range (calculated using Simple Moving Average - SMA).
    *   `'BUB'` (Float64): Basic Upper Band.
    *   `'BLB'` (Float64): Basic Lower Band.
    *   `'FUB'` (Float64): Final Upper Band.
    *   `'FLB'` (Float64): Final Lower Band.
    *   `'Supertrend'` (Float64): The Supertrend indicator line.
    *   `'Supertrend_Direction'` (String): Indicates the trend direction ('Up' or 'Down').

**Example Usage:**

```python
import polars as pl
from polars_supertrend import supertrend # Assuming the script is in your PYTHONPATH

# Sample data (replace with your actual data)
data = {
    'High': [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 12.0, 11.8, 11.5, 11.0],
    'Low':  [9.5, 9.8, 10.5, 10.2, 10.8, 11.0, 11.5, 11.2, 11.0, 10.5],
    'Close':[9.8, 10.2, 10.8, 10.5, 11.0, 11.2, 11.8, 11.5, 11.2, 10.8]
}
# Ensure schema includes Float64 for HLC columns
schema = {'High': pl.Float64, 'Low': pl.Float64, 'Close': pl.Float64}
ohlc_df = pl.DataFrame(data, schema=schema)

# Calculate Supertrend
# Using period=7 and multiplier=3 as common defaults
supertrend_df = supertrend(ohlc_df, period=7, multiplier=3.0)

print(supertrend_df)
```

**Notes:**

*   The ATR is calculated using a Simple Moving Average (SMA).
*   The implementation involves iterative calculations for Final Bands and the Supertrend line, which are performed using loops internally for clarity and correctness within Polars' DataFrame structure.
