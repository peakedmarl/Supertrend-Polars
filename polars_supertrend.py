import polars as pl
from polars import col

def supertrend(df: pl.DataFrame, period: int, multiplier: float) -> pl.DataFrame:
    """
    Calculates the Supertrend indicator.
    This part calculates the True Range (TR).
    """
    hl = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift(1)).abs()
    lpc = (df['Low'] - df['Close'].shift(1)).abs()

    tr = pl.max_horizontal([hl, hpc, lpc])
    df = df.with_columns(tr.alias('TR'))

    # Calculate ATR
    atr = df['TR'].rolling_mean(window_size=period, min_samples=1)
    df = df.with_columns(atr.alias('ATR'))

    # Calculate Basic Upper Band (BUB)
    basic_upper_band = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
    df = df.with_columns(basic_upper_band.alias('BUB'))

    # Calculate Basic Lower Band (BLB)
    basic_lower_band = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']
    df = df.with_columns(basic_lower_band.alias('BLB'))

    # Calculate Final Upper Band (FUB) and Final Lower Band (FLB)
    n = df.height
    fub_list = []
    flb_list = []

    # Get Series for faster access
    bub_series = df['BUB']
    blb_series = df['BLB']
    close_series = df['Close']
    prev_close_series = close_series.shift(1)

    # First row
    if n > 0:
        fub_list.append(bub_series[0])
        flb_list.append(blb_series[0])

        # Iterate for subsequent rows
        for i in range(1, n):
            current_bub = bub_series[i]
            current_blb = blb_series[i]
            previous_fub = fub_list[i-1]
            previous_flb = flb_list[i-1]
            previous_close_val = prev_close_series[i]

            # FUB Logic
            if current_bub < previous_fub or (previous_close_val is not None and previous_close_val > previous_fub):
                fub_list.append(current_bub)
            else:
                fub_list.append(previous_fub)

            # FLB Logic
            if current_blb > previous_flb or (previous_close_val is not None and previous_close_val < previous_flb):
                flb_list.append(current_blb)
            else:
                flb_list.append(previous_flb)
    
    if n == 0: # Handle empty dataframe case
        df = df.with_columns([
            pl.Series("FUB", [], dtype=pl.Float64),
            pl.Series("FLB", [], dtype=pl.Float64),
            pl.Series("Supertrend", [], dtype=pl.Float64),
            pl.Series("Supertrend_Direction", [], dtype=pl.String) # Ensure String dtype for empty
        ])
        return df

    # If n > 0, proceed with FUB/FLB list creation and column addition
    df = df.with_columns([
        pl.Series("FUB", fub_list),
        pl.Series("FLB", flb_list)
    ])
      
    # Calculate Supertrend (ST)
    st_list = []

    # Get Series for faster access (FUB and FLB are now definitely in df)
    fub_series = df['FUB']
    flb_series = df['FLB']
    # close_series is already defined above

    # First row for ST
    st_list.append(fub_series[0]) # Initialize with FUB_0

    # Iterate for subsequent rows for ST (from index 1)
    for i in range(1, n):
        current_close = close_series[i]
        current_fub = fub_series[i]
        current_flb = flb_series[i]
        previous_st = st_list[i-1]
        # previous_fub_val and previous_flb_val are used for comparison with previous_st
        previous_fub_val = fub_series[i-1] 
        previous_flb_val = flb_series[i-1]

        # ST Logic
        if previous_st == previous_fub_val:
            if current_close <= current_fub:
                st_list.append(current_fub)
            else: # current_close > current_fub
                st_list.append(current_flb)
        elif previous_st == previous_flb_val:
            if current_close >= current_flb:
                st_list.append(current_flb)
            else: # current_close < current_flb
                st_list.append(current_fub)
        else:
            # This case implies previous_st was neither previous_fub_val nor previous_flb_val.
            # This could happen if the initial ST value (fub_series[0]) is NaN, or if data is unusual.
            # Defaulting to current_fub as a fallback, or choose a more robust default if necessary.
            st_list.append(current_fub) 

    df = df.with_columns(pl.Series("Supertrend", st_list))

    # Calculate Supertrend Direction
    supertrend_direction = pl.when(df['Close'] > df['Supertrend'])\
                             .then(pl.lit('Up'))\
                             .otherwise(pl.lit('Down'))
    df = df.with_columns(supertrend_direction.alias('Supertrend_Direction'))
    
    return df
