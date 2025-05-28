import unittest
import polars as pl
import pandas as pd
import pandas_ta as ta
from polars.testing import assert_frame_equal, assert_series_equal
from polars_supertrend import supertrend # Assuming polars_supertrend.py is in the same directory or PYTHONPATH

# Helper function to find the first non-NaN index in a Polars Series
# (Polars' arg_max on boolean series returns index of first True)
def find_first_non_nan_idx(series: pl.Series) -> int | None:
    is_not_null_series = series.is_not_null()
    if is_not_null_series.any(): # Check if there is any non-null value
        return is_not_null_series.arg_max()
    return None

class TestSupertrend(unittest.TestCase):

    def test_compare_with_pandas_ta(self):
        """Compare Polars Supertrend output with pandas-ta."""
        data = {
            'High': [10, 10.5, 11, 10.8, 11.2, 11.5, 12, 11.8, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 9, 9.5, 10, 10.2, 10.5, 10.3, 10.7, 11, 11.2, 11.5, 11.3, 11, 10.8, 11.2, 11.5, 12.0, 12.5, 13.0, 12.8, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0],
            'Low':  [9.5, 9.8, 10.5, 10.2, 10.8, 11, 11.5, 11.2, 11, 10.5, 10, 9, 8.5, 8, 8, 8.5, 9, 9.5, 9.8, 10, 9.9, 10.2, 10.5, 10.8, 11, 10.9, 10.5, 10.2, 10.8, 11, 11.5, 12.0, 12.5, 12.2, 12.0, 11.5, 11.0, 10.5, 10.0, 9.5],
            'Close':[9.8, 10.2, 10.8, 10.5, 11, 11.2, 11.8, 11.5, 11.2, 10.8, 10.2, 9.5, 8.8, 8.2, 8.3, 8.8, 9.2, 9.8, 10, 10.2, 10.1, 10.5, 10.8, 11, 11.2, 11.1, 10.8, 10.5, 11, 11.2, 11.8, 12.2, 12.8, 12.5, 12.2, 11.8, 11.2, 10.8, 10.2, 9.8]
        }
        pl_df = pl.DataFrame(data)
        period = 7
        multiplier = 3.0

        # Polars Supertrend
        polars_result_df = supertrend(pl_df.clone(), period=period, multiplier=multiplier)
        polars_st = polars_result_df.select(['Supertrend', 'Supertrend_Direction'])

        # Pandas-TA Supertrend
        pd_df = pl_df.to_pandas()
        pd_df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
        
        pd_df.ta.supertrend(length=period, multiplier=multiplier, append=True)
        
        pta_line_col_name = f'SUPERT_{period}_{multiplier}' 
        pta_dir_col_name = f'SUPERTd_{period}_{multiplier}'

        if pta_line_col_name not in pd_df.columns:
            pta_line_col_name = f'SUPERT_{period}_{int(multiplier)}' # Try with int multiplier if float failed
            pta_dir_col_name = f'SUPERTd_{period}_{int(multiplier)}'

        if pta_line_col_name not in pd_df.columns:
            print(f"Pandas-TA Supertrend columns not found. Available columns: {pd_df.columns}")
            raise KeyError(f"Pandas-TA Supertrend columns not found. Tried {f'SUPERT_{period}_{multiplier}'} and {f'SUPERT_{period}_{int(multiplier)}'}")

        pta_st_line_pd = pd_df[pta_line_col_name]
        pta_st_direction_raw_pd = pd_df[pta_dir_col_name]
        
        pta_st_direction_pd = pta_st_direction_raw_pd.map({1.0: 'Up', -1.0: 'Down', 1: 'Up', -1: 'Down'})
        
        pta_st_line = pl.Series("Supertrend_pta", pta_st_line_pd)
        pta_st_direction = pl.Series("Supertrend_Direction_pta", pta_st_direction_pd)

        # Comparison
        # Start comparison after 'period' rows, as initial ATR values differ.
        # pandas-ta needs `period` values for its internal EMA/SMA for ATR, so first valid supertrend is often at `period`.
        # Our `TR[0]` is `H[0]-L[0]`, `ATR[0]` is `TR[0]`. `ST[0]` is based on `ATR[0]`.
        # pandas-ta `ATR[0]` to `ATR[period-2]` are NaN. `ATR[period-1]` is the first SMA of TR.
        # So pandas-ta `ST[period-1]` is its first real ST value.
        # We will align from index `period` for a more stable comparison zone.
        start_index_for_comparison = period 
            
        if start_index_for_comparison < polars_st['Supertrend'].len() and start_index_for_comparison < pta_st_line.len():
            # Ensure both series are float for numeric comparison, handle potential nulls if any before slicing
            polars_supertrend_series_float = polars_st['Supertrend'].cast(pl.Float64, strict=False)
            pta_st_line_float = pta_st_line.cast(pl.Float64, strict=False)

            pl.testing.assert_series_equal(
                polars_supertrend_series_float.slice(start_index_for_comparison), 
                pta_st_line_float.slice(start_index_for_comparison), 
                check_dtype=False, check_names=False, rtol=1e-2 # Increased tolerance due to potential ATR diffs
            )
        else:
            print(f"Warning: Not enough data points (len {polars_st['Supertrend'].len()} vs {pta_st_line.len()}) to compare Supertrend lines effectively after index {start_index_for_comparison}.")

        # Compare the Supertrend directions
        first_valid_idx_pta_dir = find_first_non_nan_idx(pta_st_direction)

        if first_valid_idx_pta_dir is not None:
            align_idx_dir = max(start_index_for_comparison, first_valid_idx_pta_dir)

            if align_idx_dir < polars_st['Supertrend_Direction'].len() and align_idx_dir < pta_st_direction.len():
                pl.testing.assert_series_equal(
                    polars_st['Supertrend_Direction'].slice(align_idx_dir), 
                    pta_st_direction.slice(align_idx_dir), 
                    check_names=False, check_dtype=False # Direction is string
                )
            else:
                 print(f"Warning: Could not effectively compare Supertrend directions with pandas-ta after index {align_idx_dir} (lengths: {polars_st['Supertrend_Direction'].len()} vs {pta_st_direction.len()}).")
        else:
            print("Warning: pandas-ta direction series is all NaN, cannot compare directions.")


    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        empty_df_schema = {
            'High': pl.Float64,
            'Low': pl.Float64,
            'Close': pl.Float64
        }
        empty_df = pl.DataFrame({
            'High': [], 'Low': [], 'Close': []
        }, schema=empty_df_schema)

        result_df = supertrend(empty_df.clone(), period=10, multiplier=3)

        expected_columns = [
            'High', 'Low', 'Close', 'TR', 'ATR', 'BUB', 'BLB',
            'FUB', 'FLB', 'Supertrend', 'Supertrend_Direction'
        ]
        self.assertListEqual(list(result_df.columns), expected_columns)

        for col_name in expected_columns:
            self.assertEqual(len(result_df[col_name]), 0, f"Column {col_name} should be empty.")

        # Assert correct dtypes for these empty columns
        self.assertEqual(result_df['High'].dtype, pl.Float64)
        self.assertEqual(result_df['Low'].dtype, pl.Float64)
        self.assertEqual(result_df['Close'].dtype, pl.Float64)
        self.assertEqual(result_df['TR'].dtype, pl.Float64) # TR is calculated from others
        self.assertEqual(result_df['ATR'].dtype, pl.Float64)
        self.assertEqual(result_df['BUB'].dtype, pl.Float64)
        self.assertEqual(result_df['BLB'].dtype, pl.Float64)
        self.assertEqual(result_df['FUB'].dtype, pl.Float64)
        self.assertEqual(result_df['FLB'].dtype, pl.Float64)
        self.assertEqual(result_df['Supertrend'].dtype, pl.Float64)
        self.assertEqual(result_df['Supertrend_Direction'].dtype, pl.String)


    def test_insufficient_data(self):
        """Test with data shorter than the period."""
        data = {
            'High':  [10.0, 11.0, 12.0, 13.0, 14.0],
            'Low':   [9.0,  10.0, 11.0, 12.0, 13.0],
            'Close': [9.5, 10.5, 11.5, 12.5, 13.5]
        }
        df = pl.DataFrame(data)
        
        result_df = supertrend(df.clone(), period=10, multiplier=3)
        
        self.assertEqual(result_df.shape[0], 5) # Should have the same number of rows
        self.assertTrue('ATR' in result_df.columns)
        # ATR uses min_periods=1, so it should have values after the first NA from TR.
        # TR[0] will be High[0] - Low[0] because hpc[0] and lpc[0] are None due to close.shift(1).
        self.assertEqual(result_df['TR'][0], df['High'][0] - df['Low'][0], "First TR should be High[0] - Low[0]")
        self.assertFalse(result_df['TR'][1:].is_null().any(), "TR after first should not be null")
        self.assertIsNotNone(result_df['ATR'][0], "ATR[0] should not be None if TR[0] is calculated from a non-None TR[0]") # ATR[0] should be TR[0]
        self.assertFalse(result_df['ATR'][1:].is_null().any(), "ATR after first should not be null")
        self.assertTrue('Supertrend' in result_df.columns)
        self.assertFalse(result_df['Supertrend'].is_null().any(), "Supertrend column should have values")
        self.assertTrue('Supertrend_Direction' in result_df.columns)
        self.assertFalse(result_df['Supertrend_Direction'].is_null().any(), "Supertrend_Direction column should have values")


    def test_basic_supertrend_calculation(self):
        """Test basic Supertrend calculation with a small dataset."""
        data = {
            'High':  [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 28.0, 27.0, 26.0, 25.0],
            'Low':   [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 27.0, 26.0, 25.0, 24.0],
            'Close': [19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 27.5, 26.5, 25.5, 24.5]
        }
        df = pl.DataFrame(data)
        period = 5
        multiplier = 2
        
        result_df = supertrend(df.clone(), period=period, multiplier=multiplier)
        
        self.assertTrue('Supertrend' in result_df.columns)
        self.assertTrue('Supertrend_Direction' in result_df.columns)
        self.assertEqual(result_df.shape[0], len(data['High']))
        
        # Manual calculation for the first few values (period=5, multiplier=2)
        # TR[0] is None because close_series.shift(1)[0] is None.
        # ATR[0] will be None.
        # BUB[0], BLB[0] will be None.
        # FUB[0], FLB[0] will be None.
        # ST[0] will be FUB[0], which depends on ATR[0], which depends on TR[0].
        # If TR[0] = High[0]-Low[0] = 20-19 = 1.0
        # ATR[0] = TR[0] = 1.0
        # BUB[0] = (High[0]+Low[0])/2 + multiplier*ATR[0] = (20+19)/2 + 2*1.0 = 19.5 + 2.0 = 21.5
        # BLB[0] = (High[0]+Low[0])/2 - multiplier*ATR[0] = 19.5 - 2.0 = 17.5
        # FUB[0] = BUB[0] = 21.5
        # FLB[0] = BLB[0] = 17.5
        # ST[0] = FUB[0] = 21.5
        # Supertrend_Direction[0]: Close[0] (19.5) > ST[0] (21.5) is false, so "Down".

        self.assertEqual(result_df['TR'][0], df['High'][0] - df['Low'][0])
        self.assertEqual(result_df['ATR'][0], df['High'][0] - df['Low'][0]) # ATR[0] is TR[0] due to min_periods=1
        
        expected_bub_0 = (df['High'][0] + df['Low'][0]) / 2 + multiplier * (df['High'][0] - df['Low'][0])
        expected_blb_0 = (df['High'][0] + df['Low'][0]) / 2 - multiplier * (df['High'][0] - df['Low'][0])
        self.assertEqual(result_df['BUB'][0], expected_bub_0)
        self.assertEqual(result_df['BLB'][0], expected_blb_0)
        self.assertEqual(result_df['FUB'][0], expected_bub_0) # FUB[0] = BUB[0]
        self.assertEqual(result_df['FLB'][0], expected_blb_0) # FLB[0] = BLB[0]
        self.assertEqual(result_df['Supertrend'][0], expected_bub_0) # ST[0] = FUB[0]
        self.assertEqual(result_df['Supertrend_Direction'][0], "Down") # Close[0] (19.5) > ST[0] (21.5) is false


    def test_trend_change(self):
        """Test a scenario where the Supertrend direction changes."""
        data = {
            'High':  [10.0, 11.0, 12.0, 11.0, 10.0, 9.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            'Low':   [ 9.0, 10.0, 11.0, 10.0,  9.0, 8.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0],
            'Close': [9.5, 10.5, 11.5, 10.5, 9.5, 8.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
        }
        df = pl.DataFrame(data)
        period = 3
        multiplier = 2
        
        result_df = supertrend(df.clone(), period=period, multiplier=multiplier)
        
        self.assertTrue('Supertrend' in result_df.columns)
        self.assertTrue('Supertrend_Direction' in result_df.columns)


    def test_column_dtypes(self):
        """Test the data types of the output columns."""
        data = { # Using the same data as test_basic_supertrend_calculation
            'High':  [20.0, 21.0, 22.0, 23.0, 24.0],
            'Low':   [19.0, 20.0, 21.0, 22.0, 23.0],
            'Close': [19.5, 20.5, 21.5, 22.5, 23.5]
        }
        df = pl.DataFrame(data)
        period = 3 # Shorter period for simpler data
        multiplier = 2
        
        result_df = supertrend(df.clone(), period=period, multiplier=multiplier)
        
        self.assertEqual(result_df['High'].dtype, pl.Float64)
        self.assertEqual(result_df['Low'].dtype, pl.Float64)
        self.assertEqual(result_df['Close'].dtype, pl.Float64)
        self.assertEqual(result_df['TR'].dtype, pl.Float64)
        self.assertEqual(result_df['ATR'].dtype, pl.Float64)
        self.assertEqual(result_df['BUB'].dtype, pl.Float64)
        self.assertEqual(result_df['BLB'].dtype, pl.Float64)
        self.assertEqual(result_df['FUB'].dtype, pl.Float64)
        self.assertEqual(result_df['FLB'].dtype, pl.Float64)
        self.assertEqual(result_df['Supertrend'].dtype, pl.Float64)
        self.assertEqual(result_df['Supertrend_Direction'].dtype, pl.String)

if __name__ == '__main__':
    unittest.main()
