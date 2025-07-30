# Placeholder module
class TimeSeriesProcessor:
    def add_time_features(self, df): return df
    def add_lag_features(self, df, lags): return df
    def create_sequences(self, data, seq_len, features): return [[0.1] * len(features)]
