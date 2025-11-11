
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- НАСТРОЙКИ ----------
REFRESH_SEC = 10
LOOKBACK_MIN = 180
INTERVAL = "1m"
MIN_BARS = 50

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD":
