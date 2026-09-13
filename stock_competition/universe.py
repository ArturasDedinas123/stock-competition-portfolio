"""Stocks the simulated rivals pick from: well-known US-listed names popular with students.

Only stocks that traded over the whole simulation window (since October 2020) are used; the
notebook lists any that are dropped. Edit freely to model a different competition.
"""

POPULAR_STOCKS = (
    # Technology
    "MSFT", "AMZN", "META", "TSLA", "AMD", "AVGO", "NFLX", "ORCL", "CRM", "ADBE",
    "INTC", "QCOM", "MU", "TXN", "AMAT", "IBM", "CSCO", "NOW", "INTU", "PANW",
    "PLTR", "SHOP", "UBER", "SNOW", "SPOT", "PYPL", "TSM", "ASML", "CRWD", "NET",
    "DDOG", "ZM", "ROKU", "PINS", "ETSY", "EBAY", "SMCI", "DELL", "MRVL", "ANET",
    # Consumer
    "WMT", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "KO", "PEP", "PG",
    "DIS", "LULU", "CMG", "BKNG", "MAR", "YUM", "EL", "F", "GM", "NIO",
    # Financials
    "V", "MA", "BAC", "GS", "MS", "WFC", "C", "BRK-B", "AXP", "SCHW", "BLK",
    # Health care
    "LLY", "UNH", "JNJ", "PFE", "MRNA", "ABBV", "MRK", "TMO", "ISRG", "NVO", "AMGN",
    # Energy, industrials and materials
    "XOM", "CVX", "COP", "OXY", "BA", "GE", "LMT", "RTX", "HON", "UPS", "FDX", "ENPH", "FSLR", "FCX",
    # Communication
    "T", "VZ", "TMUS", "CMCSA",
)
