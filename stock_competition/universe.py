"""Stock lists.

``POPULAR_STOCKS``      well-known companies the simulated rival students pick from
``INDEX_STOCKS``        S&P 500 and Nasdaq-100 members (a 2025 snapshot, not kept up to date)
``SPECULATIVE_STOCKS``  meme, crypto-linked and other very volatile stocks popular with retail investors

Only stocks with complete prices over the simulation window (since October 2020) are used; the
notebook lists any that are dropped. Edit freely to model a different competition.
"""

POPULAR_STOCKS = (
    # Technology and internet
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO", "INTC",
    "QCOM", "MU", "TXN", "AMAT", "LRCX", "KLAC", "ADI", "NXPI", "ON", "MRVL",
    "TSM", "ASML", "SMCI", "DELL", "HPQ", "IBM", "CSCO", "ORCL", "CRM", "ADBE",
    "NOW", "INTU", "WDAY", "ADSK", "SNPS", "CDNS", "PANW", "FTNT", "CRWD", "ZS",
    "NET", "DDOG", "SNOW", "MDB", "TEAM", "TWLO", "OKTA", "DOCU", "HUBS", "ZM",
    "PLTR", "SHOP", "UBER", "LYFT", "SPOT", "NFLX", "ROKU", "PINS", "SNAP", "TTD",
    "PYPL", "ETSY", "EBAY", "W", "CHWY", "MELI", "SE", "BABA", "JD", "PDD",
    "BIDU", "SONY", "ANET", "BSY", "U", "TTWO", "MTCH", "EXPE", "BKNG", "DKNG",
    # Consumer
    "WMT", "COST", "TGT", "HD", "LOW", "TJX", "ROST", "DG", "DLTR", "KR",
    "MCD", "SBUX", "CMG", "DPZ", "YUM", "WING", "DRI", "KO", "PEP", "MNST",
    "CELH", "PG", "CL", "KMB", "EL", "ULTA", "NKE", "LULU", "DECK", "CROX",
    "RL", "TPR", "DIS", "WBD", "CMCSA", "MAR", "HLT", "RCL", "CCL", "NCLH",
    "LVS", "WYNN", "MGM", "DAL", "UAL", "AAL", "LUV", "F", "GM", "TM",
    "RACE", "NIO", "LI", "XPEV", "CVNA", "PTON",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "BX", "KKR",
    "V", "MA", "AXP", "COF", "SYF", "ALLY", "SPGI", "MCO", "ICE", "CME",
    "BRK-B",
    # Health care
    "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "REGN",
    "VRTX", "BIIB", "MRNA", "TMO", "ISRG", "DXCM", "NVO", "CVS", "CI", "ZTS",
    # Energy, industrials, materials and utilities
    "XOM", "CVX", "COP", "OXY", "SLB", "EOG", "BA", "GE", "LMT", "RTX",
    "HON", "CAT", "DE", "MMM", "UPS", "FDX", "UNP", "ETN", "ENPH", "FSLR",
    "FCX", "NEM", "NUE", "NEE", "DUK",
    # Telecom
    "T", "VZ", "TMUS",
)

INDEX_STOCKS = (
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP",
    "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME",
    "AMGN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP", "APTV", "ARE",
    "ARM", "ASML", "ATO", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZN", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR",
    "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BX", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCEP", "CCI", "CCL", "CDNS", "CDW", "CEG",
    "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS",
    "CNC", "CNP", "COF", "COIN", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM",
    "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DASH", "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM",
    "EQIX", "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE",
    "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS", "FITB", "FOX", "FOXA",
    "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GFS", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG",
    "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HOOD", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBKR", "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR",
    "IRM", "ISRG", "IT", "ITW", "IVZ",
    "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE",
    "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU",
    "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDB", "MDLZ", "MDT", "MELI", "MET", "META",
    "MGM", "MHK", "MKC", "MKTX", "MLM", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRVL", "MS", "MSCI", "MSFT", "MSI", "MSTR", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PDD", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM",
    "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX",
    "PTC", "PWR", "PYPL",
    "QCOM",
    "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SHOP", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG",
    "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TEAM", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS",
    "TPL", "TPR", "TRGP", "TRI", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN",
    "TXT", "TYL",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ",
    "WAB", "WAT", "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST",
    "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL", "XYZ",
    "YUM",
    "ZBH", "ZBRA", "ZS", "ZTS",
)

SPECULATIVE_STOCKS = (
    "GME", "AMC", "BB", "NOK", "MSTR", "RIOT", "MARA", "CLSK", "PLUG", "FCEL",
    "BLNK", "TLRY", "SNDL", "CGC", "CVNA", "DKNG", "BYND", "PTON", "NVAX", "SPCE",
    "CRSP", "TDOC", "ROKU", "FUBO", "NIO", "XPEV", "LI", "SEDG", "RUN", "CELH",
    "OCGN", "BNGO", "WKHS", "APPS",
)
