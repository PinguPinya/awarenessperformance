"""Manual issuer → Yahoo-Finance ticker map for Situational Awareness LP holdings.

40 distinct issuers across 5 quarters. Hardcoded because OpenFIGI returns bond
tickers for the few convertible-CUSIPs present, and the universe is small.
"""

TICKER_MAP: dict[str, str] = {
    "APPLIED DIGITAL": "APLD",
    "BABCOCK AND WILCOX": "BW",
    "BITDEER": "BTDR",
    "BITFARMS": "BITF",
    "BLOOM ENERGY": "BE",
    "BROADCOM": "AVGO",
    "CIPHER MINING": "CIFR",
    "CLEANSPARK": "CLSK",
    "COHERENT": "COHR",
    "CONSTELLATION ENERGY": "CEG",
    "CORE SCIENTIFIC INC NEW": "CORZ",
    "COREWEAVE": "CRWV",
    "EQT": "EQT",
    "GALAXY DIGITAL INC": "GLXY",
    "HUT 8": "HUT",
    "INFOSYS": "INFY",
    "INTEL": "INTC",
    "IREN": "IREN",
    "KILROY": "KRC",
    "LIBERTY ENERGY": "LBRT",
    "LUMENTUM": "LITE",
    "MARVELL": "MRVL",
    "MICRON": "MU",
    "MODINE": "MOD",
    "NVIDIA": "NVDA",
    "ONTO INNOVATION": "ONTO",
    "POWER SOLUTIONS INTL": "PSIX",
    "PROPETRO": "PUMP",
    "RIOT PLATFORMS": "RIOT",
    "SANDISK": "SNDK",
    "SEAGATE TECHNOLOGY HLDNGS PL": "STX",
    "SOLARIS ENERGY INFRAS": "SEI",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "TALEN ENERGY": "TLN",
    "TOWER SEMICONDUCTOR": "TSEM",
    "VANECK ETF TRUST": "SMH",
    "VERTIV HOLDINGS": "VRT",
    "VISTRA": "VST",
    "WESTERN DIGITAL": "WDC",
    "WHITEFIBER": "WYFI",
    # Added in Q1 2026 filing:
    "ADVANCED MICRO DEVICES": "AMD",
    "ASML HLDG NV N Y REGISTRY": "ASML",
    "CORNING": "GLW",
    "HIVE DIGITAL TECHNOLOGIES LT": "HIVE",
    "ORACLE": "ORCL",
    "SHARONAI": "SHAZ",
    "T1 ENERGY": "TE",
    # Same issuer as "TAIWAN SEMICONDUCTOR" but new raw-name in Q1 2026:
    "TAIWAN SEMICONDUCTOR MANUFAC": "TSM",
}

