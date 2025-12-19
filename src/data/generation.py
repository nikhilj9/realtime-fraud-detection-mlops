"""Synthetic transaction data enrichment."""

import argparse
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import Config, load_config, get_logger
from src.utils.exceptions import DataLoadError

logger = get_logger(__name__)


def assign_by_segment(
    df: pd.DataFrame,
    column: str,
    distributions: Dict[str, Dict[str, float]]
) -> None:
    """Assign categorical values based on amount_segment."""
    df[column] = "Unknown"
    for segment in ["low", "mid", "high"]:
        mask = df["amount_segment"] == segment
        if mask.sum() == 0:
            continue
        dist = distributions.get(f"{segment}_amount", {})
        if not dist:
            continue
        cats, probs = list(dist.keys()), list(dist.values())
        df.loc[mask, column] = np.random.choice(cats, size=mask.sum(), p=probs)


def assign_conditional(
    df: pd.DataFrame,
    column: str,
    mask: pd.Series,
    choices: List[str],
    probs: List[float]
) -> None:
    """Assign values to subset of df."""
    if mask.sum() > 0:
        df.loc[mask, column] = np.random.choice(choices, size=mask.sum(), p=probs)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw credit card data."""
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def add_ids(df: pd.DataFrame, n_users: int, n_merchants: int) -> None:
    """Add transaction, user, and merchant IDs."""
    n = len(df)
    
    df["transaction_id"] = [f"txn_{uuid.uuid4().hex[:12]}" for _ in range(n)]
    
    user_idx = np.clip(np.random.zipf(1.5, n), 1, n_users)
    df["user_id"] = pd.Series(user_idx).apply(lambda x: f"user_{x:05d}")
    
    merchant_idx = np.clip(np.random.zipf(1.3, n), 1, n_merchants)
    df["merchant_id"] = pd.Series(merchant_idx).apply(lambda x: f"merchant_{x:05d}")
    
    logger.info(f"IDs: {df['user_id'].nunique():,} users, {df['merchant_id'].nunique():,} merchants")


def add_temporal_features(df: pd.DataFrame, start_date: str, exchange_rate: float) -> None:
    """Add timestamp and convert amount to INR."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    df["timestamp"] = df["Time"].apply(lambda x: start + timedelta(seconds=int(x)))
    
    df["amount_inr"] = (df["Amount"] * exchange_rate).round(2)
    df.loc[df["amount_inr"] == 0, "amount_inr"] = 1.0
    
    logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def add_amount_segments(df: pd.DataFrame) -> Tuple[float, float]:
    """Add amount_segment column."""
    low = df["amount_inr"].quantile(0.25)
    high = df["amount_inr"].quantile(0.90)
    
    df["amount_segment"] = pd.cut(
        df["amount_inr"],
        bins=[-np.inf, low, high, np.inf],
        labels=["low", "mid", "high"]
    )
    
    logger.info(f"Segments: low≤₹{low:,.0f}, high>₹{high:,.0f}")
    return low, high


def add_card_attributes(
    df: pd.DataFrame,
    network_dist: Dict[str, Dict[str, float]],
    low_limit: float,
    high_limit: float
) -> None:
    """Add card network, tier, issuer, and profile."""
    
    assign_by_segment(df, "card_network", network_dist)
    
    tiers = ["Classic", "Gold", "Platinum", "Signature"]
    df["card_tier"] = "Classic"
    
    mask_amex = df["card_network"] == "Amex"
    assign_conditional(df, "card_tier", mask_amex, tiers, [0.0, 0.3, 0.4, 0.3])
    
    mask_other = ~mask_amex
    assign_conditional(df, "card_tier", mask_other & (df["amount_inr"] <= low_limit), tiers, [0.6, 0.3, 0.08, 0.02])
    assign_conditional(df, "card_tier", mask_other & (df["amount_inr"] > low_limit) & (df["amount_inr"] <= high_limit), tiers, [0.3, 0.4, 0.2, 0.1])
    assign_conditional(df, "card_tier", mask_other & (df["amount_inr"] > high_limit), tiers, [0.1, 0.2, 0.4, 0.3])
    
    banks = ["HDFC Bank", "SBI Card", "ICICI Bank", "Axis Bank", "Kotak Mahindra",
             "IndusInd Bank", "RBL Bank", "IDFC First", "Yes Bank", "Standard Chartered"]
    bank_probs = [0.27, 0.19, 0.17, 0.14, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03]
    
    df["card_issuer"] = np.random.choice(banks, size=len(df), p=bank_probs)
    df.loc[df["card_network"] == "Amex", "card_issuer"] = "American Express"
    
    df["card_age"] = np.random.randint(1, 61, size=len(df))
    
    limits = {"Classic": (25000, 100000), "Gold": (100000, 300000),
              "Platinum": (300000, 750000), "Signature": (750000, 1500000)}
    
    df["credit_limit"] = 50000
    for tier, (lo, hi) in limits.items():
        mask = df["card_tier"] == tier
        df.loc[mask, "credit_limit"] = (np.random.randint(lo, hi, size=mask.sum()) // 1000) * 1000
    
    mask_over = df["amount_inr"] > df["credit_limit"]
    df.loc[mask_over, "credit_limit"] = (df.loc[mask_over, "amount_inr"] * 1.2).astype(int)
    
    logger.info("Card attributes added")


def add_merchant_attributes(df: pd.DataFrame, category_dist: Dict[str, Dict[str, float]]) -> None:
    """Add merchant category and location."""
    
    assign_by_segment(df, "merchant_category", category_dist)
    
    cities = (
        ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"] +
        ["Jaipur", "Lucknow", "Chandigarh", "Indore", "Kochi", "Surat", "Nagpur", "Coimbatore", "Bhopal", "Patna"] +
        ["Varanasi", "Agra", "Nashik", "Vadodara", "Ludhiana", "Madurai", "Vizag", "Guwahati", "Bhubaneswar", "Raipur"]
    )
    weights = [0.075]*8 + [0.030]*10 + [0.010]*10
    
    df["merchant_city"] = np.random.choice(cities, size=len(df), p=weights)
    
    city_state = {
        "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra", "Nashik": "Maharashtra",
        "Delhi": "Delhi", "Bengaluru": "Karnataka",
        "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Madurai": "Tamil Nadu",
        "Hyderabad": "Telangana", "Kolkata": "West Bengal",
        "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat",
        "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Agra": "Uttar Pradesh",
        "Varanasi": "Uttar Pradesh", "Chandigarh": "Chandigarh",
        "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
        "Kochi": "Kerala", "Patna": "Bihar", "Ludhiana": "Punjab",
        "Vizag": "Andhra Pradesh", "Guwahati": "Assam",
        "Bhubaneswar": "Odisha", "Raipur": "Chhattisgarh"
    }
    df["merchant_state"] = df["merchant_city"].map(city_state)
    
    logger.info(f"Merchant: {df['merchant_category'].nunique()} categories, {df['merchant_city'].nunique()} cities")


def add_transaction_attributes(df: pd.DataFrame) -> None:
    """Add channel, entry mode, and boolean flags."""
    
    df["transaction_channel"] = "POS"
    
    online_cats = ["Digital Services", "Airline", "Entertainment", "Public Transport", "Hotel"]
    df.loc[df["merchant_category"].isin(online_cats), "transaction_channel"] = "Online"
    
    mixed = {"Dining": 0.3, "Grocery": 0.3, "Supermarket": 0.3, "Fashion": 0.5,
             "Electronics": 0.5, "Furniture": 0.5, "Jewelry": 0.5, "Utility": 0.6, "Pharmacy": 0.6}
    
    for cat, online_prob in mixed.items():
        mask = df["merchant_category"] == cat
        assign_conditional(df, "transaction_channel", mask, ["Online", "POS"], [online_prob, 1-online_prob])
    
    df["entry_mode"] = "Unknown"
    df.loc[df["transaction_channel"] == "Online", "entry_mode"] = "CVC"
    
    mask_legit_pos = (df["Class"] == 0) & (df["transaction_channel"] == "POS")
    assign_conditional(df, "entry_mode", mask_legit_pos & (df["amount_inr"] <= 5000), ["Tap", "Chip", "Swipe"], [0.585, 0.4, 0.015])
    assign_conditional(df, "entry_mode", mask_legit_pos & (df["amount_inr"] > 5000), ["Chip", "Swipe"], [0.985, 0.015])
    
    mask_fraud_pos = (df["Class"] == 1) & (df["transaction_channel"] == "POS")
    assign_conditional(df, "entry_mode", mask_fraud_pos, ["Swipe", "Tap", "Chip"], [0.5, 0.4, 0.1])
    
    df["is_recurring"] = 0
    df.loc[df["merchant_category"] == "Utility", "is_recurring"] = 1
    mask_digital = df["merchant_category"] == "Digital Services"
    assign_conditional(df, "is_recurring", mask_digital, [1, 0], [0.8, 0.2])
    
    df["is_international"] = 0
    assign_conditional(df, "is_international", df["Class"] == 0, [0, 1], [0.98, 0.02])
    assign_conditional(df, "is_international", df["Class"] == 1, [0, 1], [0.75, 0.25])
    df.loc[df["card_network"] == "RuPay", "is_international"] = 0
    
    logger.info("Transaction attributes added")


def finalize(df: pd.DataFrame) -> None:
    """Rename target, drop original columns, reorder."""
    df.rename(columns={"Class": "is_fraud"}, inplace=True)
    df.drop(columns=["Time", "Amount", "amount_segment"], inplace=True, errors="ignore")
    
    col_order = (
        ["transaction_id", "user_id", "merchant_id", "timestamp"] +
        ["card_network", "card_issuer", "card_tier", "credit_limit", "card_age"] +
        ["amount_inr", "merchant_category", "merchant_city", "merchant_state",
         "transaction_channel", "entry_mode", "is_international", "is_recurring"] +
        [f"V{i}" for i in range(1, 29)] +
        ["is_fraud"]
    )
    
    existing = [c for c in col_order if c in df.columns]
    for i, col in enumerate(existing):
        df.insert(i, col, df.pop(col))
    
    logger.info(f"Final shape: {df.shape}")


def generate(config: Config) -> pd.DataFrame:
    """Run full enrichment pipeline."""
    np.random.seed(config.generation.random_seed)
    
    logger.info("Starting data generation")
    
    df = load_raw_data(config.paths.raw_data)
    
    add_ids(df, config.generation.n_users, config.generation.n_merchants)
    add_temporal_features(df, config.generation.start_date, config.generation.exchange_rate)
    
    low, high = add_amount_segments(df)
    
    if config.distributions:
        add_card_attributes(df, config.distributions["card_networks"].model_dump(), low, high)
        add_merchant_attributes(df, config.distributions["merchant_categories"].model_dump())
    
    add_transaction_attributes(df)
    finalize(df)
    
    logger.info("Generation complete")
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    """Save to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enriched transaction data")
    parser.add_argument("--config", type=Path, default=Path("src/config/config.yaml"))
    args = parser.parse_args()
    
    config = load_config(args.config)
    df = generate(config)
    
    output_path = config.paths.processed_data / config.generation.output_filename
    save(df, output_path)


if __name__ == "__main__":
    main()