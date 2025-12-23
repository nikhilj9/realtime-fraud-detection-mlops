"""
Expectation suite definitions for fraud detection data.

Suites:
1. raw_data_suite: Validates raw creditcard.csv BEFORE feature engineering
2. processed_data_suite: Validates enriched data BEFORE training

Uses GX Core 1.10.0 class-based expectations.
"""

from enum import Enum

import great_expectations as gx
import pandas as pd
from great_expectations.data_context import AbstractDataContext
from great_expectations.expectations import (
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToNotBeNull,
    ExpectTableColumnsToMatchSet,
)


class SuiteType(Enum):
    """Available validation suite types."""

    RAW = "raw_data_suite"
    PROCESSED = "processed_data_suite"


# =============================================================================
# RAW DATA SCHEMA (creditcard.csv)
# =============================================================================
RAW_DATA_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Class",
]

RAW_REQUIRED_NOT_NULL = ["Time", "Amount", "Class"]


# =============================================================================
# PROCESSED DATA SCHEMA (transactions_enriched.parquet)
# =============================================================================
PROCESSED_DATA_COLUMNS = [
    # IDs
    "transaction_id",
    "user_id",
    "merchant_id",
    # Timestamp
    "timestamp",
    # Card information
    "card_network",
    "card_issuer",
    "card_tier",
    "credit_limit",
    "card_age",
    # Transaction details
    "amount_inr",
    "merchant_category",
    "merchant_city",
    "merchant_state",
    "transaction_channel",
    "entry_mode",
    # Flags
    "is_international",
    "is_recurring",
    # PCA features
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    # Target
    "is_fraud",
]

PROCESSED_REQUIRED_NOT_NULL = [
    "transaction_id",
    "user_id",
    "amount_inr",
    "is_fraud",
]


# =============================================================================
# SUITE CREATORS
# =============================================================================
def create_raw_data_suite(context: AbstractDataContext) -> gx.ExpectationSuite:
    """
    Create expectation suite for raw creditcard.csv data.

    Validates:
    - All 31 columns exist
    - Critical columns have no nulls
    - Amount >= 0
    - Time >= 0
    - Class is binary (0 or 1)
    """
    suite = context.suites.add(gx.ExpectationSuite(name=SuiteType.RAW.value))

    # Schema validation
    suite.add_expectation(
        ExpectTableColumnsToMatchSet(
            column_set=RAW_DATA_COLUMNS,
            exact_match=False,
        )
    )

    # Null checks
    for col in RAW_REQUIRED_NOT_NULL:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    # Range checks
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="Amount", min_value=0.0, max_value=None)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="Time", min_value=0.0, max_value=None)
    )

    # Target binary check
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="Class", value_set=[0, 1])
    )

    suite.save()
    print(f"Created suite '{suite.name}' with {len(suite.expectations)} expectations")
    return suite


def create_processed_data_suite(context: AbstractDataContext) -> gx.ExpectationSuite:
    """
    Create expectation suite for processed/enriched transaction data.

    Validates:
    - All 46 columns exist
    - Critical columns have no nulls
    - Numeric columns are non-negative
    - Binary columns are 0 or 1
    """
    suite = context.suites.add(gx.ExpectationSuite(name=SuiteType.PROCESSED.value))

    # Schema validation
    suite.add_expectation(
        ExpectTableColumnsToMatchSet(
            column_set=PROCESSED_DATA_COLUMNS,
            exact_match=False,
        )
    )

    # Null checks
    for col in PROCESSED_REQUIRED_NOT_NULL:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    # Numeric range checks
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="amount_inr", min_value=0.0, max_value=None)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="credit_limit", min_value=0.0, max_value=None)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="card_age", min_value=0.0, max_value=None)
    )

    # Binary checks
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="is_fraud", value_set=[0, 1])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="is_international", value_set=[0, 1])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="is_recurring", value_set=[0, 1])
    )

    suite.save()
    print(f"Created suite '{suite.name}' with {len(suite.expectations)} expectations")
    return suite


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================
def validate_dataframe(
    df: pd.DataFrame,
    suite_type: SuiteType = SuiteType.PROCESSED,
) -> tuple[bool, list[str]]:
    """
    Validate a DataFrame against the specified suite.

    Args:
        df: DataFrame to validate
        suite_type: Which suite to use (RAW or PROCESSED)

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    context = gx.get_context(mode="file")

    # Data source components (unique per suite type)
    ds_name = f"{suite_type.value}_datasource"
    asset_name = f"{suite_type.value}_asset"
    batch_name = f"{suite_type.value}_batch"

    # DataSource
    if ds_name in context.data_sources.all():
        data_source = context.data_sources.get(ds_name)
    else:
        data_source = context.data_sources.add_pandas(name=ds_name)

    # Asset
    try:
        data_asset = data_source.get_asset(asset_name)
    except LookupError:
        data_asset = data_source.add_dataframe_asset(name=asset_name)

    # Batch Definition
    try:
        batch_def = data_asset.get_batch_definition(batch_name)
    except KeyError:
        batch_def = data_asset.add_batch_definition_whole_dataframe(batch_name)

    # Recreate suite (picks up code changes)
    suite_name = suite_type.value
    try:
        context.suites.delete(suite_name)
    except Exception:
        pass

    # Create appropriate suite
    if suite_type == SuiteType.RAW:
        suite = create_raw_data_suite(context)
    else:
        suite = create_processed_data_suite(context)

    # Validation definition
    val_name = f"{suite_name}_validation"
    try:
        context.validation_definitions.delete(val_name)
    except Exception:
        pass

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=val_name,
            data=batch_def,
            suite=suite,
        )
    )

    # Run validation
    results = validation_def.run(batch_parameters={"dataframe": df})

    # Collect errors
    errors = []
    for result in results.results:
        if not result.success:
            exp_type = result.expectation_config.type

            # Add column info if available
            col = result.expectation_config.kwargs.get("column", "")
            if col:
                error_msg = f"Failed: {exp_type} (column: {col})"
            else:
                error_msg = f"Failed: {exp_type}"

            if "partial_unexpected_list" in result.result:
                samples = result.result["partial_unexpected_list"][:3]
                error_msg += f" samples: {samples}"

            errors.append(error_msg)

    return results.success, errors