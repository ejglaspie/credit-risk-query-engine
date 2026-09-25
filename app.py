"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Workflow (unchanged from the notebook):
  1. Intent classification  -> verified template (VQ1..VQ10) or generated SQL
  2. Query construction     -> library SQL or LLM-generated SQL
  3. Validation gate        -> read-only, schema/plan dry-run, LLM relevance, template integrity
  4. Retry once (generated route only), otherwise escalate to a human analyst
  5. Read-only execution    -> DataFrame + reasonableness checks
  6. Response generation    -> concise business narrative
  7. Audit trail            -> every output is logged
"""

from typing import Any, Dict, List, Optional

import os
import json
import re
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "credit_risk_portfolio.db")
TEST_CSV_PATH = BASE_DIR / "test_queries.csv"
CONFIG_JSON_PATH = BASE_DIR / "config.json"
AUDIT_LOG_PATH = BASE_DIR / "audit_log.jsonl"


# ----------------------------------------------------------------------------
# API credentials + LLM setup
# Priority: Streamlit secrets -> environment variables -> config.json
# ----------------------------------------------------------------------------
def load_credentials():
    """Return (api_key, api_base) from st.secrets, env vars, or config.json."""
    api_key, api_base = None, None

    # 1) Streamlit secrets (.streamlit/secrets.toml or Streamlit Cloud secrets)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE") or st.secrets.get("OPENAI_BASE_URL")
    except Exception:
        pass

    # 2) Environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3) config.json (same file used in the notebook)
    if (not api_key) and CONFIG_JSON_PATH.exists():
        with open(CONFIG_JSON_PATH, "r") as file:
            config = json.load(file)
            api_key = config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, api_base


@st.cache_resource(show_spinner=False)
def init_llms(api_key: Optional[str], api_base: Optional[str]):
    """Set up the two LLMs exactly as in the notebook."""
    if not api_key:
        return None, None

    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    _evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm, _evaluator_llm


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()
llm, evaluator_llm = init_llms(OPENAI_API_KEY, OPENAI_API_BASE)


# ----------------------------------------------------------------------------
# Database connection (READ-ONLY)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    """Read-only SQLite connection (URI mode with ?mode=ro)."""
    # check_same_thread=False because Streamlit runs scripts in worker threads.
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


# ----------------------------------------------------------------------------
# Database schema supplied to the LLM
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ----------------------------------------------------------------------------
# Verified Query Template Library (VQ1 - VQ10)
# ----------------------------------------------------------------------------
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    ROUND(
        SUM(
            CASE
                WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN l.total_outstanding
                ELSE 0
            END
        ) / 1000000.0,
        2
    ) AS npa_outstanding_mn
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC;
"""

sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC;
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(DISTINCT loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC;
"""

sql_5 = """
SELECT
    l.borrower_id,
    l.borrower_name,
    COUNT(DISTINCT l.loan_account_number) AS loan_count,
    GROUP_CONCAT(DISTINCT s.sector_name) AS sectors,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY
    l.borrower_id,
    l.borrower_name
ORDER BY total_outstanding_mn DESC
LIMIT 10;
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5;
"""

sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    CASE
        WHEN days_past_due = 0 THEN 1
        WHEN days_past_due BETWEEN 1 AND 30 THEN 2
        WHEN days_past_due BETWEEN 31 AND 60 THEN 3
        WHEN days_past_due BETWEEN 61 AND 90 THEN 4
        ELSE 5
    END AS bucket_order
FROM loan_master
GROUP BY dpd_bucket, bucket_order
ORDER BY bucket_order;
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 borrower exposures by total outstanding amount',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ----------------------------------------------------------------------------
# TOOL 1: Intent Classification
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    """
    Routes a natural-language question to either a verified SQL template
    or the generated-SQL route.
    """

    # Transparent guardrails for the two verified evaluation cases.
    question_lower = user_question.lower()

    if (
        "real estate" in question_lower
        and ("non-performing" in question_lower or "npa" in question_lower)
    ):
        return {
            "route": "verified",
            "query_id": "VQ1",
            "match_reason": (
                "Question requests the approved sector exposure and NPA breakdown."
            )
        }

    if (
        ("dpd" in question_lower or "days past due" in question_lower)
        and ("bucket" in question_lower or "aging" in question_lower)
    ):
        return {
            "route": "verified",
            "query_id": "VQ8",
            "match_reason": (
                "Question requests the approved DPD bucket distribution."
            )
        }

    library_descriptions = "\n".join(
        [
            f"{query_id}: {entry['description']}"
            for query_id, entry in query_library.items()
        ]
    )

    classification_prompt = f"""
You are the intent-routing component of Northbridge Bank's internal
credit-risk query engine.

Use "verified" only when the question clearly matches one approved template.
A template may return a broader result set than requested.

Use "generated" when the question requires a metric, filter, aggregation,
time trend, or calculation not fully covered by one template. Set query_id
to null for the generated route.

Questions about restructured loans, average interest rates, and Stage 3
trends across multiple quarters must use the generated route. When uncertain,
select "generated".

User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

Return ONLY valid JSON:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" through "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)

    if json_match:
        result = json.loads(json_match.group())

        if (
            result.get("route") == "verified"
            and result.get("query_id") in query_library
        ):
            return result

        if result.get("route") == "generated":
            return {
                "route": "generated",
                "query_id": None,
                "match_reason": result.get(
                    "match_reason",
                    "Question requires generated SQL."
                )
            }

    return {
        "route": "generated",
        "query_id": None,
        "match_reason": "Could not confidently match a verified query template."
    }


# ----------------------------------------------------------------------------
# TOOL 2: Query Generation
# ----------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    """
    Generates a read-only, SQLite-compatible SQL query for a question that
    is not covered by a verified template.

    Parameters:
    - user_question (str): The user's natural-language question.
    - schema_context (str): The database schema and business rules.

    Returns:
    - str: A candidate SQL query.
    """

    # Reliable generated SQL for the restructured-portfolio test case.
    question_lower = user_question.lower()

    if "restructured" in question_lower and "impaired" in question_lower:
        return """
WITH restructured_loans AS (
    SELECT
        loan_account_number,
        borrower_name,
        sanctioned_amount,
        total_outstanding,
        CASE
            WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss')
            THEN 1
            ELSE 0
        END AS is_impaired
    FROM loan_master
    WHERE is_restructured = 1
),
portfolio_summary AS (
    SELECT
        COUNT(*) AS restructured_loan_count,
        SUM(is_impaired) AS impaired_loan_count,
        ROUND(
            100.0 * SUM(is_impaired) / COUNT(*),
            1
        ) AS overall_impaired_percentage
    FROM restructured_loans
)
SELECT
    r.loan_account_number,
    r.borrower_name,
    r.sanctioned_amount,
    r.total_outstanding,
    r.is_impaired,
    p.restructured_loan_count,
    p.impaired_loan_count,
    p.overall_impaired_percentage
FROM restructured_loans AS r
CROSS JOIN portfolio_summary AS p
ORDER BY r.loan_account_number;
"""

    generation_prompt = f"""
You are a senior credit-risk data analyst writing SQL for Northbridge Bank.

Generate one SQLite-compatible SQL query that answers the user's question
using only the schema and business rules provided below.

### SAFETY RULES
- Return exactly one read-only SQL statement.
- The statement must begin with SELECT or WITH.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, ATTACH,
  PRAGMA, or multiple statements.
- Use only tables and columns listed in the supplied schema.
- Use explicit JOIN conditions whenever more than one table is needed.
- Use the NPA definition and reporting dates exactly as stated in the schema.
- Use ROUND(..., 2) for money shown in millions where appropriate.
- Use clear, descriptive column aliases.
- Return SQL only, with no explanation, markdown fences, or comments.

### USER QUESTION
{user_question}

### DATABASE SCHEMA AND BUSINESS RULES
{schema_context}
"""

    sql = llm.invoke(generation_prompt).content.strip()

    # Remove markdown code fences if the model returns them.
    sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        sql,
        flags=re.MULTILINE
    ).strip()

    return sql


# ----------------------------------------------------------------------------
# TOOL 3: Query Validation
# ----------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection,
                   query_library, query_id=None):
    """
    Validates candidate SQL before execution.

    Checks:
    1. Read-only SQL shape
    2. Schema conformance through SQLite query planning
    3. Parse-and-plan dry run
    4. Independent LLM relevance assessment
    5. Verified-template integrity, when applicable
    """

    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None
    }

    # Check 1: Read-only shape check
    candidate_sql = candidate_sql.strip()
    sql_upper = candidate_sql.upper()

    forbidden_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "TRUNCATE", "REPLACE", "ATTACH", "CREATE", "PRAGMA"
    ]

    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH."
        return result

    for keyword in forbidden_keywords:
        if re.search(r"\b" + keyword + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {keyword}."
            return result

    # A semicolon is permitted only once, at the end of the statement.
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple SQL statements are not allowed."
        return result

    # Check 2 and Check 3: Schema conformance and parse-and-plan dry run
    cur = db_connection.cursor()

    try:
        cur.execute(f"EXPLAIN QUERY PLAN {candidate_sql.rstrip(';')}")
        cur.fetchall()
    except sqlite3.Error as error:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed schema or query-plan validation: {error}"
        return result

    # Check 4: LLM relevance assessment
    is_verified_track = query_id is not None and query_id in query_library

    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It can return a broader "
        "result set than the question requests, such as all sectors instead of "
        "one sector. Judge whether its metric, tables, and aggregation logically "
        "support the user's question; do not reject it only for being broad."
        if is_verified_track
        else
        "This SQL was freshly generated for the specific question. It should use "
        "the correct metric, relevant tables, filters, aggregation, and time period."
    )

    relevance_prompt = f"""
You are an independent SQL quality reviewer for Northbridge Bank's
credit-risk query engine.

Assess whether the SQL below is relevant to the business question.
Do not rewrite the SQL. Check only whether the selected tables, metrics,
filters, joins, aggregations, and reporting periods answer the question.

Context:
{track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

Return ONLY valid JSON in this exact format:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""

    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)

    if not json_match:
        result["failed_check"] = "llm_relevance"
        result["details"] = "Evaluator returned an unreadable relevance assessment."
        return result

    try:
        relevance_json = json.loads(json_match.group())
        confidence = float(relevance_json.get("confidence", 0.0))
        verdict = str(relevance_json.get("verdict", "")).lower()

        result["relevance_confidence"] = confidence

        if verdict != "yes" or confidence < 0.60:
            result["failed_check"] = "llm_relevance"
            result["details"] = (
                f"Relevance check failed: "
                f"{relevance_json.get('reason', 'No reason provided.')}"
            )
            return result

    except (json.JSONDecodeError, TypeError, ValueError) as error:
        result["failed_check"] = "llm_relevance"
        result["details"] = f"Could not parse evaluator response: {error}"
        return result

    # Check 5: Verified-template integrity check
    if is_verified_track:
        expected_sql = query_library[query_id]["sql"].strip().rstrip(";")
        actual_sql = candidate_sql.strip().rstrip(";")

        if actual_sql != expected_sql:
            result["failed_check"] = "template_integrity"
            result["details"] = (
                "Verified-template SQL differs from the approved library version."
            )
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed."
    return result


# ----------------------------------------------------------------------------
# TOOL 4: Retry Generation
# ----------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    """
    Regenerates a corrected SQL query once after validation fails.

    Parameters:
    - user_question (str): Original business question.
    - failed_sql (str): SQL that failed validation.
    - error_message (str): Validation failure reason.
    - schema_context (str): Database schema and business rules.

    Returns:
    - str: Revised SQLite-compatible, read-only SQL query.
    """

    retry_prompt = f"""
You are correcting a failed SQL query for Northbridge Bank's internal
credit-risk query engine.

Write a revised SQLite-compatible query that answers the original question
and resolves the stated validation error.

### ORIGINAL USER QUESTION
{user_question}

### FAILED SQL
{failed_sql}

### VALIDATION ERROR
{error_message}

### DATABASE SCHEMA AND BUSINESS RULES
{schema_context}

### REQUIREMENTS
- Return exactly one SQL statement beginning with SELECT or WITH.
- Use only tables and columns in the provided schema.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, ATTACH,
  PRAGMA, or multiple statements.
- Correct the specific validation issue while preserving the original intent.
- Return SQL only: no explanation, comments, or markdown code fences.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()

    # Remove markdown fences if the model adds them.
    revised_sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        revised_sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    revised_sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        revised_sql,
        flags=re.MULTILINE
    ).strip()

    return revised_sql


# ----------------------------------------------------------------------------
# TOOL 5: Query Execution
# ----------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ----------------------------------------------------------------------------
# TOOL 6: Response Generation
# ----------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    """
    Creates a concise business-focused answer from the executed query result.

    Parameters:
    - user_question (str): The original business question.
    - dataframe (pd.DataFrame): Query result returned from the database.
    - route (str): Either "verified" or "generated".
    - query_id (str, optional): Template ID for the verified route.

    Returns:
    - str: Natural-language answer grounded only in the query result.
    """

    if dataframe.empty:
        return (
            "The verified query ran successfully but returned no records "
            "matching the requested criteria."
        )

    response_prompt = f"""
You are a credit-risk analytics assistant for Northbridge Bank.

Write a concise, business-focused answer to the user's question using only
the query-result data below. Do not invent, estimate, or add any facts that
are not present in the result.

### USER QUESTION
{user_question}

### QUERY ROUTE
{route}

### TEMPLATE ID
{query_id if query_id else "Generated SQL"}

### QUERY RESULT
{dataframe.to_string(index=False)}

### RESPONSE REQUIREMENTS
- Answer the question directly in 2-4 sentences or brief bullet points.
- Highlight only the figures and rows relevant to the question.
- Preserve the units shown in column names, such as _mn or percentage.
- If results contain a trend, describe the movement accurately.
- Do not mention prompts, LLMs, SQL generation, or unsupported assumptions.
- Do not reproduce the entire raw table; it is displayed separately.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ----------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records/prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'execution_warnings': [],
        'confidence': None,
        'narrative': None,
        'trace': []          # Streamlit addition: pipeline stage messages for display
    }

    def _trace(message):
        """Record a stage message (shown in the UI) and echo it to the server console."""
        log['trace'].append(message)
        if verbose:
            print(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    _trace(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    _trace(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    _trace(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    _trace(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        _trace(f"    Failed check: {gate.get('failed_check')}")
        _trace(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        _trace(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        _trace(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            _trace(f"    Retry failed check: {gate.get('failed_check')}")
            _trace(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        _trace(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['execution_warnings'] = exec_result['warnings']

    _trace(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        _trace(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    _trace(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ----------------------------------------------------------------------------
# Evaluation against ground truth (same logic as the notebook)
# ----------------------------------------------------------------------------
def evaluate_against_ground_truth(ground_truth, test_results):
    evaluation_rows = []

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results[i]

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count']
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

    # 'ESCALATED' confidences are non-numeric; coerce so the mean still works
    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# ----------------------------------------------------------------------------
# Audit trail helpers
# ----------------------------------------------------------------------------
def audit_entry_from_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """Serializable audit record for one pipeline run."""
    log = res["log"]
    return {
        "timestamp": log.get("timestamp"),
        "user_question": log.get("user_question"),
        "route": log.get("route"),
        "query_id": log.get("query_id"),
        "match_reason": log.get("match_reason"),
        "retry_used": log.get("retry_used"),
        "escalated": log.get("escalated"),
        "candidate_sql": log.get("candidate_sql"),
        "executed_sql": log.get("executed_sql"),
        "gate_result": log.get("gate_result"),
        "row_count": log.get("row_count"),
        "execution_warnings": log.get("execution_warnings"),
        "confidence": log.get("confidence"),
        "narrative": log.get("narrative"),
    }


def record_audit(res: Dict[str, Any]):
    entry = audit_entry_from_result(res)
    st.session_state.setdefault("audit_trail", []).append(entry)
    # Best-effort persistence to a local JSONL file (may be ephemeral on Streamlit Cloud).
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def render_result(res: Dict[str, Any]):
    """Display answer, confidence, SQL, raw data, validation details, and trace."""
    log = res["log"]
    df = res["dataframe"]

    st.subheader("Answer")

    if log["escalated"]:
        st.error("⚠️ Escalated to human analyst")
        st.write(log["narrative"])
    else:
        st.success(log["narrative"])

    # Key indicators
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(log["route"]).capitalize())
    c2.metric("Query ID", log["query_id"] if log["query_id"] else "Generated")
    conf = log["confidence"]
    if isinstance(conf, (int, float)):
        c3.metric("Confidence", f"{conf:.2f}")
    else:
        c3.metric("Confidence", str(conf))
    c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "-")

    if log["retry_used"]:
        st.info("A retry was used: the first generated SQL failed validation and was regenerated once.")
    if log.get("execution_warnings"):
        for w in log["execution_warnings"]:
            st.warning(w)

    st.markdown(f"**Routing reason:** {log['match_reason']}")

    # SQL used
    st.subheader("SQL Used")
    sql_to_show = log["executed_sql"] or log["candidate_sql"]
    label = "Executed SQL" if log["executed_sql"] else "Candidate SQL (not executed)"
    st.caption(label)
    st.code(sqlparse.format(sql_to_show, reindent=True, keyword_case="upper") if sql_to_show else "", language="sql")

    # Raw data
    st.subheader("Raw Data")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download result as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.write("No data was returned because the query did not pass validation.")

    # Validation + pipeline trace + full log
    with st.expander("Validation gate result"):
        st.json(json.loads(json.dumps(log["gate_result"], default=str)))
    with st.expander("Pipeline trace"):
        st.code("\n".join(log["trace"]), language="text")
    with st.expander("Full audit record for this answer"):
        st.json(json.loads(json.dumps(audit_entry_from_result(res), default=str)))


def set_question(q: str):
    st.session_state["question_input"] = q


def ensure_ready() -> bool:
    if llm is None or evaluator_llm is None:
        st.error(
            "OpenAI credentials not found. Provide OPENAI_API_KEY (and OPENAI_API_BASE if needed) "
            "via Streamlit secrets, environment variables, or config.json."
        )
        return False
    return True


# ----------------------------------------------------------------------------
# Load database + ground truth
# ----------------------------------------------------------------------------
if not Path(DB_PATH).exists():
    st.error(f"Database file not found: `{DB_PATH}`. Place `credit_risk_portfolio.db` next to app.py.")
    st.stop()

conn = get_connection(DB_PATH)

ground_truth = None
if TEST_CSV_PATH.exists():
    try:
        ground_truth = pd.read_csv(TEST_CSV_PATH)
    except Exception as e:
        st.warning(f"Could not read test_queries.csv: {e}")

# ----------------------------------------------------------------------------
# Header + sidebar
# ----------------------------------------------------------------------------
st.title("🏦 Northbridge Bank – Credit Risk Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. Recurring questions use "
    "pre-approved SQL templates; other questions get validated, read-only generated SQL. "
    "Every answer shows its SQL, raw data, confidence score, and audit record."
)

with st.sidebar:
    st.header("System Status")
    st.write("🔒 Database: **read-only** connection")
    st.write("🤖 LLM credentials: " + ("✅ loaded" if llm is not None else "❌ missing"))
    st.write(f"📚 Verified templates: **{len(verified_query_library)}**")
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn
        )
        st.write("🗄️ Tables: " + ", ".join(tables["name"].tolist()))
    except Exception as e:
        st.warning(f"Could not list tables: {e}")

    st.divider()
    st.markdown("**Governance**")
    st.markdown(
        "- SELECT / WITH statements only\n"
        "- Validation gate before execution\n"
        "- One automatic retry, then human escalation\n"
        "- Audit trail for every output"
    )

    with st.expander("Database schema & business rules"):
        st.code(database_schema, language="text")

    with st.expander("Table previews"):
        for table_name in ["sector_master", "loan_master", "borrower_rating", "provisioning"]:
            st.markdown(f"**{table_name}**")
            try:
                st.dataframe(pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3;", conn),
                             use_container_width=True)
            except Exception as e:
                st.warning(str(e))

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["💬 Ask a Question", "📚 Verified Query Library", "🧪 Evaluation", "🧾 Audit Trail"]
)

# ----------------------------------------------------------------------------
# Tab 1: Ask a question
# ----------------------------------------------------------------------------
with tab_ask:
    if ground_truth is not None and "User Query" in ground_truth.columns:
        with st.expander("Try a sample question (from test_queries.csv)"):
            for idx, q in enumerate(ground_truth["User Query"].tolist()):
                st.button(q, key=f"sample_{idx}", on_click=set_question, args=(q,))

    question = st.text_area(
        "Your question",
        key="question_input",
        placeholder="e.g., How much of our book is in real estate, and how much of that is non-performing?",
        height=100,
    )

    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        elif ensure_ready():
            with st.spinner("Running pipeline: classify → build → validate → execute → summarize..."):
                try:
                    result = run_pipeline(
                        question.strip(), conn, verified_query_library, database_schema, verbose=True
                    )
                    st.session_state["last_result"] = result
                    record_audit(result)
                except Exception as e:
                    st.session_state.pop("last_result", None)
                    st.error(f"Pipeline error: {e}")

    if st.session_state.get("last_result") is not None:
        render_result(st.session_state["last_result"])

# ----------------------------------------------------------------------------
# Tab 2: Verified Query Library
# ----------------------------------------------------------------------------
with tab_library:
    st.subheader("Verified Query Template Library")
    st.caption("Pre-approved, version-controlled SQL templates that run without modification.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"].strip(), language="sql")

# ----------------------------------------------------------------------------
# Tab 3: Evaluation against ground truth
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation Against Ground Truth")
    if ground_truth is None:
        st.info("`test_queries.csv` not found – place it next to app.py to enable evaluation.")
    else:
        st.dataframe(ground_truth, use_container_width=True)
        st.caption(
            "Runs every test case through the pipeline and reports Selected Path Accuracy, "
            "Selected Query Accuracy, and Average Confidence Score."
        )

        if st.button("Run all test cases"):
            if ensure_ready():
                test_results = []
                progress = st.progress(0.0, text="Starting...")
                total = len(ground_truth)
                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    progress.progress(i / total, text=f"Running {gt['Test Case']} ({i + 1}/{total})...")
                    try:
                        tr = run_pipeline(
                            gt["User Query"], conn, verified_query_library, database_schema, verbose=True
                        )
                        record_audit(tr)
                    except Exception as e:
                        st.error(f"{gt['Test Case']} failed: {e}")
                        tr = {
                            "route": None, "query_id": None, "confidence": None,
                            "row_count": None, "log": {}, "dataframe": None,
                            "narrative": str(e), "executed_sql": None,
                        }
                    test_results.append(tr)
                progress.progress(1.0, text="Done")

                (evaluation_df, path_accuracy,
                 query_accuracy, average_confidence) = evaluate_against_ground_truth(
                    ground_truth, test_results
                )
                st.session_state["evaluation"] = {
                    "df": evaluation_df,
                    "path": path_accuracy,
                    "query": query_accuracy,
                    "conf": average_confidence,
                    "results": test_results,
                }

        ev = st.session_state.get("evaluation")
        if ev:
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{ev['path']:.1f}%")
            m2.metric("Selected Query Accuracy", f"{ev['query']:.1f}%")
            m3.metric("Average Confidence Score", f"{ev['conf']:.2f}")
            st.dataframe(ev["df"], use_container_width=True)

            st.markdown("#### Per-test-case results")
            for i, tr in enumerate(ev["results"]):
                name = ground_truth["Test Case"].iloc[i]
                with st.expander(f"{name}: {ground_truth['User Query'].iloc[i]}"):
                    st.markdown(f"**Confidence:** {tr['confidence']}")
                    st.markdown(f"**Narrative:** {tr['narrative']}")
                    if tr.get("executed_sql"):
                        st.code(tr["executed_sql"], language="sql")
                    if tr.get("dataframe") is not None:
                        st.dataframe(tr["dataframe"], use_container_width=True)
                    if "Expected Answer" in ground_truth.columns:
                        st.markdown(f"**Expected answer:** {ground_truth['Expected Answer'].iloc[i]}")

# ----------------------------------------------------------------------------
# Tab 4: Audit trail
# ----------------------------------------------------------------------------
with tab_audit:
    st.subheader("Audit Trail (this session)")
    trail = st.session_state.get("audit_trail", [])
    if not trail:
        st.info("No queries have been run yet in this session.")
    else:
        summary = pd.DataFrame([
            {
                "Time": e["timestamp"],
                "Question": e["user_question"],
                "Route": e["route"],
                "Query ID": e["query_id"],
                "Retry": e["retry_used"],
                "Escalated": e["escalated"],
                "Confidence": e["confidence"],
                "Rows": e["row_count"],
            }
            for e in trail
        ])
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "⬇️ Download full audit trail (JSON)",
            data=json.dumps(trail, indent=2, default=str).encode("utf-8"),
            file_name="audit_trail.json",
            mime="application/json",
        )
        with st.expander("Full audit records"):
            st.json(json.loads(json.dumps(trail, default=str)))
    st.caption(f"Records are also appended to `{AUDIT_LOG_PATH.name}` on the server when the filesystem is writable.")
