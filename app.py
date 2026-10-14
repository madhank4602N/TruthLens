
import os
import re
import json
import hmac
import hashlib
import secrets
import sqlite3
from html import escape
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
from tavily import TavilyClient


# =========================================================
# TruthLens: Multi-user Fake News NLP Prototype
# =========================================================

st.set_page_config(
    page_title="TruthLens | News Checker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_pipeline.joblib"
DB_PATH = BASE_DIR / "truthlens.db"


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                news_text TEXT NOT NULL,
                prediction TEXT,
                scores_json TEXT,
                sources_json TEXT,
                searched_web INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        conn.commit()


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash passwords securely using PBKDF2."""
    if salt is None:
        salt = secrets.token_bytes(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310_000,
    )

    return f"pbkdf2_sha256${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_hex, expected_hex = stored_hash.split("$", 2)

        if algorithm != "pbkdf2_sha256":
            return False

        actual_hash = hash_password(
            password,
            bytes.fromhex(salt_hex),
        ).split("$", 2)[2]

        return hmac.compare_digest(actual_hash, expected_hex)

    except (ValueError, TypeError):
        return False


def create_user(name: str, email: str, password: str):
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (
                    name.strip(),
                    email.strip().lower(),
                    hash_password(password),
                ),
            )

            conn.commit()

            user = conn.execute(
                "SELECT id, name, email FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        st.session_state.user = dict(user)
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."


def authenticate_user(email: str, password: str):
    with get_connection() as conn:
        user = conn.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ? COLLATE NOCASE
            """,
            (email.strip(),),
        ).fetchone()

    if user and verify_password(password, user["password_hash"]):
        return {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        }

    return None


def save_history(
    user_id: int,
    news_text: str,
    prediction: str | None,
    scores: list[dict[str, Any]],
    sources: list[dict[str, str]],
    searched_web: bool,
):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO news_history
            (
                user_id,
                news_text,
                prediction,
                scores_json,
                sources_json,
                searched_web
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                news_text,
                prediction,
                json.dumps(scores),
                json.dumps(sources),
                int(searched_web),
            ),
        )
        conn.commit()


def get_user_history(user_id: int):
    # Each user can only view their own history.
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                id,
                news_text,
                prediction,
                scores_json,
                sources_json,
                searched_web,
                created_at
            FROM news_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def delete_history_item(item_id: int, user_id: int):
    # Only the owner of an entry can delete it.
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM news_history
            WHERE id = ? AND user_id = ?
            """,
            (item_id, user_id),
        )
        conn.commit()


init_db()


# =========================================================
# OWNER LOGIN AND ADMIN REVIEW
# =========================================================

def owner_credentials():
    """Read owner credentials from Streamlit Secrets, never hard-code them."""
    try:
        owner_email = str(st.secrets.get("OWNER_EMAIL", "")).strip()
        owner_password = str(st.secrets.get("OWNER_PASSWORD", ""))
    except Exception:
        owner_email = ""
        owner_password = ""
    return owner_email, owner_password


def authenticate_owner(email: str, password: str) -> bool:
    configured_email, configured_password = owner_credentials()
    if not configured_email or not configured_password:
        return False
    return (
        hmac.compare_digest(email.strip().lower(), configured_email.lower())
        and hmac.compare_digest(password, configured_password)
    )


def render_owner_login():
    st.title("🔐 Owner Login")
    st.write("Sign in to review account activity and submitted news claims.")
    configured_email, configured_password = owner_credentials()
    if not configured_email or not configured_password:
        st.error(
            "Owner login is not configured. Add OWNER_EMAIL and OWNER_PASSWORD "
            "to Streamlit Secrets before using this page."
        )
        return

    with st.form("owner_login_form"):
        email = st.text_input("Owner email")
        password = st.text_input("Owner password", type="password")
        submitted = st.form_submit_button("Owner Login", type="primary",
                                          use_container_width=True)
    if submitted:
        if authenticate_owner(email, password):
            st.session_state.owner_authenticated = True
            st.session_state.user = None
            st.session_state.page = "Owner Dashboard"
            st.rerun()
        else:
            st.error("Incorrect owner email or password.")


def render_owner_dashboard():
    st.title("🛡️ Owner Dashboard")
    st.caption(
        "Review submitted claims and the app's model/search outputs. "
        "These outputs do not automatically establish whether a claim is true."
    )

    with get_connection() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        analysis_count = conn.execute(
            "SELECT COUNT(*) FROM news_history"
        ).fetchone()[0]
        rows = conn.execute("""
            SELECT
                news_history.id AS analysis_id,
                users.name AS user_name,
                users.email AS user_email,
                news_history.news_text AS news_text,
                news_history.prediction AS prediction,
                news_history.sources_json AS sources_json,
                news_history.searched_web AS searched_web,
                news_history.created_at AS created_at
            FROM news_history
            JOIN users ON users.id = news_history.user_id
            ORDER BY news_history.id DESC
        """).fetchall()
        users = conn.execute("""
            SELECT id, name, email, created_at
            FROM users
            ORDER BY id DESC
        """).fetchall()

    col1, col2 = st.columns(2)
    col1.metric("Registered users", user_count)
    col2.metric("Saved analyses", analysis_count)

    st.subheader("Registered users")
    if users:
        st.dataframe(
            [{"Name": r["name"], "Email": r["email"],
              "Joined (UTC)": r["created_at"]} for r in users],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No registered users yet.")

    st.subheader("User search and analysis review")
    search_filter = st.text_input(
        "Filter by email or claim text",
        placeholder="Enter an email address or a phrase from a claim",
    ).strip().lower()

    visible_rows = []
    for r in rows:
        if search_filter and (
            search_filter not in str(r["user_email"]).lower()
            and search_filter not in str(r["news_text"]).lower()
        ):
            continue
        try:
            sources = json.loads(r["sources_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            sources = []
        visible_rows.append({
            "Analysis ID": r["analysis_id"],
            "Date (UTC)": r["created_at"],
            "User": r["user_name"],
            "Email": r["user_email"],
            "Submitted claim": r["news_text"],
            "Model prediction": r["prediction"] or "Unavailable",
            "Web search enabled": bool(r["searched_web"]),
            "Sources found": len(sources),
        })

    if visible_rows:
        st.dataframe(visible_rows, use_container_width=True, hide_index=True)
        csv_data = pd.DataFrame(visible_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered review as CSV",
            data=csv_data,
            file_name="truthlens_owner_review.csv",
            mime="text/csv",
        )
    else:
        st.info("No analyses match this filter.")

    st.subheader("Review details")
    if rows:
        # Use the aliases defined in the query above consistently.
        row_options = {
            f"#{r['analysis_id']} · {r['created_at']} · {r['user_email']}": r["analysis_id"]
            for r in rows
            if not search_filter
            or search_filter in str(r["user_email"]).lower()
            or search_filter in str(r["news_text"]).lower()
        }
        if row_options:
            selected_id = st.selectbox("Choose an analysis", list(row_options.keys()))
            selected = next(
                r for r in rows if r["analysis_id"] == row_options[selected_id]
            )
            st.markdown("**Submitted claim**")
            st.write(selected["news_text"])
            st.write(f"**Model prediction:** {selected['prediction'] or 'Unavailable'}")
            try:
                sources = json.loads(selected["sources_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                sources = []
            st.markdown("**Retrieved sources**")
            if sources:
                for source in sources:
                    title = escape(str(source.get("title", "Source")))
                    url = str(source.get("url", ""))
                    if looks_like_url(url):
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.write(f"- {title}")
                    if source.get("content"):
                        st.caption(str(source["content"]))
            else:
                st.info("No sources were saved for this analysis.")
            st.warning(
                "Review the original sources and evidence yourself. "
                "The model label and search results are not proof of truth."
            )
    else:
        st.info("No saved analyses are available yet.")


# =========================================================
# MODEL AND LIVE SEARCH
# =========================================================

@st.cache_resource(show_spinner="Loading the NLP model...")
def load_model(model_path: str, modified_time: float):
    return joblib.load(model_path)


@st.cache_resource
def get_tavily_client(key: str):
    return TavilyClient(api_key=key)


def get_api_key():
    key = os.getenv("TAVILY_API_KEY", "").strip()

    try:
        key = (st.secrets.get("TAVILY_API_KEY", key) or key).strip()
    except Exception:
        pass

    return key


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value.strip())

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )


def do_analysis(
    claim: str,
    search_enabled: bool,
    number_of_sources: int,
):
    prediction = None
    scores: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    pipeline = None

    # Load trained NLP model.
    if MODEL_PATH.exists():
        try:
            pipeline = load_model(
                str(MODEL_PATH),
                MODEL_PATH.stat().st_mtime,
            )
        except Exception as exc:
            st.error(f"Could not load the saved model: {exc}")
    else:
        st.warning(
            "The trained model was not found. "
            "Train it using `python train_model.py` after "
            "adding `data/news.csv`."
        )

    # Predict using the NLP model.
    if pipeline is not None:
        try:
            prediction = str(pipeline.predict([claim])[0]).strip()

            if hasattr(pipeline, "predict_proba"):
                try:
                    probabilities = pipeline.predict_proba([claim])[0]
                    classes = list(pipeline.classes_)

                    scores = [
                        {
                            "label": str(label),
                            "score": float(score),
                        }
                        for label, score in sorted(
                            zip(classes, probabilities),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]

                except Exception:
                    scores = []

        except Exception as exc:
            st.error(f"Model prediction failed: {exc}")

    # Search the live web using Tavily.
    if search_enabled:
        api_key = get_api_key()

        if not api_key:
            st.warning(
                "Tavily API key is missing. Set TAVILY_API_KEY "
                "in your PowerShell terminal before launching Streamlit."
            )
        else:
            query = (
                claim
                if looks_like_url(claim)
                else " ".join(claim.split())[:700]
            )

            try:
                client = cast(Any, get_tavily_client(api_key))

                with st.spinner("Searching the web for related sources..."):
                    response = cast(
                        dict[str, Any],
                        client.search(
                            query=query,
                            topic="news",
                            search_depth="basic",
                            max_results=int(number_of_sources),
                            include_answer=False,
                        ),
                    )

                for result in response.get("results", []):
                    sources.append(
                        {
                            "title": str(
                                result.get("title") or "Untitled source"
                            ),
                            "url": str(result.get("url") or ""),
                            "content": str(result.get("content") or ""),
                        }
                    )

            except Exception as exc:
                st.error(f"Web search failed: {exc}")
                st.caption(
                    "Check your Tavily API key, internet connection, "
                    "and account status."
                )

    return prediction, scores, sources


# =========================================================
# SESSION STATE
# =========================================================

if "owner_authenticated" not in st.session_state:
    st.session_state.owner_authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "Login"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "reset_navigation" not in st.session_state:
    st.session_state.reset_navigation = False


# =========================================================
# LOGOUT CONFIRMATION DIALOG
# =========================================================

@st.dialog("Confirm Logout")
def confirm_logout():
    st.warning("Are you sure you want to log out?")

    st.caption(
        "You will need to log in again to access your account."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.reset_navigation = True
            st.session_state.page = "My Dashboard"
            st.rerun()

    with col2:
        if st.button(
            "Yes, Logout",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.user = None
            st.session_state.analysis_result = None
            st.session_state.page = "Login"
            st.session_state.reset_navigation = False
            st.rerun()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.title("📰 TruthLens")
    st.caption("NLP • Live Search • Source Review")
    st.divider()

    if st.session_state.owner_authenticated:
        st.success("Signed in as **Owner**")
        owner_menu = st.radio(
            "Owner Navigation",
            ["Owner Dashboard", "Owner Logout"],
            key="owner_navigation",
        )
        if owner_menu == "Owner Logout":
            st.session_state.owner_authenticated = False
            st.session_state.page = "Login"
            st.rerun()
        else:
            st.session_state.page = "Owner Dashboard"

    elif st.session_state.user:
        user = st.session_state.user
        st.success(f"Signed in as **{user['name']}**")
        if st.session_state.pop("reset_navigation", False):
            st.session_state["logged_in_navigation"] = "My Dashboard"
        menu = st.radio(
            "Navigation",
            ["My Dashboard", "Analyze News", "My History", "My Profile", "Logout"],
            key="logged_in_navigation",
        )
        if menu == "Logout":
            confirm_logout()
        else:
            st.session_state.page = menu

    else:
        menu = st.radio(
            "Welcome",
            ["Login", "Create Account", "Owner Login"],
            key="logged_out_navigation",
        )
        st.session_state.page = menu

    st.divider()
    st.markdown("### About TruthLens")
    st.write(
        "The NLP model finds learned text patterns; "
        "live search finds related sources."
    )
    st.caption(
        "Educational prototype. Model predictions and search "
        "relevance do not prove truth."
    )

# =========================================================
# LOGIN PAGE
# =========================================================

def render_login():
    st.title("🔐 Login to TruthLens")

    st.write(
        "Sign in to access your personal dashboard "
        "and saved analysis history."
    )

    with st.form("login_form"):
        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
        )

        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not email.strip() or not password:
            st.error("Enter both your email and password.")

        else:
            user = authenticate_user(email, password)

            if user:
                st.session_state.user = user
                st.session_state.page = "My Dashboard"
                st.session_state.analysis_result = None

                st.rerun()

            else:
                st.error("Incorrect email or password.")

    st.info(
        "New to TruthLens? Choose **Create Account** "
        "in the left sidebar."
    )


# =========================================================
# SIGN-UP PAGE
# =========================================================

def render_signup():
    st.title("✨ Create your TruthLens account")

    st.write(
        "Create an account to save your own news-analysis history."
    )

    with st.form("signup_form"):
        name = st.text_input("Full name")

        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
        )

        password = st.text_input(
            "Password",
            type="password",
            help="Use at least 8 characters.",
        )

        confirm = st.text_input(
            "Confirm password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Create account",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        clean_name = name.strip()
        clean_email = email.strip()

        if not clean_name or not clean_email or not password or not confirm:
            st.error("Please complete every field.")

        elif not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            clean_email,
        ):
            st.error("Enter a valid email address.")

        elif len(password) < 8:
            st.error("Use a password with at least 8 characters.")

        elif password != confirm:
            st.error("The passwords do not match.")

        else:
            ok, message = create_user(
                clean_name,
                clean_email,
                password,
            )

            if ok:
                st.success(message)
                st.session_state.page = "My Dashboard"
                st.session_state.analysis_result = None

                st.rerun()

            else:
                st.error(message)


# =========================================================
# DASHBOARD
# =========================================================

def render_dashboard():
    user = st.session_state.user

    st.title(f"Welcome, {user['name']} 👋")

    st.write("This is your personal TruthLens workspace.")

    rows = get_user_history(user["id"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Your analyses", len(rows))
    col2.metric("Account", "Active")
    col3.metric("Available tools", "NLP + Web")

    st.divider()

    st.subheader("Get started")

    st.write(
        "Go to **Analyze News** in the left sidebar "
        "to check a claim or headline."
    )

    st.write(
        "Open **My History** to review previous analyses "
        "saved to your account."
    )

    st.warning(
        "A model prediction is not a fact-check verdict. "
        "Review original sources and dates before deciding."
    )


# =========================================================
# NEWS ANALYSIS PAGE
# =========================================================

def render_analyze():
    st.title("🔎 Fake News Detection & Verification")

    st.markdown(
        "Investigate a headline using the trained text "
        "classifier and live web search."
    )

    st.info(
        "A model label is only a prediction based on training data. "
        "It does not establish whether a claim is factually true."
    )

    left, right = st.columns([2.2, 1], gap="large")

    with left:
        news_text = st.text_area(
            "News headline, claim, article, or URL",
            height=190,
            placeholder="Paste a specific news claim here...",
            key="analysis_news_text",
        )

    with right:
        st.markdown("#### Search settings")

        number_of_sources = st.select_slider(
            "Sources to retrieve",
            options=[3, 5, 7, 10],
            value=5,
        )

        search_enabled = st.checkbox(
            "Search the live web",
            value=True,
        )

    if st.button(
        "🚀 Analyse news",
        type="primary",
        use_container_width=True,
    ):
        claim = news_text.strip()

        if not claim:
            st.error("Please enter a headline or claim first.")
            return

        with st.spinner("Analysing your claim..."):
            prediction, scores, sources = do_analysis(
                claim,
                search_enabled,
                number_of_sources,
            )

        save_history(
            st.session_state.user["id"],
            claim,
            prediction,
            scores,
            sources,
            search_enabled,
        )

        st.session_state.analysis_result = {
            "claim": claim,
            "prediction": prediction,
            "scores": scores,
            "sources": sources,
            "search_enabled": search_enabled,
        }

        st.success(
            "Analysis completed and saved to your personal history."
        )

    result = st.session_state.analysis_result

    if result:
        st.divider()
        st.subheader("Analysis results")

        col1, col2 = st.columns([1, 1.4], gap="large")

        with col1:
            st.markdown("#### 🤖 NLP model result")

            prediction = result["prediction"]

            if prediction is None:
                st.warning(
                    "NLP prediction is unavailable because "
                    "the trained model could not be loaded."
                )

            elif prediction.lower() in {"fake", "false", "1"}:
                st.error(f"Prediction: {prediction.upper()}")

            elif prediction.lower() in {"real", "true", "0"}:
                st.success(f"Prediction: {prediction.upper()}")

            else:
                st.warning(f"Predicted label: {prediction}")

            st.caption(
                "This is a learned text pattern, not a fact-check verdict."
            )

        with col2:
            st.markdown("#### Model scores")

            if result["scores"]:
                for item in result["scores"]:
                    st.write(
                        f"**{item['label']}** — {item['score']:.1%}"
                    )

                    st.progress(
                        max(0.0, min(1.0, item["score"]))
                    )

                st.caption(
                    "Scores are class estimates, not the probability "
                    "that news is true."
                )

            else:
                st.caption("This model does not provide class scores.")

        st.markdown("#### 🌐 Live web search")

        if not result["search_enabled"]:
            st.info("Live web search was turned off.")

        elif result["sources"]:
            st.success(
                f"Found {len(result['sources'])} related search result(s)."
            )

            for index, source in enumerate(
                result["sources"],
                start=1,
            ):
                with st.container(border=True):
                    st.markdown(
                        f"**{index}. {escape(source['title'])}**"
                    )

                    source_url = source["url"]

                    # Only display valid HTTP/HTTPS source links.
                    if looks_like_url(source_url):
                        st.markdown(
                            f"[Read original source ↗]({source_url})"
                        )

                    if source["content"]:
                        st.markdown(
                            "<div style='font-size:15px;"
                            "line-height:1.6;overflow-wrap:anywhere;"
                            "white-space:pre-wrap;'>"
                            + escape(source["content"])
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        else:
            st.info(
                "No live sources were saved. Check your Tavily API key "
                "or try another claim."
            )

        st.divider()

        st.warning(
            "Review original links, dates, and evidence. Search "
            "relevance and model scores do not prove a story is "
            "true or false."
        )


# =========================================================
# PERSONAL HISTORY PAGE
# =========================================================

def render_history():
    user_id = st.session_state.user["id"]

    st.title("🕘 My History")

    st.write("Only your own saved analyses are shown here.")

    rows = get_user_history(user_id)

    if not rows:
        st.info(
            "You don't have any saved analyses yet. "
            "Use Analyze News to get started."
        )
        return

    st.caption(f"{len(rows)} saved analysis/analyses")

    for row in rows:
        prediction_label = (
            row["prediction"] or "No model prediction"
        ).upper()

        with st.expander(
            f"{row['created_at']} — {prediction_label}"
        ):
            st.markdown("**Submitted text**")
            st.write(row["news_text"])

            st.markdown("**Model scores**")

            try:
                saved_scores = json.loads(
                    row["scores_json"] or "[]"
                )
            except (json.JSONDecodeError, TypeError):
                saved_scores = []

            if saved_scores:
                for item in saved_scores:
                    st.write(
                        f"{item['label']}: "
                        f"{float(item['score']):.1%}"
                    )
            else:
                st.caption("No model scores were available.")

            st.markdown("**Sources**")

            try:
                saved_sources = json.loads(
                    row["sources_json"] or "[]"
                )
            except (json.JSONDecodeError, TypeError):
                saved_sources = []

            if saved_sources:
                for source in saved_sources:
                    title = escape(
                        str(source.get("title", "Source"))
                    )
                    source_url = str(source.get("url", ""))

                    if looks_like_url(source_url):
                        st.markdown(
                            f"- [{title}]({source_url})"
                        )
                    else:
                        st.write(f"- {title}")

            else:
                st.caption("No sources saved for this analysis.")

            if st.button(
                "Delete this history entry",
                key=f"delete_{row['id']}",
            ):
                delete_history_item(row["id"], user_id)
                st.success("History entry deleted.")
                st.rerun()


# =========================================================
# PROFILE PAGE
# =========================================================

def render_profile():
    user = st.session_state.user

    st.title("👤 My Profile")

    st.write("Your account details")

    st.text_input(
        "Full name",
        value=user["name"],
        disabled=True,
    )

    st.text_input(
        "Email address",
        value=user["email"],
        disabled=True,
    )

    st.caption(
        "This prototype does not yet provide profile editing "
        "or password reset."
    )


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.owner_authenticated:
    render_owner_dashboard()
elif not st.session_state.user:
    if st.session_state.page == "Owner Login":
        render_owner_login()
    elif st.session_state.page == "Create Account":
        render_signup()
    else:
        render_login()

else:
    page = st.session_state.page

    if page == "My Dashboard":
        render_dashboard()

    elif page == "Analyze News":
        render_analyze()

    elif page == "My History":
        render_history()

    elif page == "My Profile":
        render_profile()

    else:
        render_dashboard()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "TruthLens • BCA Internship Prototype • "
    "NLP classification + source review • "
    "Human verification required"
)
