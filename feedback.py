"""Sign-up + feature-request form, saved to a private Google Sheet.

Streamlit Community Cloud has no persistent disk, so submissions go to a
Google Sheet the owner controls. Configuration (Streamlit secrets):

    FEEDBACK_SHEET_URL = "https://docs.google.com/spreadsheets/d/<id>/edit"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "signalbridge-form@<project>.iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"

Share the sheet with the service account's client_email as an Editor.

Each submission appends one row:
    timestamp_utc | email | role | organization_type | wants_to_see |
    consent_updates | source | questions_asked_this_session

Values are written with value_input_option="RAW", so a visitor typing
"=FORMULA(...)" is stored as plain text, never evaluated.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

import streamlit as st

HEADER = [
    "timestamp_utc",
    "email",
    "role",
    "organization_type",
    "wants_to_see",
    "consent_updates",
    "source",
    "questions_asked_this_session",
]

ROLES = (
    "Select…",
    "PV scientist / case processor",
    "Safety physician / medical reviewer",
    "Signal management lead",
    "QPPV / PV head",
    "Regulatory affairs",
    "Data scientist / engineer",
    "Academic / researcher",
    "Student",
    "Other",
)

ORG_TYPES = (
    "Select…",
    "Pharma / biotech",
    "CRO / PV service provider",
    "Consultancy",
    "Regulator / public health",
    "Hospital / health system",
    "Academia",
    "Tech / software vendor",
    "Independent",
    "Other",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MAX_TEXT = 2000


def _secret(name: str) -> Any:
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:
        return None


def is_configured() -> bool:
    return bool(_secret("FEEDBACK_SHEET_URL") and _secret("gcp_service_account"))


@st.cache_resource(show_spinner=False)
def _worksheet() -> Any:
    """Open (and initialise) the first worksheet of the feedback sheet."""
    import gspread  # imported lazily so the app runs without it configured

    creds = dict(_secret("gcp_service_account"))
    client = gspread.service_account_from_dict(creds)
    ws = client.open_by_url(_secret("FEEDBACK_SHEET_URL")).sheet1
    if ws.row_values(1) != HEADER:
        if not any(ws.row_values(1)):
            ws.update([HEADER], "A1", value_input_option="RAW")
    return ws


def save_submission(row: dict[str, Any]) -> None:
    """Append one submission. Raises on failure so the UI can say so."""
    ws = _worksheet()
    ws.append_row([row.get(h, "") for h in HEADER], value_input_option="RAW")


def validate(email: str, role: str, org: str, wants: str, consent: bool) -> list[str]:
    errors = []
    if not _EMAIL_RE.match(email.strip()):
        errors.append("Enter a valid email address.")
    if role == ROLES[0]:
        errors.append("Pick your role.")
    if org == ORG_TYPES[0]:
        errors.append("Pick your organization type.")
    if len(wants) > MAX_TEXT:
        errors.append(f"Keep the request under {MAX_TEXT} characters.")
    if not consent:
        errors.append("Tick the box so we're allowed to email you.")
    return errors


def render_form(source: str, questions_asked: int = 0) -> None:
    """Render the sign-up form. `source` must be unique per placement."""
    if st.session_state.get("feedback_submitted"):
        st.success("Thanks - you're on the list. We'll be in touch.")
        return

    with st.form(f"feedback_{source}", clear_on_submit=False, border=False):
        email = st.text_input("Work email", placeholder="you@company.com", key=f"fb_email_{source}")
        c1, c2 = st.columns(2)
        role = c1.selectbox("Your role", ROLES, key=f"fb_role_{source}")
        org = c2.selectbox("Organization type", ORG_TYPES, key=f"fb_org_{source}")
        wants = st.text_area(
            "What would you like SignalBridge to do next?",
            placeholder="e.g. PRR / BCPNN, compare two drugs, export to Excel, EudraVigilance data…",
            max_chars=MAX_TEXT,
            key=f"fb_wants_{source}",
        )
        consent = st.checkbox(
            "Email me about SignalBridge updates and early access. I can unsubscribe any time.",
            key=f"fb_consent_{source}",
        )
        submitted = st.form_submit_button("Keep me posted", use_container_width=True)

    st.caption(
        "We store only what you type here, in a private sheet, and use it only "
        "to contact you about SignalBridge. We never sell or share it."
    )

    if not submitted:
        return

    errors = validate(email, role, org, wants, consent)
    if errors:
        for e in errors:
            st.warning(e)
        return

    if not is_configured():
        st.error("Sign-ups aren't switched on yet on this server. Please try again later.")
        return

    try:
        save_submission(
            {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "email": email.strip().lower(),
                "role": role,
                "organization_type": org,
                "wants_to_see": wants.strip(),
                "consent_updates": "yes",
                "source": source,
                "questions_asked_this_session": questions_asked,
            }
        )
    except Exception:
        st.error("Sorry - we couldn't save that just now. Please try again in a minute.")
        return

    st.session_state.feedback_submitted = True
    st.rerun()
