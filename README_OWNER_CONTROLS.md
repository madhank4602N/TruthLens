# TruthLens owner controls

## Included changes
- One Login page for both the owner and regular users.
- Owner credentials are checked from Streamlit Secrets before regular-user credentials.
- Owner Dashboard includes a User access management section to enable/disable accounts.
- Existing SQLite databases are migrated automatically with an `is_active` column.
- Disabled accounts cannot start a new login session.

## Configure secrets
For local Streamlit, create `.streamlit/secrets.toml` beside `app.py`:

```toml
OWNER_EMAIL = "your-owner-email@example.com"
OWNER_PASSWORD = "your-new-unique-password"
TAVILY_API_KEY = "your-tavily-key-if-used"
```

Do not commit that file. For Streamlit Community Cloud, put the values in the app's Secrets settings instead.

## Apply
1. Back up your existing `app.py` and `truthlens.db`.
2. Replace `app.py` with the included one.
3. Restart/redeploy Streamlit.
4. Sign in from the normal Login page with owner credentials to open Owner Dashboard.
5. In User access management, select an account, toggle access, and click Save account access.

## Scope and privacy
These controls manage TruthLens accounts and app activity only. They do not give the owner access to a visitor's camera, microphone, files, or device. Any camera feature should be opt-in, browser-permission based, and limited to the specific feature the user chooses.

This patch passes Python syntax compilation; test the login and enable/disable flow in your own deployment before relying on it.
