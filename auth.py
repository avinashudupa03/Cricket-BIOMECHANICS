"""Authentication for the Cricket Biomechanics AI web app.

Authentication is disabled - all routes are publicly accessible.
"""

from __future__ import annotations

import functools
import os
import secrets
from datetime import datetime

from flask import (
    Flask,
    redirect,
    request,
    session,
    url_for,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_KEY = "cbai_user"


def init_auth(app: Flask):
    """Initialise authentication on the Flask app.

    Sets the secret key from environment (if not already set).
    Authentication is disabled - all routes are accessible without login.
    """
    # Ensure a strong secret key for sessions
    if not app.secret_key or app.secret_key == "cricket-biomechanics-dev-secret":
        env_key = os.environ.get("FLASK_SECRET_KEY")
        if env_key:
            app.secret_key = env_key
        else:
            # In production, require explicit secret key
            if os.environ.get("FLASK_ENV") == "production":
                raise RuntimeError(
                    "FLASK_SECRET_KEY must be set in production. "
                    "Set it via environment variable."
                )
            # Generate a random key for this server instance (dev only)
            app.secret_key = secrets.token_hex(32)
            print("[auth] WARNING: FLASK_SECRET_KEY not set — using a random key.")
            print("[auth] Sessions will not survive a server restart.")

    # Auto-login middleware - sets a default user session on first request
    @app.before_request
    def auto_login():
        if SESSION_KEY not in session:
            session[SESSION_KEY] = "admin"
            session["auth_time"] = datetime.now().isoformat()
            session.permanent = True


# ---------------------------------------------------------------------------
# Decorator (no-op - authentication disabled)
# ---------------------------------------------------------------------------

def login_required(view):
    """Decorator that does nothing - authentication is disabled."""
    return view


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def is_authenticated() -> bool:
    """Always return True - authentication is disabled."""
    return True


def get_current_user() -> str | None:
    """Return the current username (always 'admin' since auth is disabled)."""
    return session.get(SESSION_KEY) or "admin"


def _set_session(username: str):
    """Set the authenticated session."""
    session[SESSION_KEY] = username
    session["auth_time"] = datetime.now().isoformat()
    session.permanent = True


def _clear_session():
    """Clear the authentication session."""
    session.pop(SESSION_KEY, None)
    session.pop("auth_time", None)
    session.pop("next_url", None)