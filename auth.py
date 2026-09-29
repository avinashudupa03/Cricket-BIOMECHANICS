"""Authentication for the Cricket Biomechanics AI web app.

Provides session-based authentication with a single admin account.
The password is read from the ``CBAI_ADMIN_PASSWORD`` environment variable;
if not set, a default dev password is used (with a printed warning).

Usage:
    from auth import login_required, init_auth
    init_auth(app)  # call once after creating the Flask app

    @app.route("/protected")
    @login_required
    def protected():
        return "secret"

Routes added:
    GET  /login          - login form
    POST /login          - authenticate
    GET  /logout         - log out
"""

from __future__ import annotations

import functools
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DEV_PASSWORD = "cricket-admin"
SESSION_KEY = "cbai_user"
SESSION_EXPIRY_HOURS = 12

# Password from environment, or default dev password
ADMIN_PASSWORD = os.environ.get("CBAI_ADMIN_PASSWORD", DEFAULT_DEV_PASSWORD)
IS_DEV_PASSWORD = ADMIN_PASSWORD == DEFAULT_DEV_PASSWORD


def init_auth(app: Flask):
    """Initialise authentication on the Flask app.

    Sets the secret key from environment (if not already set) and registers
    the login/logout routes.
    """
    # Ensure a strong secret key for sessions
    if not app.secret_key or app.secret_key == "cricket-biomechanics-dev-secret":
        env_key = os.environ.get("FLASK_SECRET_KEY")
        if env_key:
            app.secret_key = env_key
        else:
            # Generate a random key for this server instance
            app.secret_key = secrets.token_hex(32)
            print("[auth] WARNING: FLASK_SECRET_KEY not set — using a random key.")
            print("[auth] Sessions will not survive a server restart.")

    if IS_DEV_PASSWORD:
        print("[auth] WARNING: Using default admin password.")
        print("[auth] Set CBAI_ADMIN_PASSWORD environment variable in production.")

    # Register login/logout routes
    app.add_url_rule("/login", "login_view", login_view, methods=["GET", "POST"])
    app.add_url_rule("/logout", "logout_view", logout_view)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def login_required(view):
    """Decorator that redirects to the login page if the user is not authenticated."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not is_authenticated():
            # Save the URL the user was trying to reach
            session["next_url"] = request.url
            return redirect(url_for("login_view"))
        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def is_authenticated() -> bool:
    """Return True if the current session is authenticated."""
    return session.get(SESSION_KEY) is not None


def get_current_user() -> str | None:
    """Return the current username, or None if not authenticated."""
    return session.get(SESSION_KEY)


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


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def login_view():
    """Handle GET (show form) and POST (authenticate) for /login."""
    if request.method == "POST":
        password = request.form.get("password", "")
        if _check_password(password):
            _set_username("admin")
            # Redirect to the originally requested URL, or dashboard
            next_url = session.pop("next_url", None)
            if next_url:
                return redirect(next_url)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid password.", "error")
            return render_template("login.html"), 401

    # GET: show login form
    if is_authenticated():
        return redirect(url_for("dashboard"))
    return render_template("login.html")


def logout_view():
    """Log out the current user."""
    _clear_session()
    flash("You have been logged out.", "info")
    return redirect(url_for("login_view"))


# ---------------------------------------------------------------------------
# Password verification
# ---------------------------------------------------------------------------

def _check_password(password: str) -> bool:
    """Constant-time password comparison to prevent timing attacks."""
    if not password:
        return False
    return secrets.compare_digest(password, ADMIN_PASSWORD)


def _set_username(username: str):
    """Set the authenticated session."""
    _set_session(username)
