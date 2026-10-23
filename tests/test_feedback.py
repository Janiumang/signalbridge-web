"""Sign-up form validation."""
from __future__ import annotations

import feedback

OK = ("a@b.com", feedback.ROLES[1], feedback.ORG_TYPES[1], "PRR please", True)


def test_valid_submission_passes():
    assert feedback.validate(*OK) == []


def test_each_rule_rejects():
    assert feedback.validate("not-an-email", *OK[1:])
    assert feedback.validate("a@b", *OK[1:])
    assert feedback.validate(OK[0], feedback.ROLES[0], *OK[2:])
    assert feedback.validate(*OK[:2], feedback.ORG_TYPES[0], *OK[3:])
    assert feedback.validate(*OK[:3], "x" * (feedback.MAX_TEXT + 1), True)
    assert feedback.validate(*OK[:4], False)


def test_not_configured_without_secrets():
    assert feedback.is_configured() is False
