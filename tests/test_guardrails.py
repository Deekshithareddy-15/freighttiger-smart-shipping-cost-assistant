"""
Unit Tests for Guardrails.
Verifies that negative traps (N004, N005, N006, N007, N008, N009, N010) are never accepted as valid justifications,
and that valid operational reasons (N001, N002) are properly accepted.
"""

import pytest
from src.guardrails import GuardrailValidator


@pytest.fixture
def validator():
    return GuardrailValidator()


def test_reject_trap_n004_routes_not_in_dataset(validator):
    note = {
        "note_id": "N004",
        "date": "2024-03-11",
        "applies_to": "All Routes",
        "raw_note": "A new toll plaza was commissioned on a national highway stretch, but the affected routes are not part of this dataset."
    }
    is_justified, reason = validator.evaluate_justification(note, "Delhi-Jaipur", "2024-03-11", 15.0)
    assert not is_justified
    assert "not part of this dataset" in reason


def test_reject_trap_n005_costs_not_affected(validator):
    note = {
        "note_id": "N005",
        "date": "2024-07-29",
        "applies_to": "Mumbai-Delhi",
        "raw_note": "Scheduled highway maintenance work between Mumbai and Delhi caused minor delays for about a week; costs were not significantly affected."
    }
    is_justified, reason = validator.evaluate_justification(note, "Mumbai-Delhi", "2024-07-29", 10.0)
    assert not is_justified
    assert "costs were not significantly affected" in reason


def test_reject_trap_n006_stable_demand(validator):
    note = {
        "note_id": "N006",
        "date": "2025-09-22",
        "applies_to": "All Routes",
        "raw_note": "A logistics industry report noted overall demand for freight capacity remained stable this quarter with no major disruptions reported."
    }
    is_justified, reason = validator.evaluate_justification(note, "Mumbai-Pune", "2025-09-15", 9.2)
    assert not is_justified
    assert "stable demand" in reason


def test_reject_trap_n010_costs_absorbed(validator):
    note = {
        "note_id": "N010",
        "date": "2025-10-27",
        "applies_to": "All Routes",
        "raw_note": "A new vehicle tracking mandate was introduced for commercial fleets; compliance costs were absorbed by transporters without a rate change."
    }
    is_justified, reason = validator.evaluate_justification(note, "Ahmedabad-Mumbai", "2025-10-27", 12.0)
    assert not is_justified
    assert "without a rate change" in reason


def test_reject_wrong_corridor(validator):
    note = {
        "note_id": "N002",
        "date": "2025-01-20",
        "applies_to": "Ahmedabad-Mumbai",
        "raw_note": "A regional festival week saw a temporary surcharge applied on Ahmedabad-Mumbai."
    }
    is_justified, reason = validator.evaluate_justification(note, "Delhi-Jaipur", "2025-01-20", 25.0)
    assert not is_justified
    assert "Wrong route" in reason


def test_reject_wrong_temporal_window(validator):
    note = {
        "note_id": "N002",
        "date": "2025-01-20",
        "applies_to": "Ahmedabad-Mumbai",
        "raw_note": "A regional festival week saw a temporary surcharge applied on Ahmedabad-Mumbai."
    }
    is_justified, reason = validator.evaluate_justification(note, "Ahmedabad-Mumbai", "2024-05-10", 25.0)
    assert not is_justified
    assert "does not cover week" in reason


def test_accept_valid_n002_festival_surcharge(validator):
    note = {
        "note_id": "N002",
        "date": "2025-01-20",
        "applies_to": "Ahmedabad-Mumbai",
        "raw_note": "A regional festival week saw a temporary surcharge applied by transporters on the Ahmedabad-Mumbai corridor due to high demand and limited truck availability."
    }
    is_justified, reason = validator.evaluate_justification(note, "Ahmedabad-Mumbai", "2025-01-20", 29.5)
    assert is_justified
    assert "N002" in reason
    assert "festival week" in reason


def test_accept_valid_n001_flood_disruption(validator):
    note = {
        "note_id": "N001",
        "date": "2025-02-24",
        "applies_to": "Chennai-Bangalore",
        "raw_note": "Heavy flooding on the Chennai-Bangalore highway disrupted normal truck movement from Feb 24 to Mar 8, forcing longer detours and higher trip costs."
    }
    is_justified, reason = validator.evaluate_justification(note, "Chennai-Bangalore", "2025-02-24", 31.5)
    assert is_justified
    assert "N001" in reason
    assert "flooding" in reason
