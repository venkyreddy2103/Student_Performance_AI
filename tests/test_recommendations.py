import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from recommendations import generate_recommendations


def test_high_support_recommendations():

    student = {
        "studytime": 1,
        "failures": 2,
        "absences": 15,
        "schoolsup": "no",
        "famsup": "no",
        "internet": "yes"
    }

    recommendations = generate_recommendations(
        student,
        "High Support"
    )

    assert len(recommendations) > 0


def test_medium_support_recommendations():

    student = {
        "studytime": 2,
        "failures": 0,
        "absences": 5,
        "schoolsup": "yes",
        "famsup": "yes",
        "internet": "yes"
    }

    recommendations = generate_recommendations(
        student,
        "Medium Support"
    )

    assert len(recommendations) > 0


def test_low_support_recommendations():

    student = {
        "studytime": 3,
        "failures": 0,
        "absences": 2,
        "schoolsup": "yes",
        "famsup": "yes",
        "internet": "yes"
    }

    recommendations = generate_recommendations(
        student,
        "Low Support"
    )

    assert len(recommendations) > 0