from typing import Optional

VARIABLE_GROUPS = {
    "Academic": ["Q3", "Q6", "Q7", "Q8", "Q9", "Q33", "Q34", "Q35"],
    "Balance": ["Q7", "Q20", "Q21", "Q32", "Q12", "Q13"],
    "Psychological": ["Q4", "Q5", "Q17", "Q18", "Q19", "Q23", "Q24"],
    "Unusual": ["Q2", "Q28", "Q29", "Q1", "Q15", "Q16"],
}

INTERVENTION_MATRIX = {
    ("Academic", "CRITICAL"): {
        "actions": ["Academic coaching", "Learning assessment", "Course review"],
        "timeline": "1 week",
        "owner": "Director + Learning Advisor",
        "frequency": "Weekly meetings",
        "metrics": ["GPA improvement", "Study hours consistency"],
    },
    ("Academic", "HIGH"): {
        "actions": ["Tutoring program", "Study skills workshop"],
        "timeline": "1-2 weeks",
        "owner": "Learning Advisor",
        "frequency": "Bi-weekly check-ins",
        "metrics": ["Assignment completion", "Grade trajectory"],
    },
    ("Academic", "MODERATE"): {
        "actions": ["Study group enrollment", "Online tutoring access"],
        "timeline": "2-4 weeks",
        "owner": "Semillero Mentor",
        "frequency": "Monthly monitoring",
        "metrics": ["Participation rate", "Performance improvement"],
    },
    ("Balance", "CRITICAL"): {
        "actions": ["Work-study plan reduction", "Financial aid review"],
        "timeline": "1 week",
        "owner": "Director + Financial Aid",
        "frequency": "Bi-weekly meetings",
        "metrics": ["Work hours reduction", "Financial situation improvement"],
    },
    ("Balance", "HIGH"): {
        "actions": ["Work schedule optimization", "Scholarship exploration"],
        "timeline": "1-2 weeks",
        "owner": "Academic Advisor + Financial Aid",
        "frequency": "Monthly meetings",
        "metrics": ["Schedule optimization approval", "Financial aid increase"],
    },
    ("Balance", "MODERATE"): {
        "actions": ["Time management workshop", "Resource awareness session"],
        "timeline": "2-4 weeks",
        "owner": "Semillero Mentor",
        "frequency": "Quarterly check-in",
        "metrics": ["Workshop attendance", "Schedule planning"],
    },
    ("Psychological", "CRITICAL"): {
        "actions": [
            "Psychological assessment",
            "Counseling referral",
            "Stress management plan",
        ],
        "timeline": "1 week",
        "owner": "Director + Counselor + Psychologist",
        "frequency": "Weekly counseling",
        "metrics": ["Counseling engagement", "Stress level reduction"],
    },
    ("Psychological", "HIGH"): {
        "actions": [
            "Career counseling",
            "Stress reduction workshop",
            "Peer mentoring",
        ],
        "timeline": "1-2 weeks",
        "owner": "Career Counselor + Mentor",
        "frequency": "Bi-weekly counseling",
        "metrics": ["Career clarity", "Engagement improvement"],
    },
    ("Psychological", "MODERATE"): {
        "actions": ["Wellness program enrollment", "Interest assessment"],
        "timeline": "2-4 weeks",
        "owner": "Semillero Mentor",
        "frequency": "Monthly check-in",
        "metrics": ["Program participation", "Motivation indicators"],
    },
    ("Unusual", "CRITICAL"): {
        "actions": [
            "Comprehensive profile review",
            "Special circumstances assessment",
        ],
        "timeline": "1-2 weeks",
        "owner": "Director + Academic Committee",
        "frequency": "Weekly meetings",
        "metrics": ["Situation clarification", "Action plan development"],
    },
    ("Unusual", "HIGH"): {
        "actions": ["Context assessment", "Individualized plan development"],
        "timeline": "2-4 weeks",
        "owner": "Academic Advisor",
        "frequency": "Bi-weekly monitoring",
        "metrics": ["Plan completion", "Adaptation indicators"],
    },
    ("Unusual", "MODERATE"): {
        "actions": ["Stakeholder communication", "Resource customization"],
        "timeline": "1 month",
        "owner": "Semillero Director",
        "frequency": "Monthly check-in",
        "metrics": ["Stakeholder alignment", "Resource utilization"],
    },
}


def classify_risk(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MODERATE"
    return "NORMAL"


def classify_type(raw_q: dict, means: dict, stds: dict) -> str:
    group_scores: dict[str, float] = {g: 0.0 for g in VARIABLE_GROUPS}
    for group, variables in VARIABLE_GROUPS.items():
        for var in variables:
            if var in raw_q and var in means:
                z = abs((raw_q[var] - means[var]) / (stds[var] + 1e-10))
                group_scores[group] += z
    return max(group_scores, key=group_scores.get)


def get_intervention(
    anomaly_type: str, risk_level: str
) -> Optional[dict]:
    return INTERVENTION_MATRIX.get((anomaly_type, risk_level))
