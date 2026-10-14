# ============================================================
# PERSONALIZED LEARNING RECOMMENDATIONS
# ============================================================


def generate_recommendations(student, support_level):

    recommendations = []


    # ========================================================
    # 1. SUPPORT LEVEL
    # ========================================================

    if support_level == "High Support":

        recommendations.append(
            "Create a basic study plan with short daily study sessions."
        )

        recommendations.append(
            "Provide additional academic support and regular faculty guidance."
        )

    elif support_level == "Medium Support":

        recommendations.append(
            "Follow a consistent weekly study schedule."
        )

        recommendations.append(
            "Focus additional practice on difficult subjects."
        )

    else:

        recommendations.append(
            "Continue the current study routine."
        )

        recommendations.append(
            "Use advanced practice materials to maintain performance."
        )


    # ========================================================
    # 2. STUDY TIME
    # ========================================================

    if student["studytime"] <= 2:

        recommendations.append(
            "Increase weekly study time gradually."
        )


    # ========================================================
    # 3. PREVIOUS FAILURES
    # ========================================================

    if student["failures"] > 0:

        recommendations.append(
            "Review topics from previously failed subjects "
            "and schedule targeted revision."
        )


    # ========================================================
    # 4. ABSENCES
    # ========================================================

    if student["absences"] > 10:

        recommendations.append(
            "Reduce unnecessary absences and improve class attendance."
        )


    # ========================================================
    # 5. SCHOOL SUPPORT
    # ========================================================

    if student["schoolsup"] == "no":

        recommendations.append(
            "Consider using available school-based academic support."
        )


    # ========================================================
    # 6. FAMILY SUPPORT
    # ========================================================

    if student["famsup"] == "no":

        recommendations.append(
            "Discuss study goals with family and build a supportive "
            "study environment at home."
        )


    # ========================================================
    # 7. INTERNET ACCESS
    # ========================================================

    if student["internet"] == "no":

        recommendations.append(
            "Use offline study materials, textbooks, and downloaded "
            "learning resources where internet access is limited."
        )

    else:

        recommendations.append(
            "Use trusted online learning resources for additional practice."
        )


    # ========================================================
    # 8. FINAL RECOMMENDATIONS
    # ========================================================

    return recommendations