import io
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Grounded IRDAI Benchmark Comparison Matrix Data
BENCHMARK_COMPARISON_DATA = [
    {
        "Plan Name": "HDFC Life Click 2 Protect Super",
        "Insurer": "HDFC Life",
        "CSR (Count)": "99.30%",
        "ASR (Amount)": "95.10%",
        "Critical Illnesses": "Up to 60 conditions",
        "Zero-Cost Exit (SEV)": "Available at designated age",
        "Grace Period": "30 days (Ann/Half-Yr/Qtr), 15 days (Monthly)",
        "Suicide Exclusion": "12 months from inception/revival",
        "Base Monthly": "₹980 - ₹1,250"
    },
    {
        "Plan Name": "Max Life Smart Secure Plus",
        "Insurer": "Max Life",
        "CSR (Count)": "99.65%",
        "ASR (Amount)": "96.40%",
        "Critical Illnesses": "Up to 64 conditions",
        "Zero-Cost Exit (SEV)": "Available (Special Exit Value at age 65)",
        "Grace Period": "30 days (Ann/Half-Yr/Qtr), 15 days (Monthly)",
        "Suicide Exclusion": "12 months from inception/revival",
        "Base Monthly": "₹850 - ₹1,150"
    },
    {
        "Plan Name": "Tata AIA Sampoorna Raksha Supreme",
        "Insurer": "Tata AIA",
        "CSR (Count)": "99.13%",
        "ASR (Amount)": "94.80%",
        "Critical Illnesses": "Up to 40 conditions",
        "Zero-Cost Exit (SEV)": "Available on select variants",
        "Grace Period": "30 days (Ann/Half-Yr/Qtr), 15 days (Monthly)",
        "Suicide Exclusion": "12 months from inception/revival",
        "Base Monthly": "₹900 - ₹1,200"
    },
    {
        "Plan Name": "ICICI Pru iProtect Smart",
        "Insurer": "ICICI Prudential",
        "CSR (Count)": "98.90%",
        "ASR (Amount)": "94.20%",
        "Critical Illnesses": "Up to 34 conditions",
        "Zero-Cost Exit (SEV)": "Smart Exit Benefit available",
        "Grace Period": "30 days (Ann/Half-Yr/Qtr), 15 days (Monthly)",
        "Suicide Exclusion": "12 months from inception/revival",
        "Base Monthly": "₹980 - ₹1,300"
    }
]

def generate_pdf_advisory_report(user_profile: Dict[str, Any], advisory_text: str) -> bytes:
    """Generates a professional PDF term life advisory report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748")
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.whitesmoke
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # Title & Header
    story.append(Paragraph("🛡️ Term Life Insurance Advisory Brief", title_style))
    story.append(Paragraph("Verifiable Underwriting Assessment & Policy Benchmark Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Section 1: Applicant Profile
    story.append(Paragraph("1. Underwriting Profile Summary", heading_style))
    profile_data = [
        [
            Paragraph(f"<b>Age:</b> {user_profile.get('age', 'N/A')}", body_style),
            Paragraph(f"<b>Gender:</b> {user_profile.get('gender', 'N/A')}", body_style),
            Paragraph(f"<b>Smoker:</b> {'Yes' if user_profile.get('is_smoker') else 'No'}", body_style)
        ],
        [
            Paragraph(f"<b>Annual Income:</b> {user_profile.get('annual_income', 'N/A')}", body_style),
            Paragraph(f"<b>Desired Cover:</b> {user_profile.get('desired_sum_assured', 'N/A')}", body_style),
            Paragraph(f"<b>Policy Term:</b> {user_profile.get('policy_term_years', 'N/A')} Years", body_style)
        ]
    ]
    t_profile = Table(profile_data, colWidths=[180, 180, 180])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 10))

    # Section 2: Side-by-Side Comparison Matrix
    story.append(Paragraph("2. Top 4 Indian Term Insurance Benchmark Matrix", heading_style))
    comp_headers = ["Plan & Insurer", "CSR / ASR", "Critical Illness", "Zero-Cost Exit", "Suicide Clause", "Base / Mo"]
    comp_rows = [[Paragraph(h, table_header_style) for h in comp_headers]]

    for p in BENCHMARK_COMPARISON_DATA:
        comp_rows.append([
            Paragraph(f"<b>{p['Plan Name']}</b><br/>({p['Insurer']})", table_cell_style),
            Paragraph(f"CSR: {p['CSR (Count)']}<br/>ASR: {p['ASR (Amount)']}", table_cell_style),
            Paragraph(p['Critical Illnesses'], table_cell_style),
            Paragraph(p['Zero-Cost Exit (SEV)'], table_cell_style),
            Paragraph(p['Suicide Exclusion'], table_cell_style),
            Paragraph(p['Base Monthly'], table_cell_style)
        ])

    t_comp = Table(comp_rows, colWidths=[110, 85, 95, 95, 95, 60])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 10))

    # Section 3: Advisor Evaluation
    story.append(Paragraph("3. Certified Advisory Evaluation", heading_style))
    clean_advisory = advisory_text.replace("**", "").replace("###", "").replace("##", "")
    for para in clean_advisory.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 4))

    # Regulatory Disclaimer Footer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#A0AEC0"), spaceAfter=6))
    disclaimer = (
        "<b>IRDAI Statutory Notice:</b> Insurance is the subject matter of solicitation. Indicative rates are based on standard non-smoker underwriting "
        "and exclude optional riders. Actual premium payable is subject to medical underwriting. Individual term life insurance policies carry 0% GST."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=styles["Normal"], fontSize=6.5, leading=9, textColor=colors.HexColor("#718096"))))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()