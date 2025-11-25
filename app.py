import tensorflow as tf
from flask import Flask, render_template, request, send_file, url_for
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


app = Flask(__name__)

# SAVE INTO STATIC FOLDER SO BROWSER CAN LOAD IMAGES
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folder if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






def predict_class(path):
    img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    image = np.array(RGBImg) / 255.0

    model = tf.saved_model.load("64x3-CNN.model")
    infer = model.signatures["serving_default"]
    
    predict = infer(tf.constant([image], dtype=tf.float32))
    probabilities = predict['dense_1'].numpy()[0].tolist()

    diagnosis = (
        "No Diabetic Retinopathy Detected"
        if np.argmax(probabilities) == 1
        else "Diabetic Retinopathy Detected"
    )

    return diagnosis, probabilities


from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, Image, Frame, PageTemplate)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

def generate_pdf_report(image_path, diagnosis, probabilities):
    """
    Generates a professional medical report PDF with enhanced formatting.
    """
    file_name = "Medical_Analysis_Report.pdf"
    doc = SimpleDocTemplate(
        file_name,
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    story = []

    # ==================== CUSTOM STYLES ====================
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#0b5e3a'),
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )

    hospital_style = ParagraphStyle(
        'HospitalStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=20
    )

    report_title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0b5e3a'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica-Bold',
        backColor=colors.HexColor('#f0f8f5'),
        borderWidth=2,
        borderColor=colors.HexColor('#0b5e3a'),
        borderPadding=10
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#0b5e3a'),
        fontName='Helvetica-Bold',
        spaceAfter=12,
        spaceBefore=16,
        backColor=colors.HexColor('#e8f5e9'),
        borderPadding=6
    )

    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['BodyText'],
        fontSize=10,
        leading=16,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#333333')
    )

    diagnosis_style = ParagraphStyle(
        'DiagnosisStyle',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#d32f2f'),
        spaceAfter=10,
        alignment=TA_CENTER,
        backColor=colors.HexColor('#ffebee'),
        borderWidth=1,
        borderColor=colors.HexColor('#d32f2f'),
        borderPadding=10
    )

    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER
    )

    # ==================== PAGE 1 ====================

    story.append(Paragraph("<b>Retinex Retinopathy Detection</b>", title_style))
    story.append(Paragraph("Ai Based Retinopathy Detection<br/>", hospital_style))

    story.append(Spacer(1, 10))
    line_table = Table([['']], colWidths=[500])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (0, 0), 2, colors.HexColor('#0b5e3a')),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("DIABETIC RETINOPATHY SCREENING REPORT", report_title_style))

    analysis_date = datetime.now().strftime("%B %d, %Y at %H:%M")
    report_id = f"DR-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    info_data = [
        ['Report ID :', report_id],
        ['Analysis Date :', analysis_date],
        ['Scan Type :', 'Fundus Photography (Digital Retinal Imaging)'],
        ['Image Resolution :', '32×32 pixels (Processed)'],
        ['AI Model :', 'Convolutional Neural Network v2.3'],
        ['Processing Time :', '< 2 seconds']
    ]

    info_table = Table(info_data, colWidths=[150, 340])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
    ]))

    story.append(info_table)
    story.append(Spacer(1, 25))

    story.append(Paragraph("CLINICAL ASSESSMENT", section_heading_style))
    story.append(Paragraph(
        "This automated screening analysis evaluates digital fundus photography for clinical "
        "indicators of diabetic retinopathy...",
        body_style
    ))
    story.append(Spacer(1, 20))

    # Diagnosis
    story.append(Paragraph("SCREENING RESULT", section_heading_style))
    story.append(Paragraph(diagnosis, diagnosis_style))
    story.append(Spacer(1, 20))

    # Probabilities
    story.append(Paragraph("PREDICTION CONFIDENCE LEVELS", section_heading_style))

    p_pos = round(probabilities[0] * 100, 2)
    p_neg = round(probabilities[1] * 100, 2)

    prob_table = Table([
        ['Classification', 'Confidence (%)', 'Interpretation'],
        ['Diabetic Retinopathy Detected', f'{p_pos}%', 'Positive finding' if p_pos > 50 else 'Low probability'],
        ['No Diabetic Retinopathy', f'{p_neg}%', 'Negative finding' if p_neg > 50 else 'Low probability']
    ], colWidths=[200, 140, 150])

    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b5e3a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0b5e3a'))
    ]))

    story.append(prob_table)
    story.append(Spacer(1, 25))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())

    # -----------------------------------------
    # IMPROVED CONDITION FOR SHOWING LIFESTYLE + DIET
    # Check if diabetic retinopathy is detected (positive case)
    # -----------------------------------------
    diagnosis_lower = diagnosis.lower().strip()
    # Show recommendations only if DR is detected (when it's NOT "no diabetic retinopathy")
    show_recommendations = "no diabetic retinopathy" not in diagnosis_lower

    # ==================== LIFESTYLE (ONLY IF POSITIVE) ====================
    if show_recommendations:
        story.append(Paragraph("LIFESTYLE RECOMMENDATIONS", section_heading_style))

        lifestyle_recommendations = [
            ['<b>Blood Glucose Control</b>', 'Maintain HbA1c below 7%.'],
            ['<b>Blood Pressure</b>', 'Target < 130/80 mmHg.'],
            ['<b>Exercise</b>', '150 minutes of activity weekly.'],
            ['<b>Smoking</b>', 'Quit immediately.'],
            ['<b>Weight</b>', 'Maintain healthy BMI.'],
            ['<b>Stress</b>', 'Practice relaxation and sleep well.']
        ]

        for item in lifestyle_recommendations:
            lifestyle_table = Table([[Paragraph(item[0], body_style), Paragraph(item[1], body_style)]],
                                    colWidths=[140, 350])
            lifestyle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#e8f5e9')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
            ]))
            story.append(lifestyle_table)
            story.append(Spacer(1, 8))

        story.append(Spacer(1, 15))

    # ==================== DIET (ONLY IF POSITIVE) ====================
    if show_recommendations:
        story.append(Paragraph("DIETARY RECOMMENDATIONS", section_heading_style))

        diet_intro = Paragraph(
            "A retinal-protective diet rich in antioxidants and omega-3 helps prevent progression:",
            body_style
        )
        story.append(diet_intro)
        story.append(Spacer(1, 12))

        diet_table = Table([
            ['Food Category', 'Recommended Foods', 'Foods to Limit'],
            ['Vegetables', 'Spinach, broccoli', 'Fried vegetables'],
            ['Fruits', 'Berries, apples', 'Fruit juices'],
            ['Proteins', 'Fish, legumes', 'Processed meats'],
            ['Grains', 'Whole grains', 'White bread'],
            ['Fats', 'Olive oil, nuts', 'Trans fats'],
            ['Beverages', 'Water, green tea', 'Sugary drinks']
        ], colWidths=[90, 200, 200])

        diet_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b5e3a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0b5e3a'))
        ]))

        story.append(diet_table)
        story.append(Spacer(1, 20))

    # ==================== SCREENING (ALWAYS SHOW) ====================
    story.append(Paragraph("FOLLOW-UP SCREENING SCHEDULE", section_heading_style))

    screening_text = Paragraph(
        "• <b>No DR detected:</b> Annual eye exam<br/>"
        "• <b>Mild DR:</b> Every 6–12 months<br/>"
        "• <b>Moderate–Severe:</b> Every 3–6 months<br/>"
        "• <b>Any vision change:</b> Immediate consultation",
        body_style
    )
    story.append(screening_text)
    story.append(Spacer(1, 20))

    # ==================== IMAGE ====================
    story.append(Paragraph("ANALYZED FUNDUS IMAGE", section_heading_style))
    story.append(Spacer(1, 15))

    try:
        img = Image(image_path)
        img.drawWidth = 3.5 * inch
        img.drawHeight = 3.5 * inch
        img_table = Table([[img]], colWidths=[5 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('BOX', (0, 0), (0, 0), 2, colors.HexColor('#0b5e3a'))
        ]))
        story.append(img_table)
    except:
        story.append(Paragraph("<i>Image load failed.</i>", body_style))

    story.append(Spacer(1, 15))

    # ==================== FOOTER ====================
    footer_line = Table([['']], colWidths=[490])
    footer_line.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (0, 0), 1, colors.HexColor('#cccccc'))
    ]))
    story.append(footer_line)
    story.append(Paragraph(
        f"<b>Retinex Retinopathy Detection</b> | Report generated on {analysis_date}",
        footer_style
    ))

    # ==================== PAGE 3 DISCLAIMER ====================
    story.append(PageBreak())
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", section_heading_style))

    disclaimer_table = Table([[Paragraph(
        "<b>THIS IS AN AI-ASSISTED SCREENING TOOL ONLY</b><br/><br/>"
        "Not a substitute for medical diagnosis. Seek professional evaluation.",
        body_style
    )]], colWidths=[490])

    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#fff3cd')),
        ('BOX', (0, 0), (0, 0), 2, colors.HexColor('#d32f2f')),
        ('TOPPADDING', (0, 0), (0, 0), 15),
        ('BOTTOMPADDING', (0, 0), (0, 0), 15),
        ('LEFTPADDING', (0, 0), (0, 0), 15)
    ]))

    story.append(disclaimer_table)
    story.append(Spacer(1, 20))

    doc.build(story)
    return file_name





@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file and allowed_file(file.filename):
        filename = file.filename
        # Save inside static/uploads so browser + PDF can access it
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        diagnosis, probabilities = predict_class(save_path)

        # TIMESTAMP to show on web page (optional)
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

        return render_template(
            'predict.html',
            diagnosis=diagnosis,
            probabilities=probabilities,
            user_image=filename,   # pass only filename so template uses url_for('static', ...)
            timestamp=timestamp
        )

    return render_template('index.html', message='Error occurred')


@app.route('/download_report', methods=['POST'])
def download_report():
    image_filename = request.form['image_path']
    diagnosis = request.form['diagnosis']
    probs = request.form.getlist('probabilities')

    probabilities = list(map(float, probs))

    # Correct static path for uploaded image
    image_path = os.path.join("static", "uploads", image_filename)

    pdf_file = generate_pdf_report(image_path, diagnosis, probabilities)
    return send_file(pdf_file, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
