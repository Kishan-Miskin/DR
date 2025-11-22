import tensorflow as tf
from flask import Flask, render_template, request, send_file, url_for
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
        borderWidth=2,
        borderColor=colors.HexColor('#0b5e3a'),
        borderPadding=10,
        backColor=colors.HexColor('#f0f8f5')
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#0b5e3a'),
        fontName='Helvetica-Bold',
        spaceAfter=12,
        spaceBefore=16,
        borderWidth=0,
   
        leftIndent=0,
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

    # ==================== PAGE 1: HEADER & INFORMATION ====================
    
    # Hospital Header
    story.append(Paragraph("<b>Retinex Retinopathy Detection</b>", title_style))
    story.append(Paragraph(
        "Ai Based Retinopathy Detection<br/>",
        hospital_style
    ))
    
    # Horizontal line
    story.append(Spacer(1, 10))
    line_data = [['', '']]
    line_table = Table(line_data, colWidths=[500])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 2, colors.HexColor('#0b5e3a')),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 20))

    # Report Title
    story.append(Paragraph("DIABETIC RETINOPATHY SCREENING REPORT", report_title_style))

    # Patient & Scan Information
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
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0b5e3a')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5'))
    ]))

    story.append(info_table)
    story.append(Spacer(1, 25))

    # Assessment Section
    story.append(Paragraph("CLINICAL ASSESSMENT", section_heading_style))
    story.append(Paragraph(
        "This automated screening analysis evaluates digital fundus photography for clinical "
        "indicators of diabetic retinopathy (DR). The artificial intelligence model examines "
        "retinal patterns including microaneurysms, hemorrhages, hard exudates, cotton wool spots, "
        "and neovascularization. This technology assists healthcare providers in early detection "
        "and timely referral for comprehensive ophthalmologic evaluation.",
        body_style
    ))
    story.append(Spacer(1, 20))

    # Diagnosis Result
    story.append(Paragraph("SCREENING RESULT", section_heading_style))
    story.append(Paragraph(diagnosis, diagnosis_style))
    story.append(Spacer(1, 20))

    # Probability Table
    story.append(Paragraph("PREDICTION CONFIDENCE LEVELS", section_heading_style))
    
    p_positive = round(probabilities[0] * 100, 2)
    p_negative = round(probabilities[1] * 100, 2)

    prob_data = [
        ['Classification', 'Confidence (%)', 'Interpretation'],
        ['Diabetic Retinopathy Detected', f'{p_positive}%', 
         'Positive finding' if p_positive > 50 else 'Low probability'],
        ['No Diabetic Retinopathy', f'{p_negative}%', 
         'Negative finding' if p_negative > 50 else 'Low probability']
    ]

    prob_table = Table(prob_data, colWidths=[200, 140, 150])
    prob_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b5e3a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffebee') if p_positive > 50 else colors.HexColor('#e8f5e9')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#e8f5e9') if p_negative > 50 else colors.HexColor('#ffebee')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0b5e3a')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
    ]))

    story.append(prob_table)
    story.append(Spacer(1, 25))

    # ==================== PAGE 2: RECOMMENDATIONS ====================
    story.append(PageBreak())

    # Lifestyle Recommendations
    story.append(Paragraph("LIFESTYLE RECOMMENDATIONS", section_heading_style))
    
    lifestyle_recommendations = [
        ['<b>Blood Glucose Control</b>', 
         'Maintain HbA1c levels below 7% (or as recommended by your physician). '
         'Regular monitoring and medication adherence significantly reduce retinopathy progression.'],
        
        ['<b>Blood Pressure Management</b>', 
         'Target BP < 130/80 mmHg. Hypertension accelerates microvascular damage in the retina.'],
        
        ['<b>Regular Exercise</b>', 
         'Aim for at least 150 minutes of moderate aerobic activity per week (walking, cycling, swimming). '
         'Consult your physician before starting new exercise routines.'],
        
        ['<b>Smoking Cessation</b>', 
         'Tobacco use significantly increases risk of diabetic complications including retinopathy. '
         'Seek support programs if needed.'],
        
        ['<b>Weight Management</b>', 
         'Maintain healthy BMI (18.5-24.9). Even modest weight loss improves glycemic control.'],
        
        ['<b>Stress Management</b>', 
         'Chronic stress affects blood sugar levels. Practice relaxation techniques, adequate sleep (7-8 hours), '
         'and mindfulness exercises.']
    ]

    for item in lifestyle_recommendations:
        lifestyle_data = [[Paragraph(item[0], body_style), Paragraph(item[1], body_style)]]
        lifestyle_table = Table(lifestyle_data, colWidths=[140, 350])
        lifestyle_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#e8f5e9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc'))
        ]))
        story.append(lifestyle_table)
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 15))

    # Dietary Recommendations
    story.append(Paragraph("DIETARY RECOMMENDATIONS", section_heading_style))

    diet_intro = Paragraph(
        "A retinal-protective diet rich in antioxidants, omega-3 fatty acids, and low glycemic "
        "index foods can help manage diabetes and protect vision:",
        body_style
    )
    story.append(diet_intro)
    story.append(Spacer(1, 12))

    diet_data = [
        ['Food Category', 'Recommended Foods', 'Foods to Limit'],
        
        ['Vegetables', 
         '• Leafy greens (spinach, kale)\n• Carrots, bell peppers\n• Broccoli, cauliflower',
         '• Fried vegetables\n• High-sodium canned vegetables'],
        
        ['Fruits', 
         '• Berries (blueberries, strawberries)\n• Citrus fruits\n• Apples, pears (with skin)',
         '• Fruit juices\n• Dried fruits with added sugar\n• Canned fruits in syrup'],
        
        ['Proteins', 
         '• Fatty fish (salmon, mackerel)\n• Skinless poultry\n• Legumes, beans\n• Nuts and seeds',
         '• Processed meats\n• Fried proteins\n• High-fat red meats'],
        
        ['Grains', 
         '• Whole grain bread\n• Brown rice, quinoa\n• Oats, barley',
         '• White bread, pastries\n• Refined cereals\n• White rice'],
        
        ['Fats', 
         '• Olive oil, avocado oil\n• Nuts (almonds, walnuts)\n• Avocados',
         '• Trans fats\n• Deep-fried foods\n• Excessive butter/margarine'],
        
        ['Beverages', 
         '• Water (8-10 glasses/day)\n• Green tea\n• Herbal teas',
         '• Sugary sodas\n• Excessive caffeine\n• Alcoholic beverages']
    ]

    diet_table = Table(diet_data, colWidths=[90, 200, 200])
    diet_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0b5e3a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f5f5f5')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0b5e3a')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6)
    ]))

    story.append(diet_table)
    story.append(Spacer(1, 20))

    # Screening Recommendations
    story.append(Paragraph("FOLLOW-UP SCREENING SCHEDULE", section_heading_style))
    
    screening_text = Paragraph(
        "• <b>No DR detected:</b> Annual comprehensive eye examination<br/>"
        "• <b>Mild DR:</b> Every 6-12 months or as advised by ophthalmologist<br/>"
        "• <b>Moderate to Severe DR:</b> Every 3-6 months with specialist monitoring<br/>"
        "• <b>Any visual changes:</b> Immediate ophthalmologic consultation",
        body_style
    )
    story.append(screening_text)
    story.append(Spacer(1, 20))

    # Uploaded Image (STILL ON PAGE 2)
    story.append(Paragraph("ANALYZED FUNDUS IMAGE", section_heading_style))
    story.append(Spacer(1, 15))

    try:
        img = Image(image_path)
        # Smaller image to fit on page 2
        img.drawWidth = 3.5 * inch
        img.drawHeight = 3.5 * inch
        
        # Create centered table for image
        img_table = Table([[img]], colWidths=[5 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#0b5e3a'))
        ]))
        story.append(img_table)
    except Exception as e:
        story.append(Paragraph(
            f"<i>Unable to load the uploaded image. Error: {str(e)}</i>", 
            body_style
        ))

    story.append(Spacer(1, 15))

    # Footer (AFTER IMAGE ON PAGE 2)
    footer_line = Table([['', '']], colWidths=[490])
    footer_line.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc'))
    ]))
    story.append(footer_line)
    story.append(Spacer(10, 5))
    
    story.append(Paragraph(
        "<b>Retinex Retinopathy Detection</b> | AI Based Retinopathy Detection<br/>"
        f"Report generated on {analysis_date} | Report ID: {report_id}",
        footer_style
    ))

    # ==================== PAGE 3: DISCLAIMER ONLY ====================
    story.append(PageBreak())

    # Medical Disclaimer
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", section_heading_style))
    
    disclaimer_box_data = [[Paragraph(
        "<b>THIS IS AN AI-ASSISTED SCREENING TOOL ONLY</b><br/><br/>"
        "This report is generated by an artificial intelligence algorithm for preliminary screening purposes. "
        "It is <b>NOT</b> a substitute for professional medical diagnosis or treatment. The AI model has "
        "limitations and may produce false positives or false negatives.<br/><br/>"
        "<b>REQUIRED ACTIONS:</b><br/>"
        "• Schedule a comprehensive eye examination with a certified ophthalmologist<br/>"
        "• Do not make treatment decisions based solely on this report<br/>"
        "• Seek immediate medical attention if experiencing vision changes<br/><br/>"
        "This screening does not replace dilated eye examinations, optical coherence tomography (OCT), "
        "or fluorescein angiography when clinically indicated.",
        body_style
    )]]
    
    disclaimer_table = Table(disclaimer_box_data, colWidths=[490])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3cd')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#d32f2f')),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15)
    ]))
    
    story.append(disclaimer_table)
    story.append(Spacer(1, 30))

    # Footer
    story.append(Spacer(1, 20))
    footer_line = Table([['', '']], colWidths=[490])
    footer_line.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc'))
    ]))
   

    # Build PDF
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
