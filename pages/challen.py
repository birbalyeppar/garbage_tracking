import streamlit as st
import datetime

st.set_page_config(page_title="Challan Generator", layout="wide")

# ---------------------------
# Dynamic Challan Data
# ---------------------------
generated_date = datetime.date.today()
expiry_date = generated_date + datetime.timedelta(days=30)

challan_data = {
    "receipt_no": "250110",
    "date": generated_date.strftime("%d-%m-%Y"),
    "ward": "60",
    "name_hi": "श्री तेज सिंह / मुकेश सिंह",
    "name_en": "Shri Tej Singh / Mukesh Singh",
    "address_hi": "200 फीट बाईपास, पाट पर, जयपुर",
    "address_en": "200 Feet Bypass, Pat Par, Jaipur",
    "charges": {
        "hi": {
            "गृहकर": {"amount": "", "link": None},
            "भूमि किराया (लीज़)": {"amount": "", "link": None},
            "दुकान किराया": {"amount": "", "link": None},
            "लाइसेंस शुल्क": {"amount": "", "link": None},
            "विज्ञापन शुल्क": {"amount": "", "link": None},
            "सरसरी": {"amount": "", "link": None},
            "ब्याज": {"amount": "", "link": None},
            "पेनल्टी": {"amount": "", "link": None},
            "जन्म / मृत्यु प्रमाण पत्र शुल्क": {"amount": "500", "link": None},
            "कोरोना जांच शुल्क": {"amount": "", "link": None},
            "अन्य": {"amount": "", "link": None},
            "कचरा फेंकने का शुल्क": {
                "amount": "1000",
                "link": "https://example.com/evidence/garbage123"
            },
        },
        "en": {
            "House Tax": {"amount": "", "link": None},
            "Land Lease Rent": {"amount": "", "link": None},
            "Shop Rent": {"amount": "", "link": None},
            "License Fee": {"amount": "", "link": None},
            "Advertisement Fee": {"amount": "", "link": None},
            "Miscellaneous": {"amount": "", "link": None},
            "Interest": {"amount": "", "link": None},
            "Penalty": {"amount": "", "link": None},
            "Birth / Death Certificate Fee": {"amount": "500", "link": None},
            "COVID-19 Test Fee": {"amount": "", "link": None},
            "Others": {"amount": "", "link": None},
            "Garbage Throw Penalty": {
                "amount": "1000",
                "link": "https://example.com/evidence/garbage123"
            },
        },
    },
    "collector_hi": "संग्रहकर्ता",
    "collector_en": "Collector",
}

# ---------------------------
# Language Selector
# ---------------------------
lang = st.radio("Select Language / भाषा चुनें:", ["Hindi", "English"])

if lang == "Hindi":
    name = challan_data["name_hi"]
    address = challan_data["address_hi"]
    collector = challan_data["collector_hi"]
    headline = "नगर निगम ग्रेटर जयपुर"
    subhead = "(राजस्थान नगरपालिका लेखा नियम 1963 के अनुसार)"
    charges = challan_data["charges"]["hi"]
else:
    name = challan_data["name_en"]
    address = challan_data["address_en"]
    collector = challan_data["collector_en"]
    headline = "Nagar Nigam Greater Jaipur"
    subhead = "(According to Rajasthan Municipal Accounts Rules 1963)"
    charges = challan_data["charges"]["en"]

# ---------------------------
# Charges Table HTML
# ---------------------------
rows_html = ""
total_amount = 0
for k, v in charges.items():
    amt = v["amount"]
    link = v["link"]

    # Expiry check
    if link and datetime.date.today() > expiry_date:
        link_html = "<br><span style='color:red; font-size:12px'>(Expired)</span>"
    elif link:
        link_html = f"<br><a href='{link}' target='_blank' style='font-size:12px;'>Evidence Link</a>"
    else:
        link_html = ""

    rows_html += f"""
    <tr>
      <td style="border:1px solid #999; padding:6px;">{k}</td>
      <td style="border:1px solid #999; padding:6px; text-align:center;">
        {amt if amt else ""} {link_html}
      </td>
    </tr>
    """
    if amt and amt.isdigit():
        total_amount += int(amt)

# ---------------------------
# Final HTML Template
# ---------------------------
html = f"""
<html>
<head>
<style>
  body {{
    font-family: Arial, sans-serif;
    background: #fff;
    margin: 0;
    padding: 0;
  }}
  .paper {{
    width: 750px;
    margin: auto;
    padding: 20px;
    border: 1px solid #333;
    background: white;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed #555;
    padding-bottom: 10px;
  }}
  .header img {{
    height: 80px;
  }}
  .org h2 {{
    margin: 0;
    font-size: 20px;
  }}
  .org p {{
    margin: 0;
    font-size: 13px;
    color: #555;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
  }}
  td, th {{
    border: 1px solid #999;
    padding: 6px;
    font-size: 14px;
  }}
  .footer {{
    margin-top: 20px;
    border-top: 1px dashed #555;
    padding-top: 10px;
    font-size: 13px;
    color: #555;
  }}
</style>
</head>
<body>
<div class="paper" id="challan">
  <div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/en/d/db/Jaipur_Municipal_Corporation_Logo.png">
    <div class="org">
      <h2>{headline}</h2>
      <p>{subhead}</p>
    </div>
    <div style="text-align:right; font-size:13px;">
      <div><b>Receipt No:</b> {challan_data['receipt_no']}</div>
      <div><b>Date:</b> {challan_data['date']}</div>
      <div><b>Ward:</b> {challan_data['ward']}</div>
    </div>
  </div>

  <div style="margin-top:15px;">
    <p><b>Name:</b> {name}</p>
    <p><b>Address:</b> {address}</p>
  </div>

  <table>
    <tr>
      <th>Charges</th>
      <th>Amount (₹)</th>
    </tr>
    {rows_html}
    <tr>
      <td style="text-align:right;"><b>Total</b></td>
      <td style="text-align:center;"><b>{total_amount}</b></td>
    </tr>
  </table>

  <div class="footer">
    <p>Note: This receipt is valid only with signature and seal.</p>
    <p style="text-align:right">{collector}: ____________________</p>
  </div>
</div>

<!-- Download Buttons -->
<div style="margin-top:20px;">
  <button onclick="downloadPDF()">Download PDF</button>
  <button onclick="downloadPNG()">Download PNG</button>
</div>

<!-- JS Libs -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script>
async function downloadPDF() {{
  const el = document.getElementById('challan');
  const canvas = await html2canvas(el, {{scale:2}});
  const imgData = canvas.toDataURL('image/png');
  const {{ jsPDF }} = window.jspdf;
  const pdf = new jsPDF('p','pt','a4');
  const imgProps = pdf.getImageProperties(imgData);
  const pdfWidth = pdf.internal.pageSize.getWidth();
  const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
  pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
  pdf.save('challan.pdf');
}}
async function downloadPNG() {{
  const el = document.getElementById('challan');
  const canvas = await html2canvas(el, {{scale:2}});
  const link = document.createElement('a');
  link.download = 'challan.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}}
</script>
</body>
</html>
"""

# ---------------------------
# Show in Streamlit
# ---------------------------
st.components.v1.html(html, height=950, scrolling=True)
