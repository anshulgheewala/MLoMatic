import sys
import os
import pandas as pd
import sweetviz as sv
import numpy as np
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix numpy warnings for compatibility
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning

if len(sys.argv) >= 3:
    csv_path = sys.argv[1]
    output_html = sys.argv[2]
else:
    csv_path = os.getenv("CSV_PATH", "./uploads/sample.csv")
    output_html = os.getenv("REPORT_PATH", "./reports/report.html")

try:
    df = pd.read_csv(csv_path)

    analysis_report = sv.analyze(df)

    analysis_report.show_html(output_html, open_browser=False)

    print(f"✅ Sweetviz report generated at {output_html}")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    sys.exit(1)
