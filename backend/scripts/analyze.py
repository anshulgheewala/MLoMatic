# # scripts/analyze.py
# import sys
# import pandas as pd
# from ydata_profiling import ProfileReport

# csv_path = sys.argv[1]
# output_html = sys.argv[2]

# # Load CSV
# df = pd.read_csv(csv_path)

# # Generate HTML report
# profile = ProfileReport(df, title="CSV Data Analysis Report", explorative=True)
# profile.to_file(output_html)
# backend/scripts/analyze.py

# Remember to always switch to to venv.

# sweet viz

import sys
import pandas as pd
import sweetviz as sv
import numpy as np
import warnings

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning
# Get the file paths from the command line arguments
csv_path = sys.argv[1]
output_html = sys.argv[2]

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path)

# 1. Analyze the dataframe using Sweetviz
analysis_report = sv.analyze(df)

# 2. Generate the HTML report
#    show_html() will save the report to the specified file path.
analysis_report.show_html(output_html, open_browser=False)

print(f"Sweetviz report generated at {output_html}")