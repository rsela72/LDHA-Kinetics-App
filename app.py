import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from scipy.optimize import curve_fit
from io import StringIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="LDHA Kinetics Analyzer", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔬 Assay Configuration")
EPSILON = st.sidebar.number_input("Extinction Coefficient (ε) [M⁻¹ cm⁻¹]", value=6220, help="NADH at 340nm is 6220")
PATH_LENGTH = st.sidebar.number_input("Cuvette Path Length (cm)", value=1.0)

st.sidebar.divider()
st.sidebar.header("🧪 Enzyme Details")
enz_input = st.sidebar.number_input("Enzyme Concentration", value=1.0, format="%.4f")
enz_unit = st.sidebar.selectbox("Units", ["µM", "nM", "M"])

# Convert Enzyme Concentration to µM for consistent kcat calculation
if enz_unit == "nM":
    ENZ_CONC_UM = enz_input / 1000
elif enz_unit == "M":
    ENZ_CONC_UM = enz_input * 1e6
else:
    ENZ_CONC_UM = enz_input

# --- CORE LOGIC FUNCTIONS ---

def load_and_clean_csv(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if "Time (min)" in line:
                start_idx = i
                break
        if start_idx == -1: return None
        data_lines = [lines[start_idx]]
        for line in lines[start_idx+1:]:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    float(parts[0]); data_lines.append(line)
                except ValueError: break
        df = pd.read_csv(StringIO("\n".join(data_lines)))
        df = df.iloc[:, [0, 1]]
        df.columns = ['Time', 'Abs']
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def detect_as_ae(df):
    if len(df) < 60: return 0, len(df) - 1
    abs_vals = df['Abs'].values
    initial_as_idx = 0
    rolling_diff = pd.Series(abs_vals).rolling(window=15).mean().diff()
    for i in range(15, len(rolling_diff) - 20):
        if all(rolling_diff[i:i+10] < 0):
            initial_as_idx = i - 15 + np.argmax(abs_vals[max(0, i-20):i+1])
            break
    initial_ae_idx = len(abs_vals) - 1
    for i in range(initial_as_idx + 50, len(abs_vals) - 50):
        window = abs_vals[i:i+50]
        if np.std(window) < 0.0005 or (abs(window[-1] - window[0]) < 0.001):
            initial_ae_idx = i
            break
    reaction_length = initial_ae_idx - initial_as_idx
    return initial_as