import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="BTP Uniformity Yield", layout="wide")

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
DB_FILE = "latest/db.csv"
MICRO_FILE = "latest/micro.csv"
ZF_FILE = "latest/zf.csv"
VTUODATA_FILE = "latest/vtuodata.csv"

# Set dynamic dates for the last 8 days
END_DATE_DT = datetime.now()
START_DATE_DT = END_DATE_DT - timedelta(days=7)

START_DATE = START_DATE_DT.strftime("%Y-%m-%d")
END_DATE = END_DATE_DT.strftime("%Y-%m-%d")

# DB Columns
DB_UNIFORMITY_COLS = ['GradeRRO', 'GradeUpperLRO', 'GradeLowerLRO', 'GradeUpperBulge', 'GradeLowerBulge', 'GradeUpperDepression', 'GradeLowerDepression']
DB_BALANCING_COLS = ['StaticGrade', 'UpperGrade', 'LowerGrade', 'CoupleGrade']
DB_WCID_MAPPING = {280: 'TUO9907', 279: 'TUO9908', 278: 'Micropoise'}

# Micro Columns
MICRO_UNIFORMITY_COLS = [
    'GradeRFVCW', 'GradeH1RFVCW', 'GradeLFVCW', 'GradeRFVCCW',
    'GradeH1RFVCCW', 'GradeLFVCCW', 'GradeCONICITY'
]

# ZF Columns
ZF_UNIFORMITY_COLS = ['GradeRFVCW', 'GradeH1RFVCW', 'GradeLFVCW', 'GradeRFVCCW', 'GradeH1RFVCCW', 'GradeLFVCCW', 'GradeCONICITY', 'GradeRRO', 'GradeUpperLRO', 'GradeLowerLRO', 'GradeUpperBulge', 'GradeLowerBulge', 'GradeUpperDepression', 'GradeLowerDepression']
ZF_BALANCING_COLS = ['StaticGrade', 'UpperGrade', 'LowerGrade', 'CoupleGrade']

# ==========================================
# DATA LOADING (CACHED)
# ==========================================

@st.cache_data(ttl=36000)
def load_db_data(file_path, start_date, end_date):
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path, low_memory=False)
    df['dtandTime'] = pd.to_datetime(df['dtandTime'], errors='coerce')
    df = df.dropna(subset=['dtandTime', 'BARCODE'])
    df = df[df['BARCODE'].astype(str).str.strip() != '']
    df['ProductionDate'] = (df['dtandTime'] - pd.Timedelta(hours=7)).dt.date
    df = df.sort_values('dtandTime').drop_duplicates(subset=['ProductionDate', 'BARCODE'], keep="last").reset_index(drop=True)
    
    target_dates = pd.date_range(start=start_date, end=end_date).date
    df = df[df['ProductionDate'].isin(target_dates)].copy()
    
    df['Machine'] = df['wcID'].map(DB_WCID_MAPPING)
    df = df.dropna(subset=['Machine'])
    
    for col in DB_BALANCING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['Balancing_OE'] = df[DB_BALANCING_COLS].isin([0, 1, '0', '1', 'A', 'B', 'a', 'b']).all(axis=1)
    df['Balancing_Only_0'] = df[DB_BALANCING_COLS].fillna(0).isin([0, '0', 'A', 'a']).all(axis=1)
    df['Balancing_D'] = df[DB_BALANCING_COLS].isin(['D', 'd']).any(axis=1)
    
    agg_funcs = {'BARCODE': 'count', 'Balancing_OE': 'sum', 'Balancing_Only_0': 'sum', 'Balancing_D': 'sum'}
    results = df.groupby(['Machine', 'ProductionDate']).agg(agg_funcs).reset_index()
    results.rename(columns={'BARCODE': 'Total_Count', 'ProductionDate': 'Date'}, inplace=True)
    return results

@st.cache_data(ttl=36000)
def load_micro_data(file_path, start_date, end_date):
    # Cache busted

    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path, low_memory=False)
    df['TestTime'] = pd.to_datetime(df['TestTime'], format='mixed', dayfirst=True)
    df = df.dropna(subset=['BARCODE'])
    df = df[df['BARCODE'].astype(str).str.strip() != '']
    df['ShiftDate'] = (df['TestTime'] - pd.Timedelta(hours=7)).dt.date
    df = df.sort_values('TestTime').drop_duplicates(subset=['ShiftDate', 'BARCODE'], keep="last").reset_index(drop=True)
    
    target_dates = pd.date_range(start=start_date, end=end_date).date
    df_filtered = df[df['ShiftDate'].isin(target_dates)].copy()
    
    if df_filtered.empty:
        df['MonthDay'] = df['ShiftDate'].apply(lambda d: (d.month, d.day))
        target_md = [(d.month, d.day) for d in target_dates]
        df_filtered = df[df['MonthDay'].isin(target_md)].copy()
        
    df_filtered[MICRO_UNIFORMITY_COLS] = df_filtered[MICRO_UNIFORMITY_COLS].fillna('NONE')
    df_filtered['Uniformity_OE'] = df_filtered[MICRO_UNIFORMITY_COLS].isin(['A', 'B']).all(axis=1)
    df_filtered['Uniformity_Only_0'] = df_filtered[MICRO_UNIFORMITY_COLS].isin(['A']).all(axis=1)
    df_filtered['Uniformity_D'] = df_filtered[MICRO_UNIFORMITY_COLS].isin(['D']).any(axis=1)
    
    report = df_filtered.groupby(['MachineName', 'ShiftDate']).agg(
        Total_Count=('BARCODE', 'count'),
        Uniformity_OE=('Uniformity_OE', 'sum'),
        Uniformity_Only_0=('Uniformity_Only_0', 'sum'),
        Uniformity_D=('Uniformity_D', 'sum')
    ).reset_index()
    report.rename(columns={'ShiftDate': 'Date', 'MachineName': 'Machine'}, inplace=True)
    return report

@st.cache_data(ttl=36000)
def load_zf_data(file_path, start_date, end_date):
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path, low_memory=False)
    df['TestTime'] = pd.to_datetime(df['TestTime'], errors='coerce')
    df = df.dropna(subset=['TestTime', 'BARCODE'])
    df = df[df['BARCODE'].astype(str).str.strip() != '']
    df['ProductionDate'] = (df['TestTime'] - pd.Timedelta(hours=7)).dt.date
    df = df.sort_values('TestTime').drop_duplicates(subset=['ProductionDate', 'BARCODE'], keep="last").reset_index(drop=True)
    
    target_dates = pd.date_range(start=start_date, end=end_date).date
    df = df[df['ProductionDate'].isin(target_dates)].copy()
    
    oe_values = ['A', 'B', 'a', 'b', 0, 1, '0', '1']
    only_a_values = ['A', 'a', 0, '0']
    
    df['Uniformity_OE'] = df[ZF_UNIFORMITY_COLS].isin(oe_values).all(axis=1)
    df['Uniformity_Only_0'] = df[ZF_UNIFORMITY_COLS].isin(only_a_values).all(axis=1)
    df['Uniformity_D'] = df[ZF_UNIFORMITY_COLS].isin(['D', 'd']).any(axis=1)
    
    df['Balancing_OE'] = df[ZF_BALANCING_COLS].isin(oe_values).all(axis=1)
    df['Balancing_Only_0'] = df[ZF_BALANCING_COLS].isin(only_a_values).all(axis=1)
    df['Balancing_D'] = df[ZF_BALANCING_COLS].isin(['D', 'd']).any(axis=1)
    
    agg_funcs = {
        'BARCODE': 'count', 
        'Uniformity_OE': 'sum', 'Uniformity_Only_0': 'sum', 'Uniformity_D': 'sum',
        'Balancing_OE': 'sum', 'Balancing_Only_0': 'sum', 'Balancing_D': 'sum'
    }
    results = df.groupby(['MachineName', 'ProductionDate']).agg(agg_funcs).reset_index()
    results.rename(columns={'BARCODE': 'Total_Count', 'ProductionDate': 'Date', 'MachineName': 'Machine'}, inplace=True)
    return results

@st.cache_data(ttl=36000)
def load_vtuodata(file_path, start_date, end_date):
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path, low_memory=False)
    
    df['dtandTime'] = pd.to_datetime(df['dtandTime'], errors='coerce')
    df = df.dropna(subset=['dtandTime', 'gtBarcode'])
    df = df[df['gtBarcode'].astype(str).str.strip() != '']
    
    df['ProductionDate'] = (df['dtandTime'] - pd.Timedelta(hours=7)).dt.date
    df = df.sort_values('dtandTime').drop_duplicates(subset=['ProductionDate', 'gtBarcode'], keep="last").reset_index(drop=True)
    
    target_dates = pd.date_range(start=start_date, end=end_date).date
    df = df[df['ProductionDate'].isin(target_dates)].copy()
    
    df['statusName'] = df['statusName'].astype(str).str.strip().str.upper()
    df = df[df['statusName'].isin(['OE', 'NON OE'])].copy()
    
    df['vTUODATA_OE'] = (df['statusName'] == 'OE')
    
    # Extract machine from wcName (e.g., TUO-9903 -> TUO9903)
    df['MachineName'] = df['wcName'].astype(str).str.replace('-', '', regex=False).str.strip()
    
    agg_funcs = {'gtBarcode': 'count', 'vTUODATA_OE': 'sum'}
    results = df.groupby(['MachineName', 'ProductionDate']).agg(agg_funcs).reset_index()
    results.rename(columns={'gtBarcode': 'Total_Count', 'ProductionDate': 'Date', 'MachineName': 'Machine'}, inplace=True)
    
    # Map vTUODATA_OE to Uniformity_OE so it integrates seamlessly with the individual machine charts
    results['Uniformity_OE'] = results['vTUODATA_OE']
    results['Uniformity_Only_0'] = 0
    results['Uniformity_D'] = 0
    
    return results

@st.cache_data(ttl=36000)
def get_combined_data():
    db_df = load_db_data(DB_FILE, START_DATE, END_DATE) # Ready for Balancing dashboard
    micro_df = load_micro_data(MICRO_FILE, START_DATE, END_DATE)
    micro_df['Source'] = 'Micro'
    
    zf_df = load_zf_data(ZF_FILE, START_DATE, END_DATE)
    zf_df['Source'] = 'ZF'
    
    vtuo_df = load_vtuodata(VTUODATA_FILE, START_DATE, END_DATE)
    vtuo_df['Source'] = 'vTUODATA'
    
    # Include vTUODATA so machines 9903 and 9904 appear in the dashboard lists
    combined = pd.concat([micro_df, zf_df, vtuo_df], ignore_index=True)
    
    # Global replacement for machine names
    combined['Machine'] = combined['Machine'].replace({
        '30-0204': 'TUO9908',
        '30-0224': 'TUO9908',
        '30-0297': 'TUO9907'
    })
    
    return combined

# ==========================================
# UI & DASHBOARD
# ==========================================

# --- Global Styles ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Remove default padding */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Page background */
    .stApp {
        background-color: #eef2f7;
    }

    /* Full-width dark header bar */
    .dash-header {
        background: #1c1c4f;
        color: #fff;
        padding: 12px 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: -1rem -1rem 12px -1rem;
    }
    .dash-header h1 {
        margin: 0;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* Card styling for the 3 top columns */
    div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }

    /* Prevent nested columns from inheriting card style */
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        background-color: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
        border-radius: 0;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}

    /* Compact selectbox labels */
    .stSelectbox label {
        font-weight: 600;
        font-size: 13px;
        color: #475569;
        margin-bottom: 2px;
    }
    .stSelectbox > div > div {
        min-height: 34px;
    }
</style>

<div class="dash-header">
    <h1>BTP Uniformity Yield - Dashboard</h1>
</div>
""", unsafe_allow_html=True)

df = get_combined_data()

if df.empty:
    st.error("No data available in the latest folder.")
    st.stop()

# Date string for plotting
df['Date_Str'] = pd.to_datetime(df['Date']).dt.strftime('%d-%b')

# ==========================================
# CHART HELPER
# ==========================================
CHART_FONT = dict(family="Inter, sans-serif", color="#334155")

def make_bar_chart(data, x_col, y_col, bar_color='#38bdf8', text_fmt='.1f', suffix='%', bar_colors=None, y_title=None, hover_text=None):
    text_vals = [f"{v:{text_fmt}}{suffix}" if suffix else f"{v:{text_fmt}}" for v in data[y_col]]

    bar_kwargs = dict(
        x=data[x_col],
        y=data[y_col],
        text=text_vals,
        textposition='outside',
        textfont=dict(size=11, color='#1e293b', family="Inter", weight='bold' if suffix == '%' else 'normal'),
        marker=dict(
            color=bar_colors if bar_colors else bar_color,
            line=dict(width=0),
            cornerradius=3,
        ),
        opacity=0.92,
    )
    
    if hover_text is not None:
        bar_kwargs['hovertext'] = hover_text
        bar_kwargs['hoverinfo'] = 'x+y+text'

    fig = go.Figure(data=[go.Bar(**bar_kwargs)])

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=CHART_FONT,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color='#64748b')),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color='#475569')) if y_title else None,
            showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
            zeroline=True, zerolinecolor='#e2e8f0',
            tickfont=dict(size=10, color='#94a3b8'),
        ),
        margin=dict(t=20, l=45, r=10, b=50),
        height=280,
        bargap=0.25,
    )

    if suffix == '%':
        fig.update_yaxes(range=[0, 115])
    else:
        max_val = data[y_col].max() if not data.empty else 0
        fig.update_yaxes(range=[0, (max_val * 1.15) if max_val > 0 else 10])

    return fig

# ==========================================
# ROW 1: THREE CHART CARDS
# ==========================================
colA, colB, colC = st.columns(3, gap="medium")

# --- SECTION 1: Overall Production Yield % ---
with colA:
    h1, h2, h3 = st.columns([1.6, 1, 1.2])
    with h1:
        st.markdown("<p style='color:#4338ca; font-size:15px; font-weight:700; margin:0; line-height:1.3;'>Overall Production<br>Yield %</p>", unsafe_allow_html=True)
    with h2:
        s1_grade = st.selectbox("Grade:", options=["A+B", "Only A / 0", "D Grade"], key="s1_grade")
    with h3:
        all_machines = sorted(df['Machine'].unique().tolist())
        s1_machine = st.selectbox("Machine:", options=["ALL"] + all_machines, index=0, key="s1_machine")

    st.markdown("<hr style='margin:6px 0 4px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    s1_df = df.copy()
    if s1_machine != "ALL":
        s1_df = s1_df[s1_df['Machine'] == s1_machine]
        
    if s1_grade == "A+B":
        s1_col = "Uniformity_OE"
    elif s1_grade == "Only A / 0":
        s1_col = "Uniformity_Only_0"
    else:
        s1_col = "Uniformity_D"

    if s1_machine == "ALL":
        db_df = load_db_data(DB_FILE, START_DATE, END_DATE)
        zf_df = load_zf_data(ZF_FILE, START_DATE, END_DATE)
        micro_df = load_micro_data(MICRO_FILE, START_DATE, END_DATE)
        vtuo_df = load_vtuodata(VTUODATA_FILE, START_DATE, END_DATE)
        
        dates = pd.date_range(start=START_DATE, end=END_DATE).date
        master_agg = pd.DataFrame({'Date': dates})
        master_agg['Total'] = 0.0
        master_agg['Target'] = 0.0
        
        for d in dates:
            d_total = 0
            d_target = 0
            
            # 1. Micro
            m = micro_df[micro_df['Date'] == d]
            if not m.empty:
                d_total += m['Total_Count'].sum()
                if s1_grade == "A+B": d_target += m['Uniformity_OE'].sum()
                elif s1_grade == "Only A / 0": d_target += m['Uniformity_Only_0'].sum()
                else: d_target += m['Uniformity_D'].sum()
                
            # 2 & 3. ZF (Uniformity and Balancing as separate events)
            z = zf_df[zf_df['Date'] == d]
            if not z.empty:
                d_total += z['Total_Count'].sum() * 2 # Adds to total twice (once for Unif, once for Bal)
                
                # Uniformity Passes
                if s1_grade == "A+B": d_target += z['Uniformity_OE'].sum()
                elif s1_grade == "Only A / 0": d_target += z['Uniformity_Only_0'].sum()
                else: d_target += z['Uniformity_D'].sum()
                
                # Balancing Passes
                if s1_grade == "A+B": d_target += z['Balancing_OE'].sum()
                elif s1_grade == "Only A / 0": d_target += z['Balancing_Only_0'].sum()
                else: d_target += z['Balancing_D'].sum()
                
            # 4. DB
            db_d = db_df[db_df['Date'] == d]
            if not db_d.empty:
                d_total += db_d['Total_Count'].sum()
                if s1_grade == "A+B": d_target += db_d['Balancing_OE'].sum()
                elif s1_grade == "Only A / 0": d_target += db_d['Balancing_Only_0'].sum()
                else: d_target += db_d['Balancing_D'].sum()
                
            # 5. vTUODATA
            if s1_grade == "A+B":
                v = vtuo_df[vtuo_df['Date'] == d]
                if not v.empty:
                    d_total += v['Total_Count'].sum()
                    d_target += v['vTUODATA_OE'].sum()
                    
            master_agg.loc[master_agg['Date'] == d, 'Total'] = d_total
            master_agg.loc[master_agg['Date'] == d, 'Target'] = d_target
            
        s1_agg = master_agg[master_agg['Total'] > 0].copy()
        s1_agg['Date_Str'] = pd.to_datetime(s1_agg['Date']).dt.strftime('%d-%b')
        s1_agg['Yield %'] = (s1_agg['Target'] / s1_agg['Total']) * 100
    else:
        s1_agg = s1_df.groupby('Date_Str').agg(Total=('Total_Count', 'sum'), Target=(s1_col, 'sum')).reset_index()
        s1_agg['Yield %'] = (s1_agg['Target'] / s1_agg['Total']) * 100

    hover_texts = [
        f"Passed: {int(row['Target'])}<br>Tested: {int(row['Total'])}"
        for _, row in s1_agg.iterrows()
    ]
    fig1 = make_bar_chart(s1_agg, 'Date_Str', 'Yield %', bar_color='#22c55e', text_fmt='.1f', suffix='%', y_title='Yield %', hover_text=hover_texts)
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

# --- SECTION 2: Uniformity Machine Wise Production ---
with colB:
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown("<p style='color:#4338ca; font-size:15px; font-weight:700; margin:0; line-height:1.3;'>Uniformity Machine Wise<br>Production</p>", unsafe_allow_html=True)
    with h2:
        all_machines = sorted(df['Machine'].unique().tolist())
        s2_machine = st.selectbox("Machine:", options=["ALL"] + all_machines, index=0, key="s2_machine")

    st.markdown("<hr style='margin:6px 0 4px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    s2_df = df.copy()
    if s2_machine != "ALL":
        s2_df = s2_df[s2_df['Machine'] == s2_machine]

    s2_agg = s2_df.groupby('Date_Str').agg(Total=('Total_Count', 'sum')).reset_index()

    fig2 = make_bar_chart(s2_agg, 'Date_Str', 'Total', bar_color='#3b82f6', text_fmt='.0f', suffix='', y_title='Production')
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# --- SECTION 3: Uniformity Machine Wise Yield % ---
with colC:
    h1, h2 = st.columns([1.6, 1])
    with h1:
        st.markdown("<p style='color:#4338ca; font-size:15px; font-weight:700; margin:0; line-height:1.3;'>Uniformity Machine Wise<br>Yield %</p>", unsafe_allow_html=True)
    with h2:
        available_dates = sorted(df['Date'].unique(), reverse=True)
        s3_date = st.selectbox("Date:", options=available_dates, format_func=lambda x: pd.to_datetime(x).strftime('%d-%b'), key="s3_date")

    st.markdown("<hr style='margin:6px 0 4px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    s3_df = df[df['Date'] == s3_date].copy()

    s3_agg = s3_df.groupby('Machine').agg(Total=('Total_Count', 'sum'), Target=('Uniformity_OE', 'sum')).reset_index()
    s3_agg['Yield %'] = (s3_agg['Target'] / s3_agg['Total']) * 100
    
    # Apply requested sorting order exactly matching the screenshot
    custom_order = ["ZF-6", "ZF-4", "ZF-1", "Micropoise", "TUO9907", "ZF-8", "ZF-5", "ZF-2", "TUO9908", "ZF-7", "ZF-3"]
    order_map = {m: i for i, m in enumerate(custom_order)}
    s3_agg['sort_idx'] = s3_agg['Machine'].map(lambda x: order_map.get(x, 999))
    s3_agg = s3_agg.sort_values(['sort_idx', 'Machine']).reset_index(drop=True)

    colors = ['#f59e0b' if v < 95 else '#22c55e' for v in s3_agg['Yield %']]
    fig3 = make_bar_chart(s3_agg, 'Machine', 'Yield %', text_fmt='.1f', suffix='%', bar_colors=colors, y_title='Yield %')
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

