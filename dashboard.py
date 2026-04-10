import streamlit as st
import pandas as pd
import psycopg2
import time
import os

st.set_page_config(page_title="Eagle Vision Analytics", layout="wide")
st.title("🚜 Equipment Utilization Dashboard")

def get_data():
    try:
        conn = psycopg2.connect("host=localhost dbname=eagle_vision user=postgres password=password")
        df = pd.read_sql("SELECT * FROM equipment_logs ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.header("Live Video Feed")
    video_placeholder = st.empty()
    
    st.header("Live Machine Status")
    table_placeholder = st.empty()

with col2:
    st.header("Utilization Analytics")
    chart_place = st.empty()


count = 0
while True:
    # Safety Check for Video Frame
    if os.path.exists("latest_frame.jpg"):
        try:
            # Check if file is not empty before opening
            if os.path.getsize("latest_frame.jpg") > 0:
                video_placeholder.image("latest_frame.jpg", use_container_width=True)
        except Exception:
            # If main.py is currently writing, just skip this frame and try the next one
            pass

    # Update data only every 10 loops to keep things smooth
    if count % 10 == 0:
        df = get_data()
        if not df.empty:
            latest = df.drop_duplicates('equipment_id')
            table_placeholder.table(latest[['equipment_id', 'class', 'state', 'activity', 'timestamp']])
            chart_place.bar_chart(latest.set_index('equipment_id')['utilization_pct'])
    
    count += 1
    time.sleep(0.1)