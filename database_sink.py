import json
import psycopg2
from kafka import KafkaConsumer

# 1. Connect to Postgres
conn = psycopg2.connect("host=localhost dbname=eagle_vision user=postgres password=password")
cur = conn.cursor()

# 2. Create the table
cur.execute("""
    CREATE TABLE IF NOT EXISTS equipment_logs (
        id SERIAL PRIMARY KEY,
        frame_id INT,
        equipment_id TEXT,
        class TEXT,
        timestamp TEXT,
        state TEXT,
        activity TEXT,
        utilization_pct FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
conn.commit()

# 3. Consume from Kafka
consumer = KafkaConsumer('equipment_events', bootstrap_servers=['localhost:9092'])

print("[INFO] Database Sink active. Waiting for events...")

for msg in consumer:
    data = json.loads(msg.value.decode('utf-8'))
    cur.execute("""
        INSERT INTO equipment_logs (frame_id, equipment_id, class, timestamp, state, activity, utilization_pct)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        data['frame_id'], 
        data['equipment_id'], 
        data['equipment_class'].upper(), 
        data['timestamp'],
        data['utilization']['current_state'],
        data['utilization']['current_activity'],
        data['time_analytics']['utilization_percent']
    ))
    conn.commit()