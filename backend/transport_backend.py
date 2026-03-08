from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

# -------------------------
# Create database & table
# -------------------------
def init_db():
    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
CREATE TABLE IF NOT EXISTS transport_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop TEXT,
    quantity INTEGER,
    packaging TEXT,
    price INTEGER,
    pickup_location TEXT,
    destination TEXT,
    transport_date TEXT,
    contact TEXT,
    transport_type TEXT,
    status TEXT DEFAULT 'Pending',
    driver TEXT
)
""")

    conn.commit()
    conn.close()

init_db()

# -------------------------
# Store transport request
# -------------------------
@app.route('/transport-request', methods=['POST'])
def transport_request():

    data = request.json

    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO transport_requests
    (crop, quantity, packaging, price, pickup_location, destination, transport_date, contact, transport_type)
    VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        data['crop'],
        data['quantity'],
        data['packaging'],
        data['price'],
        data['pickup'],
        data['destination'],
        data['date'],
        data['contact'],
        data['transport_type']
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "Transport request saved successfully"})


# -------------------------
# Get recent requests
# -------------------------
@app.route('/transport-requests', methods=['GET'])
def get_requests():

    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT crop, pickup_location, destination, status, driver
    FROM transport_requests
    ORDER BY id DESC
    LIMIT 5
    """)

    rows = cursor.fetchall()

    data = []

    for row in rows:
        data.append({
            "crop": row[0],
            "from": row[1],
            "to": row[2],
            "status": row[3],
            "driver": row[4]      
        })

    conn.close()

    return jsonify(data)

@app.route('/pending-requests', methods=['GET'])
def pending_requests():

    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, crop, pickup_location, destination, contact
    FROM transport_requests
    WHERE status='Pending'
    """)

    rows = cursor.fetchall()

    data = []

    for row in rows:
        data.append({
            "id": row[0],
            "crop": row[1],
            "from": row[2],
            "to": row[3],
            "phone": row[4]
        })

    conn.close()

    return jsonify(data)

@app.route('/accept-request/<int:req_id>', methods=['POST'])
def accept_request(req_id):

    data = request.json
    driver_name = data['driver']

    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE transport_requests
    SET status='In Transit', driver=?
    WHERE id=?
    """,(driver_name,req_id))

    conn.commit()
    conn.close()

    return jsonify({"message":"Request accepted"})

@app.route('/driver-requests', methods=['GET'])
def driver_requests():

    conn = sqlite3.connect("transport.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT crop, pickup_location, destination, status, contact
    FROM transport_requests
    WHERE driver='Driver1'
    """)

    rows = cursor.fetchall()

    data = []

    for row in rows:
        data.append({
            "crop": row[0],
            "from": row[1],
            "to": row[2],
            "status": row[3],
             "phone": row[4]
        })

    conn.close()

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
