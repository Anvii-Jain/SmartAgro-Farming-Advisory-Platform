# update_db.py
import sqlite3
import os

# Path to your database
db_path = os.path.join(os.path.dirname(__file__), 'community.db')

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Add images column to questions table
    cursor.execute("ALTER TABLE questions ADD COLUMN images TEXT")
    print("✅ Successfully added 'images' column to questions table")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("⚠️ Column 'images' already exists")
    else:
        print(f"❌ Error: {e}")

# Show table schema
cursor.execute("PRAGMA table_info(questions)")
columns = cursor.fetchall()
print("\n📊 Current columns in questions table:")
for col in columns:
    print(f"  {col[1]} - {col[2]}")

# Commit and close
conn.commit()
conn.close()
