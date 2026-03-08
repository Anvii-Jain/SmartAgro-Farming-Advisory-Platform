# community_db.py
import sqlite3
import json
from datetime import datetime
import os
import random

DB_PATH = "community.db"

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE,
            location TEXT,
            crops TEXT,
            reputation INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            is_expert BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            description TEXT,
            crop TEXT,
            location TEXT,
            images TEXT,
            priority TEXT DEFAULT 'normal',
            upvotes INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            answer_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create answers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            user_id INTEGER,
            content TEXT NOT NULL,
            upvotes INTEGER DEFAULT 0,
            is_expert_verified BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    print("✅ Tables created successfully")
    
    # Add demo data
    add_demo_data(cursor)
    conn.commit()
    conn.close()
    print("✅ Database ready with 15 questions!")

def add_demo_data(cursor):
    """Add sample users and questions"""
    
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) as count FROM users")
    if cursor.fetchone()[0] > 0:
        print("📊 Demo data already exists")
        return
    
    # Add 6 demo users (4 farmers, 2 experts)
    users = [
        ('Rajesh Kumar', '9876543210', 'Maharashtra', 'Tomato,Wheat', 1250, 5, 0),
        ('Dr. Sharma', '9876543211', 'Delhi', 'All Crops', 5000, 10, 1),
        ('Priya Singh', '9876543212', 'Punjab', 'Wheat,Rice', 800, 3, 0),
        ('Amit Patel', '9876543213', 'Gujarat', 'Cotton,Potato', 600, 2, 0),
        ('Dr. Kaur', '9876543214', 'Punjab', 'Wheat,Maize', 3500, 8, 1),
        ('Sunita Devi', '9876543215', 'Bihar', 'Rice,Maize', 450, 1, 0)
    ]
    
    cursor.executemany('''
        INSERT INTO users (name, phone, location, crops, reputation, level, is_expert)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', users)
    
    # 15 demo questions
    questions = [
        # Tomato questions
        (1, 'How to control bacterial wilt in tomato plants?', 
         'My tomato plants are wilting with yellow leaves. What should I do?',
         'Tomato', 'Maharashtra', 45, 12),
        (3, 'Tomato leaf curling problem', 
         'Leaves are curling upwards. Is this a disease?',
         'Tomato', 'Punjab', 32, 8),
        (4, 'Best fertilizer for tomato yield', 
         'Want to increase tomato production. Which fertilizer is best?',
         'Tomato', 'Gujarat', 28, 6),
        
        # Wheat questions
        (3, 'Best time for wheat sowing in Punjab?', 
         'When should I sow wheat for best yield?',
         'Wheat', 'Punjab', 78, 23),
        (5, 'Wheat rust treatment', 
         'Brown spots on wheat leaves. How to treat?',
         'Wheat', 'Punjab', 56, 14),
        (2, 'Wheat seed rate per acre', 
         'How much seed needed for 1 acre?',
         'Wheat', 'Delhi', 41, 9),
        
        # Rice questions
        (2, 'Rice blast disease treatment', 
         'How to control blast in paddy?',
         'Rice', 'Delhi', 56, 15),
        (6, 'Rice nursery preparation', 
         'Tips for healthy rice nursery?',
         'Rice', 'Bihar', 38, 7),
        (3, 'Yellowing in rice field', 
         'Leaves turning yellow. What deficiency?',
         'Rice', 'Punjab', 29, 5),
        
        # Cotton questions
        (4, 'Organic fertilizer for cotton', 
         'What organic fertilizers work best for cotton?',
         'Cotton', 'Gujarat', 34, 8),
        (1, 'Cotton pest control', 
         'How to control pink bollworm?',
         'Cotton', 'Maharashtra', 52, 11),
        
        # Maize questions
        (5, 'Maize crop water schedule', 
         'How often to irrigate maize?',
         'Maize', 'Punjab', 23, 5),
        (6, 'Maize fertilizer doses', 
         'NPK ratio for maize crop?',
         'Maize', 'Bihar', 19, 4),
        
        # Other crops
        (4, 'Potato blight treatment', 
         'Late blight in potato. What spray?',
         'Potato', 'Gujarat', 44, 10),
        (1, 'Onion thrips control', 
         'Small insects in onion. How to control?',
         'Onion', 'Maharashtra', 37, 8)
    ]
    
    cursor.executemany('''
        INSERT INTO questions 
        (user_id, title, description, crop, location, upvotes, answer_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '-' || ? || ' days'))
    ''', [(q[0], q[1], q[2], q[3], q[4], q[5], q[6], random.randint(1, 5)) for q in questions])
    
    # Get question ids
    question_ids = [row[0] for row in cursor.execute("SELECT id FROM questions").fetchall()]
    
    # 20+ answers (each question gets 1-2 answers)
    answers_data = [
        # Question 1 (Tomato wilt)
        (question_ids[0], 2, 'Use copper oxychloride 0.2%. Remove infected plants. Practice crop rotation for 2 years.', 45, 1),
        (question_ids[0], 4, 'I also had this problem. Neem cake helped reduce it.', 12, 0),
        
        # Question 2 (Tomato leaf curl)
        (question_ids[1], 2, 'Leaf curl is often from whiteflies. Use yellow sticky traps and neem spray.', 28, 1),
        
        # Question 3 (Tomato fertilizer)
        (question_ids[2], 5, 'Use 50:25:25 NPK + FYM 10 tons/acre. Split nitrogen into 3 doses.', 23, 1),
        
        # Question 4 (Wheat sowing)
        (question_ids[3], 5, 'Sow wheat from Oct 20 to Nov 15 in Punjab. Use 100kg seed per acre.', 34, 1),
        (question_ids[3], 3, 'I sow in first week November. Good results.', 15, 0),
        
        # Question 5 (Wheat rust)
        (question_ids[4], 2, 'Spray Propiconazole 0.1% at first sign. Use resistant varieties next season.', 41, 1),
        
        # Question 6 (Wheat seed rate)
        (question_ids[5], 5, '40-45 kg/acre for timely sowing, 50-55 kg for late sowing.', 19, 1),
        
        # Question 7 (Rice blast)
        (question_ids[6], 2, 'Spray tricyclazole 0.06% at panicle initiation. Keep field drained.', 41, 1),
        (question_ids[6], 6, 'I use carbendazim. Works well.', 8, 0),
        
        # Question 8 (Rice nursery)
        (question_ids[7], 2, 'Use 25-30 kg seed per nursery area. Apply DAP 10 days before uprooting.', 22, 1),
        
        # Question 9 (Rice yellowing)
        (question_ids[8], 5, 'Nitrogen deficiency. Apply urea 25 kg/acre.', 17, 1),
        
        # Question 10 (Cotton organic)
        (question_ids[9], 2, 'Use neem cake 200kg/acre + FYM 5 tons/acre. Vermicompost 2 tons/acre.', 23, 1),
        
        # Question 11 (Cotton pest)
        (question_ids[10], 2, 'Use pheromone traps @ 5/acre. Spray neem oil weekly. Chemical only if severe.', 38, 1),
        (question_ids[10], 1, 'I used Profenophos. Worked well.', 11, 0),
        
        # Question 12 (Maize irrigation)
        (question_ids[11], 5, 'Critical stages: knee-high, tasseling, cob formation. Irrigate at these stages.', 15, 1),
        
        # Question 13 (Maize fertilizer)
        (question_ids[12], 5, '80:40:40 NPK. Apply zinc sulfate 25kg/ha if deficient.', 13, 1),
        
        # Question 14 (Potato blight)
        (question_ids[13], 2, 'Spray mancozeb 0.2% or metalaxyl. Remove infected leaves.', 31, 1),
        (question_ids[13], 4, 'I use copper oxychloride as preventive.', 9, 0),
        
        # Question 15 (Onion thrips)
        (question_ids[14], 2, 'Spray spinosad or neem oil. Use sticky traps.', 27, 1)
    ]
    
    for ans in answers_data:
        cursor.execute('''
            INSERT INTO answers 
            (question_id, user_id, content, upvotes, is_expert_verified, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-1 day'))
        ''', ans)
    
    print(f"✅ Demo data added: {len(users)} users, {len(questions)} questions, {len(answers_data)} answers")

# Run this only when file is executed directly
if __name__ == "__main__":
    init_db()
    print("🎉 Database setup complete with 15 questions!")
else:
    # Initialize when imported
    init_db()
