import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('violence_events.db')

# Create a cursor object to execute SQL commands
c = conn.cursor()

# Create a table to store detected events
c.execute('''CREATE TABLE IF NOT EXISTS events
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_type TEXT,
              timestamp TEXT,
              frame_path TEXT,
              confidence REAL)''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")