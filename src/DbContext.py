import psycopg2
import uuid
import os

class DbContext():

    def __init__(self):
        """
        Initialize database cursor.
        """

        self._conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )

        self._cur = self._conn.cursor()

    def create_conversation(self, user_id: str = None):
        """
        Add conversation record to database.

        args:
            uder_id (str): Username to associate with conversation record.
        
        returns (int): ID of new conversation record.
        """
        query = """
        INSERT INTO Convos (id, user_id)
        VALUES (%s, %s)
        RETURNING id;
        """
        convo_id = str(uuid.uuid4())
        self._cur.execute(query, (convo_id, user_id))
        self._conn.commit()
        return convo_id
    
    def get_conversation_by_id(self, convo_id: str):
        """
        Get conversation record from databased by ID.

        args:
            convo_id (str): ID of conversation to retrieve.
        
        returns (any): Conversation record.
        """
        if not self._is_valid_uuid(convo_id):
            raise ValueError("invalid UUID")

        query = """
        SELECT id, user_id, created_at
        FROM Convos
        WHERE id = %s;
        """
        self._cur.execute(query, (convo_id,))
        return self._cur.fetchone()
    
    def add_message(self, convo_id: str, is_system: bool, message: str, user_id: str = None, subjects: list[str] = None, used_prev_subjects: bool = False):
        """
        Add message record to database.

        args:
            convo_id (str): ID of object to retrieve.
            is_system (bool): Boolean denoting whether the message is AI-generated.
            message (str): Message body.
            user_id (str): Username of message sender.
            subjects (list[str]): List of subjects (to assist context).
            used_prev_subjects (bool): Whether last message's subjects were used in message
            generation (only possible/relevant if AI generated the message),
        
        returns (any): Message record.
        """
        query = """
        INSERT INTO Msgs (convo_id, is_system, user_id, message, subjects, used_prev_subjects)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, created_at;
        """
        self._cur.execute(query, (convo_id, is_system, user_id, message, subjects, used_prev_subjects))
        self._conn.commit()
        return self._cur.fetchone()
    
    def get_recent_exchanges(self, convo_id: str, k: int):
        """
        Get k most recent transactions in conversation.

        args:
            convo_id (int): ID of conversation from which to retrieve messages.
        
        returns (any[]): Message records.
        """
        query = """
        SELECT id, is_system, user_id, message, created_at, subjects
        FROM Msgs
        WHERE convo_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        self._cur.execute(query, (convo_id, k*2)) # multiply by 2 because each exchange is comprised of 2 messages
        results = self._cur.fetchall()
        return results[::-1] # reverse to be from earliest to latest
    
    def close(self):
        """
        Close database cursor.
        """
        self._cur.close()
        self._conn.close()

    def initialize_db(self):
        """
        Commands to create database tables/entities.
        """
        convos_table_command = """
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        CREATE TABLE Convos (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        messages_table_command = """
        CREATE TABLE Msgs (
            id SERIAL PRIMARY KEY,
            convo_id UUID REFERENCES Convos(id) ON DELETE CASCADE,
            is_system BOOLEAN NOT NULL,
            user_id TEXT,
            message TEXT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            subjects TEXT[],
            used_prev_subjects BOOLEAN NOT NULL
        );
        """

        self._cur.execute(convos_table_command)
        self._cur.execute(messages_table_command)
        self._conn.commit()

    def _is_valid_uuid(self, val):
        """
        Check if UUID is valid

        args:
            val (any): Database UUID to check.
        
        returns (bool): Validity of specified UUID.
        """
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False