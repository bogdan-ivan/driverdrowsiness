import sqlite3
newUser = True

sql_create_table_user = """CREATE TABLE IF NOT EXISTS users (
                                  id integer PRIMARY KEY AUTOINCREMENT,
                                  name text ,
                                  ear float);"""

"""sql_create_features_table = CREATE TABLE IF NOT EXISTS features (
                                  id integer PRIMARY KEY AUTOINCREMENT,
                                  avgEar real,
                                  avgMar real,
                                  yawn integer,
                                  yawn_rate integer,
                                  t_b_yawns real,
                                  blinkRate real,
                                  blinkAmplitude real,
                                  blinkDuration real,
                                  t_b_blinks real,
                                  avgRoll real,
                                  avgYaw real,
                                  avgPitch real);                              
                                  """

sql_create_features_table = """CREATE TABLE IF NOT EXISTS features (
                                id integer PRIMARY KEY AUTOINCREMENT,
                                user_id integer,
                                vigilant integer,
                                low_vigilant integer,
                                drowsy integer,
                                data integer, 
                                time integer,
                                vld_mean float,
                                CONSTRAINT fk_user_id
                                    FOREIGN KEY (user_id)
                                    REFERENCES users(id)
                                    );
                                """
"""conn = create_connection(database)
cursor = conn.cursor()
if conn is not None:
    create_table(conn,sql_create_table_user)
    create_table(conn,sql_create_features_table)
else:
    print("Can't open the database")"""

#user = input("Insert your name\n")

#cursor.execute("""INSERT INTO users(name,password) VALUES ('Lucian', '123')""");

class DataBase :
    def __init__(self):
        self.connect= self.create_connection("user_database.sqlite")
        self.cursor =self.connect.cursor()


    def create_table(self, create_table_sql):
        try:
            self.cursor.execute(create_table_sql)
        except Error as e:
            print(e)

    def create_connection(self,db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
        return conn
    def drop(self):
        self.cursor.execute("""Drop TABLE users""")


    def select_name(self,user):
        self.cursor.execute("""SELECT name 
                      FROM users
                      WHERE name LIKE '%s' """ % user)
        check = self.cursor.fetchone()
        if check:
            print("Found in database")
            return True
        else:
            print("Not found in the database")
            return False

    def insert_name(self, user):
        self.cursor.execute("""Insert into users (name) values('%s')""" % user)
        self.connect.commit()

    def insert_ear(self,ear,name):
        self.cursor.execute("""UPDATE users SET ear = ?  WHERE name = ? """, (ear,name) )
        self.connect.commit()
        return True
    def get_user_id(self,name):
        self.cursor.execute("""Select id from users where name ='%s' """ % name)
        return self.cursor.fetchone()[0]

    def get_ear(self,name):
        self.cursor.execute("""Select ear from users where name ='%s' """%name)
        return self.cursor.fetchone()[0]

    def execute_query(self,query,parameters):
        self.cursor.execute(query %parameters)
        print("Query executed succesfuly !")
        self.connect.commit()

    def update_features(self,user_id,param_tuple):
        param_tuple=(user_id,)+param_tuple
        self.cursor.execute("INSERT INTO features (user_id,vigilant,low_vigilant,drowsy,time,vld_mean,data) values(?,?,?,?,?,?,?)",param_tuple)
        self.connect.commit()