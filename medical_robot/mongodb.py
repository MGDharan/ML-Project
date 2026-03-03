from pymongo import MongoClient
from pymongo.errors import PyMongoError

MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME = "quiz_battle"

seed_questions = [
    {"q": "Capital of India?", "a": "Delhi"},
    {"q": "5 + 5?", "a": "10"},
    {"q": "Sun rises from?", "a": "East"},
    {"q": "2 * 6?", "a": "12"},
    {"q": "Water formula?", "a": "H2O"},
    {"q": "Largest planet?", "a": "Jupiter"},
    {"q": "HTML stands for?", "a": "Hypertext Markup Language"},
    {"q": "CSS used for?", "a": "Styling"},
    {"q": "JS is?", "a": "Programming Language"},
    {"q": "Binary of 2?", "a": "10"},
    {"q": "7 + 8?", "a": "15"},
    {"q": "Opposite of hot?", "a": "Cold"},
    {"q": "Earth shape?", "a": "Sphere"},
    {"q": "Fastest land animal?", "a": "Cheetah"},
    {"q": "1 byte = ?", "a": "8 bits"},
]


def connect_and_seed() -> None:
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        users = db["users"]
        questions = db["questions"]
        answers = db["answers"]

        users.create_index("email", unique=True)
        users.create_index("playerId", unique=True)

        if questions.count_documents({}) == 0:
            questions.insert_many(seed_questions)

        print("MongoDB connected successfully")
        print(f"Database: {DB_NAME}")
        print(f"Collections ready: users={users.count_documents({})}, "
              f"questions={questions.count_documents({})}, "
              f"answers={answers.count_documents({})}")
    except PyMongoError as exc:
        print(f"MongoDB error: {exc}")


if __name__ == "__main__":
    connect_and_seed()
