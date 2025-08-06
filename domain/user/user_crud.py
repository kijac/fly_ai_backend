from sqlalchemy.orm import Session
from domain.user.user_schema import UserCreate
from model import User
from passlib.context import CryptContext
import datetime

pwd_context = CryptContext(schemes = ["bcrypt"], deprecated = "auto")

def create_user(db: Session, user_create: UserCreate):
    db_user = User(name = user_create.name,
                   email = user_create.email,
                   password = pwd_context.hash(user_create.password1),
                   address = user_create.address,
                   phone_number = user_create.phone_number,
                   role = user_create.role,
                   created_at=datetime.datetime.now())
    db.add(db_user)
    db.commit()

def get_existing_user(db: Session, user_create: UserCreate):
    return db.query(User).filter((User.email == user_create.email)).first()
