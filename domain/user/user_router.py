from fastapi import APIRouter, HTTPException
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status

from datetime import timedelta, datetime, timezone

from database import get_db
from domain.user import user_schema, user_crud
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError

from domain.user.user_crud import pwd_context
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60*24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"

router = APIRouter(
    prefix = "/api/user",
)

@router.post("/create", status_code=status.HTTP_204_NO_CONTENT)
def user_create(
    _user_create: user_schema.UserCreate, db: Session = Depends(get_db)):
    user = user_crud.get_existing_user(db, user_create = _user_create)
    if user:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT, detail="이미 존재하는 사용자 입니다.")
    user_crud.create_user(db = db, user_create = _user_create)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl = "/api/user/login")