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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl = "/api/user/login")

router = APIRouter(
    prefix = "/api/user",
)

from pydantic import BaseModel

class MessageResponse(BaseModel):
    message: str

@router.post("/create", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def user_create(
    _user_create: user_schema.UserCreate, db: Session = Depends(get_db)):
    user = user_crud.get_existing_user(db, user_create = _user_create)
    if user:
        raise HTTPException(status_code = status.HTTP_409_CONFLICT, detail="이미 존재하는 사용자 입니다.")
    user_crud.create_user(db = db, user_create = _user_create)
    return {"message": "회원가입을 성공하였습니다!"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl = "/api/user/login")

@router.post("/login", response_model = user_schema.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = user_crud.get_user(db, form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.password):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "잘못된 이메일 또는 비밀번호입니다.",
            headers = {"WWW-Authenticate": "Bearer"},
        )
    
    data = {
        "sub": user.email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
        
    }

    access_token = jwt.encode(data, SECRET_KEY, algorithm = ALGORITHM)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user.email,
        "user_id": user.user_id
    }

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail = "유효하지 않은 인증 정보입니다.",
        headers = {"WWW-Authenticate": "Bearer"},
    )
    try: 
        payload = jwt.decode(token, SECRET_KEY, algorithms = [ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    else:
        user = user_crud.get_user(db, username = username)
        if user is None:
            raise credentials_exception
        return user