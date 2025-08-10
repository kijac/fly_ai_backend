from pydantic import BaseModel, EmailStr
import datetime
from model import UserRole  # model.py의 UserRole enum import
from pydantic import BaseModel, field_validator, EmailStr
from pydantic_core.core_schema import FieldValidationInfo

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password1: str
    password2: str
    address: str
    phone_number: str
    role: UserRole

    @field_validator('name', 'email', 'password1', 'password2', 'address', 'phone_number', 'role')
    def not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v
    
    @field_validator('password2')
    def passwords_match(cls, v, info: FieldValidationInfo):
        if 'password1' in info.data and v != info.data['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다.')
        return v
    

class Token(BaseModel):
    access_token: str
    token_type: str
    email: EmailStr
    user_id: int


class UserResponse(BaseModel):
    user_id: int
    name: str
    email: EmailStr
    address: str | None
    role: UserRole
    phone_number: str | None
    created_at: datetime.datetime

    class Config:
        from_attributes = True