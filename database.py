from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# MySQL 데이터베이스 연결 URL
# 환경변수에서 값을 가져오거나 기본값 사용
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "myapi_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}
# naming_convention은 테이블과 컬럼의 이름 규칙을 정의하는 딕셔너리이다.
# 이 규칙은 SQLAlchemy가 테이블과 컬럼을 생성할 때 이름을 지정하는 데 사용된다.
# 예를 들어, "ix"는 인덱스의 접두사로 사용되며, "uq"는 유니크 제약 조건의 접두사로 사용된다.

# Base를 만들 때 MetaData에 naming_convention을 바로 적용
metadata = MetaData(naming_convention=naming_convention)
Base = declarative_base(metadata=metadata)