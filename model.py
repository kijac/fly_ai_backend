from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table, BigInteger, Boolean, Enum,JSON
from sqlalchemy.orm import relationship
from database import Base
import enum

# 사용자 역할 정의
class UserRole(enum.Enum):
    ADMIN = "admin"
    USER = "user"

# 기부 가능 여부 ENUM
class DonationStatus(enum.Enum):
    IMPOSSIBLE = "impossible"
    RECYCLABLE = "recyclable"
    UPCYCLE = "upcycle"

# 기부자 상태 ENUM
class ToyStatus(enum.Enum):
    PENDING = "pending"
    FOR_SALE = "for_sale"
    REJECTED = "rejected"
    DONATION = "donation"
    END_BUY = "end_buy"
    SUBMIT = "submit"

# 수령자 상태 ENUM
class RecipientStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

# 배송 상태 ENUM
class DeliveryStatus(enum.Enum):
    PREPARING = "preparing"
    ONGOING = "ongoing"
    COMPLETED = "completed"

class User(Base):
    __tablename__ = "user"

    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(20), nullable=False)
    email = Column(String(255), unique = True, nullable = False)
    password = Column(String(255), nullable=False)
    address = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    phone_number = Column(String(20), nullable=True)
    created_at = Column(DateTime, nullable=False)

    points = Column(BigInteger, nullable=False, default=0)  # 개인 포인트
    subscribe = Column(Boolean, nullable=True, default=False)  # 구독 여부

class Toy_Stock(Base):
    __tablename__ = "toy_stock"

    toy_id = Column(BigInteger, primary_key=True, autoincrement=True)  # 장난감 고유 ID
    user_id = Column(BigInteger, ForeignKey("user.user_id"), nullable=False)  # 판매매자 ID
    toy_name = Column(String(100), nullable=True)  # 장난감 이름
    toy_type = Column(String(50), nullable=True)  # 종류(태그역할)
    image_url = Column(JSON, nullable=True)  # 장난감 이미지 경로(2D)
    is_donatable = Column(Enum(DonationStatus), nullable=True)  # 기부 가능여부 판단 결과
    reject_reason = Column(Text, nullable=True)  # 기부 불가 사유
    glb_model_url = Column(String(500), nullable=True)  # 3D모델(glb) 이미지 경로
    toy_status = Column(Enum(ToyStatus), nullable=False)  # 상태 관리
    reserved_by_request_id = Column(BigInteger, ForeignKey("donor_requests.id"), nullable=True)  # 예약 요청 ID
    created_at = Column(DateTime, nullable=False)  # 생성시간
    updated_at = Column(DateTime, nullable=True)  # 수정 시간(변경 시 자동 갱신)
    description = Column(Text, nullable=True)  # LLM모델을 통한 장난감 설명

    sale_price = Column(BigInteger, nullable=True, default=0)  # 장난감 판매가
    purchase_price = Column(BigInteger, nullable=True, default=0)  # 장난감 매입가
    
    # 관계 설정
    donor = relationship("User", foreign_keys=[user_id], backref="donated_toys")
    reserved_by = relationship("Donor_Requests", foreign_keys=[reserved_by_request_id], backref="reserved_toys")

class Donor_Requests(Base):
    __tablename__ = "donor_requests"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # 요청 고유 ID
    recipient_id = Column(BigInteger, ForeignKey("user.user_id"), nullable=False)  # 요청한 사용자 ID
    recipient_status = Column(Enum(RecipientStatus), nullable=False)  # 상태 관리 - 관리자 승인관리
    requested_at = Column(DateTime, nullable=False)  # 요청 생성 시각
    delivery_status = Column(Enum(DeliveryStatus), nullable=False)  # 배송현황
    total_price = Column(BigInteger, nullable=True, default=0)  # 총 가격

    # 관계 설정
    recipient = relationship("User", foreign_keys=[recipient_id], backref="donation_requests")

    # ★ 명시적으로 정의
    request_items = relationship(
        "Donor_Requests_Item",
        back_populates="donation_request",
        cascade="all, delete-orphan",
    )

class Donor_Requests_Item(Base):
    __tablename__ = "donor_requests_item"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # item 고유 ID
    requested_id = Column(BigInteger, ForeignKey("donor_requests.id"), nullable=False)  # 요청 ID (donation_requests 테이블 참조)
    toy_stock_id = Column(BigInteger, ForeignKey("toy_stock.toy_id"), nullable=False)  # 장난감 ID (toy_stock 테이블 참조)
    unit_price = Column(BigInteger, nullable=True, default=0)  # 장난감 단가

    # 관계 설정
    donation_request = relationship("Donor_Requests", back_populates="request_items")
    toy_stock = relationship("Toy_Stock", backref="toy_request_items")

