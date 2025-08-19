USE myapi_db;
SHOW TABLES;
SELECT * from user;
SELECT * from toy_stock;

INSERT INTO user (user_id, name, email, password, address, role, phone_number, created_at)
VALUES
(1, '홍길동', 'hong@example.com', 'hashed_password', '서울시 강남구', 'user', '010-1234-5678', NOW()),
(2, '김철수', 'kim@example.com', 'hashed_password', '부산시 해운대구', 'user', '010-9876-5432', NOW());
INSERT INTO toy_stock
(donor_id, toy_name, toy_type, image_url, is_donatable, donor_status, created_at, updated_at, description)
VALUES
(1, '레고 블럭', '블럭', 'https://example.com/lego.jpg', 'recyclable', 'approved', NOW(), NOW(), '다양한 색깔의 레고 블럭 세트'),
(2, '곰인형', '인형', 'https://example.com/bear.jpg', 'upcycle', 'pending', NOW(), NOW(), '부드러운 털의 대형 곰인형');

UPDATE user SET role = 'USER' WHERE role = 'GROUP';

DESC user;

SHOW COLUMNS FROM user;
SHOW COLUMNS FROM toy_stock;

SELECT * FROM user;

DROP DATABASE myapi_db;

CREATE DATABASE myapi_db;

USE myapi_db;
SHOW TABLES;