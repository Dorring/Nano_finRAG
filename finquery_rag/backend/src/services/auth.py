from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.user import User

# 使用 secrets 模块生成的随机密钥
SECRET_KEY = os.getenv("SECRET_KEY", "9594873a5e0a8ccde1adc3af1ba93f5f2bcc5632b77b750452713d216791c013")
# JWT 令牌的加密算法
ALGORITHM = "HS256"
# 访问令牌的默认过期时间（分钟，默认24小时以支持长时间交互测试）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# 密码加密上下文，使用 bcrypt 方案
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# HTTP Bearer 令牌安全验证方案
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码与哈希密码是否匹配

    :param plain_password: 明文密码
    :type plain_password: str
    :param hashed_password: 哈希处理后的密码
    :type hashed_password: str
    :return: 如果密码匹配返回 True，否则返回 False
    :rtype: bool
    """
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    对明文密码进行哈希加密

    :param password: 需要加密的明文密码
    :type password: str
    :return: 哈希加密后的密码字符串
    :rtype: str
    """
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT 访问令牌

    :param data: 需要编码到 JWT 令牌中的负载数据
    :type data: dict
    :param expires_delta: 令牌的过期时间增量，如果为 None 则使用默认过期时间
    :type expires_delta: Optional[timedelta]
    :return: 生成的 JWT 令牌字符串
    :rtype: str
    """
    """Create a JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """
    从 JWT 令牌中提取并验证当前用户，返回完整的 User ORM 对象。

    :param credentials: HTTP Bearer 授权凭据，通过依赖注入获取
    :type credentials: HTTPAuthorizationCredentials
    :param db: 数据库会话实例，通过依赖注入获取
    :type db: Session
    :return: 当前验证通过的用户 ORM 对象
    :rtype: User
    :raises HTTPException: 如果凭据无法验证或用户不存在，抛出 401 未授权异常
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception

        try:
            user_id = int(user_id_str)
        except ValueError:
            raise credentials_exception

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise credentials_exception

        return user
    except JWTError:
        raise credentials_exception

# 可选：使某些接口不需要强制身份验证
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)), db: Session = Depends(get_db)) -> Optional[User]:
    """
    可选的用户身份验证 - 如果未提供令牌或令牌无效则返回 None

    :param credentials: 可选的 HTTP Bearer 授权凭据，通过依赖注入获取，未提供时为 None
    :type credentials: Optional[HTTPAuthorizationCredentials]
    :return: 如果令牌有效则返回 User 对象，否则返回 None
    :rtype: Optional[User]
    """
    if credentials is None:
        return None
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            return None
        try:
            user_id = int(user_id_str)
        except ValueError:
            return None
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except JWTError:
        return None
