from sqlalchemy import BigInteger, DateTime, Identity, Index, Integer, JSON, PrimaryKeyConstraint, Text, text
from typing import Optional
import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass


class H5KeyValue(Base):
    """H5 编辑器的键值存储（全局默认样式等）"""
    __tablename__ = "h5_kv"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )


class H5Asset(Base):
    """H5 编辑器的共享素材库（箭头等用户上传图片）"""
    __tablename__ = "h5_assets"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )