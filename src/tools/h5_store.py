"""
H5 编辑器持久化：全局默认样式（键值）与共享素材库。

依赖项目自带的 PostgreSQL；任何读失败都降级返回空值，
调用方继续走内置默认，不让 DB 故障阻断视频处理。
"""

import logging
import uuid
from datetime import datetime, timezone

from storage.database.db import get_session
from storage.database.shared.model import H5KeyValue, H5Asset

logger = logging.getLogger(__name__)

GLOBAL_LAYERS_KEY = "global_layer_doc"
NAMED_STYLES_KEY = "named_styles"


def get_global_layer_doc() -> dict | None:
    try:
        session = get_session()
        try:
            row = session.get(H5KeyValue, GLOBAL_LAYERS_KEY)
            return dict(row.value) if row and isinstance(row.value, dict) else None
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"[h5_store] 读取全局默认样式失败，使用内置默认: {e}")
        return None


def set_global_layer_doc(doc: dict) -> None:
    session = get_session()
    try:
        row = session.get(H5KeyValue, GLOBAL_LAYERS_KEY)
        if row is None:
            row = H5KeyValue(key=GLOBAL_LAYERS_KEY, value=doc)
            session.add(row)
        else:
            row.value = doc
            row.updated_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()


def list_named_styles() -> list:
    """已命名样式列表（最新在前）。读失败降级返回空列表。"""
    try:
        session = get_session()
        try:
            row = session.get(H5KeyValue, NAMED_STYLES_KEY)
            styles = row.value.get("styles") if row and isinstance(row.value, dict) else None
            return list(reversed(styles)) if isinstance(styles, list) else []
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"[h5_store] 读取命名样式失败: {e}")
        return []


def save_named_style(name: str, layer_doc: dict, guide_text: str = "") -> dict:
    """新增一条命名样式；同名则覆盖。返回该样式条目。"""
    name = (name or "").strip() or "未命名样式"
    session = get_session()
    try:
        row = session.get(H5KeyValue, NAMED_STYLES_KEY)
        styles = list(row.value.get("styles", [])) if row and isinstance(row.value, dict) else []
        entry = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "layer_doc": layer_doc,
            "guide_text": guide_text or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        styles = [s for s in styles if isinstance(s, dict) and s.get("name") != name]
        styles.append(entry)
        value = {"styles": styles}
        if row is None:
            session.add(H5KeyValue(key=NAMED_STYLES_KEY, value=value))
        else:
            row.value = value
            row.updated_at = datetime.now(timezone.utc)
        session.commit()
        return entry
    finally:
        session.close()


def delete_named_style(style_id: str) -> bool:
    """按 id 删除一条命名样式。"""
    session = get_session()
    try:
        row = session.get(H5KeyValue, NAMED_STYLES_KEY)
        if row is None or not isinstance(row.value, dict):
            return False
        styles = [s for s in row.value.get("styles", []) if isinstance(s, dict) and s.get("id") != style_id]
        row.value = {"styles": styles}
        row.updated_at = datetime.now(timezone.utc)
        session.commit()
        return True
    finally:
        session.close()


def add_asset(name: str, url: str, content_type: str = "") -> dict:
    session = get_session()
    try:
        asset = H5Asset(name=name, url=url, content_type=content_type or None)
        session.add(asset)
        session.commit()
        session.refresh(asset)
        return {"id": asset.id, "name": asset.name, "url": asset.url}
    finally:
        session.close()


def list_assets(limit: int = 100) -> list:
    try:
        session = get_session()
        try:
            rows = (
                session.query(H5Asset)
                .order_by(H5Asset.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {"id": r.id, "name": r.name, "url": r.url, "content_type": r.content_type or ""}
                for r in rows
            ]
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"[h5_store] 读取素材库失败: {e}")
        return []
