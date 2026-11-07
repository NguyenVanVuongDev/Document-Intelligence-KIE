from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from src.database.hashing import hamming_distance
from src.database.models import AuditLog, Receipt, ReceiptItem


def find_duplicate(
    session: Session,
    sha256: str,
    perceptual_hash: str,
    max_hamming_distance: int = 4,
) -> Receipt | None:
    exact_match = session.scalar(
        select(Receipt).where(Receipt.sha256 == sha256, Receipt.status == "processed")
    )
    if exact_match is not None:
        return exact_match

    candidates = session.scalars(
        select(Receipt).where(Receipt.status == "processed", Receipt.perceptual_hash.is_not(None))
    )
    closest = None
    closest_distance = max_hamming_distance + 1
    for candidate in candidates:
        distance = hamming_distance(perceptual_hash, candidate.perceptual_hash)
        if distance < closest_distance:
            closest = candidate
            closest_distance = distance
    return closest if closest_distance <= max_hamming_distance else None


def create_receipt(session: Session, *, items: list[dict[str, Any]] | None = None, **values: Any) -> Receipt:
    receipt = Receipt(**values)
    receipt.items = [ReceiptItem(**item) for item in (items or [])]
    session.add(receipt)
    try:
        session.flush()
        session.add(AuditLog(
            receipt_id=receipt.id,
            action="created",
            actor="system",
            changes={"status": {"old": None, "new": receipt.status}},
        ))
        session.commit()
    except IntegrityError:
        session.rollback()
        raise
    session.refresh(receipt)
    return receipt


def get_receipt(session: Session, receipt_id: int) -> Receipt | None:
    return session.scalar(
        select(Receipt)
        .options(selectinload(Receipt.items))
        .where(Receipt.id == receipt_id)
    )


def list_receipts(
    session: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
    merchant: str | None = None,
    status: str | None = None,
) -> tuple[list[Receipt], int]:
    query = select(Receipt)
    count_query = select(func.count()).select_from(Receipt)
    filters = []
    if date_from is not None:
        filters.append(Receipt.receipt_date >= date_from)
    if date_to is not None:
        filters.append(Receipt.receipt_date <= date_to)
    if merchant:
        filters.append(Receipt.merchant.ilike(f"%{merchant.strip()}%"))
    if status:
        filters.append(Receipt.status == status)
    if filters:
        query = query.where(*filters)
        count_query = count_query.where(*filters)

    total = int(session.scalar(count_query) or 0)
    receipts = list(session.scalars(
        query.options(selectinload(Receipt.items))
        .order_by(Receipt.created_at.desc(), Receipt.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ))
    return receipts, total


def update_receipt(
    session: Session,
    receipt: Receipt,
    updates: dict[str, Any],
    *,
    actor: str = "anonymous",
) -> Receipt:
    changes: dict[str, dict[str, Any]] = {}
    field_map = {"date": "receipt_date"}
    for field, new_value in updates.items():
        model_field = field_map.get(field, field)
        old_value = getattr(receipt, model_field)
        if old_value != new_value:
            changes[field] = {
                "old": old_value.isoformat() if isinstance(old_value, date) else old_value,
                "new": new_value.isoformat() if isinstance(new_value, date) else new_value,
            }
            setattr(receipt, model_field, new_value)

    if changes:
        session.add(AuditLog(receipt_id=receipt.id, action="updated", actor=actor, changes=changes))
        session.commit()
        session.refresh(receipt)
    return receipt