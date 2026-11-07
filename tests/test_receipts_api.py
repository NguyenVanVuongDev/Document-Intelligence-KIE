import io
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.database.connection import get_db
from src.database.models import AuditLog, Base, Receipt


class StubPipeline:
    def predict(self, _image):
        return []


class StubOCREngine:
    def __init__(self):
        self.pipeline = StubPipeline()

    def run(self, _image, max_side=960):
        return []


class ReceiptAPITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.db"
        self.database_engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.database_engine)
        self.session_factory = sessionmaker(bind=self.database_engine, expire_on_commit=False)

        def override_get_db():
            with self.session_factory() as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db
        self.database_patcher = patch("src.api.main.create_db_and_tables")
        self.ocr_patcher = patch("src.api.main.OCREngine", return_value=StubOCREngine())
        self.database_patcher.start()
        self.ocr_patcher.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.dependency_overrides.clear()
        self.ocr_patcher.stop()
        self.database_patcher.stop()
        self.database_engine.dispose()
        self.temp_dir.cleanup()

    @staticmethod
    def _receipt_png(compress_level: int = 6) -> bytes:
        image = Image.new("RGB", (320, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), "TEST STORE", fill="black")
        draw.text((20, 80), "TOTAL 10000", fill="black")
        output = io.BytesIO()
        image.save(output, format="PNG", compress_level=compress_level)
        return output.getvalue()

    def test_receipt_process_duplicate_filters_and_audit(self):
        payload = self._receipt_png()
        upload = {"image": ("receipt.png", payload, "image/png")}

        created = self.client.post("/api/v1/receipts/process", files=upload)
        self.assertEqual(created.status_code, 201, created.text)
        created_data = created.json()
        receipt_id = created_data["receipt_id"]
        self.assertEqual(len(created_data["sha256"]), 64)
        self.assertEqual(len(created_data["perceptual_hash"]), 16)

        duplicate = self.client.post("/api/v1/receipts/process", files=upload)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["detail"]["code"], "DUPLICATE_RECEIPT_DETECTED")
        self.assertEqual(duplicate.json()["detail"]["receipt_id"], receipt_id)

        perceptual_payload = self._receipt_png(compress_level=0)
        self.assertNotEqual(hashlib.sha256(payload).digest(), hashlib.sha256(perceptual_payload).digest())
        perceptual_duplicate = self.client.post(
            "/api/v1/receipts/process",
            files={"image": ("receipt-reencoded.png", perceptual_payload, "image/png")},
        )
        self.assertEqual(perceptual_duplicate.status_code, 409)
        self.assertEqual(perceptual_duplicate.json()["detail"]["receipt_id"], receipt_id)

        updated = self.client.put(
            f"/api/v1/receipts/{receipt_id}",
            json={"merchant": "Test Store", "date": "2026-11-06", "total": "10000", "status": "reconciled"},
            headers={"X-Actor": "accounting"},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["status"], "reconciled")

        listed = self.client.get(
            "/api/v1/receipts",
            params={
                "page": 1,
                "page_size": 10,
                "date_from": "2026-11-06",
                "date_to": "2026-11-06",
                "merchant": "Test",
                "status": "reconciled",
            },
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json()["total"], 1)
        self.assertEqual(listed.json()["items"][0]["id"], receipt_id)

        with self.session_factory() as session:
            audit_count = session.scalar(select(func.count()).select_from(AuditLog))
            receipt_count = session.scalar(select(func.count()).select_from(Receipt))
            update_log = session.scalar(select(AuditLog).where(AuditLog.action == "updated"))
        self.assertEqual(receipt_count, 1)
        self.assertEqual(audit_count, 2)
        self.assertEqual(update_log.actor, "accounting")
        self.assertEqual(update_log.changes["status"], {"old": "processed", "new": "reconciled"})


if __name__ == "__main__":
    unittest.main()