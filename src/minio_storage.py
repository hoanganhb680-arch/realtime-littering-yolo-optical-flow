import io
import os

from minio import Minio

from Config import Config, settings


class MinIOStorage:
    def __init__(self):
        self._bucket = settings.BUCKET_NAME
        self._fallback_dir = Config.OUTPUT_DIR
        self._available = False
        self._client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

        try:
            self._ensure_bucket()
            self._available = True
        except Exception as exc:
            print(f"[MinIO] Offline, using local evidence storage: {exc}")
            os.makedirs(self._fallback_dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def client(self):
        return self._client

    @property
    def bucket_name(self) -> str:
        return self._bucket

    def upload(self, filename: str, data: bytes, content_type: str) -> str:
        if not self._available:
            return self._save_local(filename, data)

        self._client.put_object(
            bucket_name=self._bucket,
            object_name=filename,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return self._build_minio_url(filename)

    def _save_local(self, filename: str, data: bytes) -> str:
        safe_name = os.path.basename(filename)
        os.makedirs(self._fallback_dir, exist_ok=True)
        path = os.path.join(self._fallback_dir, safe_name)
        with open(path, "wb") as f:
            f.write(data)
        return f"http://{settings.HOST}:{settings.PORT}/api/v1/evidence/{safe_name}"

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            print(f"[MinIO] Created bucket: '{self._bucket}'")

        import json

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self._bucket}/*"],
                }
            ],
        }
        try:
            self._client.set_bucket_policy(self._bucket, json.dumps(policy))
            print(f"[MinIO] Set public read policy for bucket '{self._bucket}'")
        except Exception as exc:
            print(f"[MinIO] Could not set bucket policy: {exc}")

    def _build_minio_url(self, filename: str) -> str:
        return f"http://{settings.MINIO_URL}/{self._bucket}/{filename}"
