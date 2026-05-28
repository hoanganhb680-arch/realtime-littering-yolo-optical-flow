# ==============================================================================
# MINIO STORAGE — Quản lý kết nối & upload file lên MinIO
# ==============================================================================
import io
from minio import Minio
from Config import settings
class MinIOStorage:
    def __init__(self):
        self._client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._bucket = settings.BUCKET_NAME
        self._ensure_bucket()
    def upload(self, filename: str, data: bytes, content_type: str) -> str:
        """Upload bytes lên MinIO, trả về public URL. Raises S3Error nếu thất bại."""
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=filename,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return self._build_url(filename)
    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            print(f"📦 Đã tạo mới bucket: '{self._bucket}'")
        
        # Thiết lập chính sách public read-only để trình duyệt xem được ảnh bằng chứng
        import json
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self._bucket}/*"]
                }
            ]
        }
        try:
            self._client.set_bucket_policy(self._bucket, json.dumps(policy))
            print(f"🔓 Đã cấu hình public read policy cho bucket '{self._bucket}'")
        except Exception as e:
            print(f"⚠️  Không thể thiết lập bucket policy: {e}")

    def _build_url(self, filename: str) -> str:
        return f"http://{settings.MINIO_URL}/{self._bucket}/{filename}"