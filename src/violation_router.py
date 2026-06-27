# ==============================================================================
# VIOLATION ROUTER — Endpoint, validate input, điều phối xử lý
# ==============================================================================
import uuid
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse

from minio.error import S3Error
from minio_storage import MinIOStorage
import db as _db

router   = APIRouter(prefix="/api/v1", tags=["violations"])
_storage = MinIOStorage()

# Lưu trữ các hành vi có điểm đánh giá cao nhất của từng person_id trong phiên chạy hiện tại
# personId -> {"db_row_id": int, "score": float, "minio_filename": str}
session_highest_violations = {}

def reset_session_violations(clear_db: bool = True, restore_from_db: bool = False):
    """Xóa lịch sử phiên chạy khi khởi động lại stream."""
    global session_highest_violations
    session_highest_violations.clear()
    if clear_db:
        _db.clear_all_violations()
        print("🧹 Đã làm sạch lịch sử session vi phạm ở Backend.")
        return
    if restore_from_db:
        for row in _db.get_highest_violations_by_person():
            session_highest_violations[int(row["person_id"])] = {
                "db_row_id": int(row["id"]),
                "score": float(row["score"]),
                "minio_filename": "",
                "video_url": row.get("evidence_video_url"),
            }
        print(f"♻️  Đã khôi phục {len(session_highest_violations)} vi phạm từ SQLite cho phiên realtime.")
        return
    print("🧹 Đã làm sạch lịch sử session vi phạm ở Backend.")

# ------------------------------------------------------------------
# POST /api/v1/violations — nhận vi phạm từ detector, lưu MinIO + SQLite
# ------------------------------------------------------------------

@router.post("/violations")
async def record_violation(
        personId:      int        = Form(...),
        trashId:       int        = Form(...),
        score:         float      = Form(...),
        violationType: str        = Form(...),
        timestamp:     str        = Form(...),
        evidenceImage: UploadFile = File(...),
        evidenceVideo: UploadFile | None = File(default=None),
):
    try:
        file_data = await evidenceImage.read()
        
        # Kiểm tra xem personId này đã có vi phạm nào trong session này chưa
        existing = session_highest_violations.get(personId)
        
        if existing:
            # Nếu vi phạm mới có điểm số cao hơn, cập nhật lại hành vi rõ rệt nhất
            if score > existing["score"]:
                filename = _build_filename(personId, trashId, evidenceImage.filename)
                # Upload a fresh evidence image so its URL matches the latest trashId.
                image_url = _storage.upload(filename, file_data, evidenceImage.content_type)
                video_url = await _upload_optional_video(personId, trashId, evidenceVideo) or existing.get("video_url")
                
                # Cập nhật SQLite
                _db.update_violation(
                    row_id=existing["db_row_id"],
                    trash_id=trashId,
                    violation_type=violationType,
                    score=score,
                    timestamp=timestamp,
                    evidence_url=image_url,
                    evidence_video_url=video_url,
                )
                
                # Cập nhật bộ nhớ phiên chạy
                session_highest_violations[personId] = {
                    "db_row_id": existing["db_row_id"],
                    "score": score,
                    "minio_filename": filename,
                    "video_url": video_url,
                }
                
                print(f"🔄 [Backend] Cập nhật vi phạm cho Person_{personId} với hành vi rõ hơn: {violationType} ({score * 100:.1f}%)")
                
                return {
                    "status" : "success",
                    "message": "Updated violation to highest score successfully",
                    "data": {
                        "id"           : existing["db_row_id"],
                        "personId"     : personId,
                        "trashId"      : trashId,
                        "violationType": violationType,
                        "evidenceUrl"  : image_url,
                        "evidenceVideoUrl": video_url,
                    },
                }
            else:
                # Bỏ qua vì hành vi mới không rõ rệt bằng hành vi cũ
                return {
                    "status" : "ignored",
                    "message": f"Ignored violation with lower score {score} for person {personId}"
                }
        else:
            # Lần đầu tiên ghi nhận vi phạm cho personId này trong session hiện tại
            filename  = _build_filename(personId, trashId, evidenceImage.filename)
            image_url = _storage.upload(filename, file_data, evidenceImage.content_type)
            video_url = await _upload_optional_video(personId, trashId, evidenceVideo)

            # Lưu mới vào SQLite
            row_id = _db.insert_violation(
                person_id      = personId,
                trash_id       = trashId,
                violation_type = violationType,
                score          = score,
                timestamp      = timestamp,
                evidence_url   = image_url,
                evidence_video_url = video_url,
            )

            # Lưu thông tin theo dõi vào session
            session_highest_violations[personId] = {
                "db_row_id": row_id,
                "score": score,
                "minio_filename": filename,
                "video_url": video_url,
            }

            _log_violation(personId, trashId, violationType, score, timestamp, image_url, video_url)

            return {
                "status" : "success",
                "message": "Recorded violation successfully",
                "data": {
                    "id"           : row_id,
                    "personId"     : personId,
                    "trashId"      : trashId,
                    "violationType": violationType,
                    "evidenceUrl"  : image_url,
                    "evidenceVideoUrl": video_url,
                },
            }
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Lỗi MinIO Storage: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {e}")

# ------------------------------------------------------------------
# GET /api/v1/violations — lấy danh sách lịch sử vi phạm từ SQLite
# ------------------------------------------------------------------

@router.get("/violations")
async def list_violations(
        limit:  int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
):
    rows  = _db.get_all_violations(limit=limit, offset=offset)
    total = _db.count_violations()
    return {
        "status": "success",
        "total" : total,
        "limit" : limit,
        "offset": offset,
        "data"  : rows,
    }


@router.get("/evidence/{filename}")
async def get_local_evidence(filename: str):
    from Config import Config
    import os

    safe_name = os.path.basename(filename)
    path = os.path.join(Config.OUTPUT_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evidence file not found.")
    ext = os.path.splitext(safe_name)[1].lower()
    media_type = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".png": "image/png",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
    }.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=safe_name)

# ------------------------------------------------------------------
# POST /api/v1/violations/clear — xóa sạch SQLite + MinIO + ảnh cục bộ
# ------------------------------------------------------------------

@router.post("/violations/clear")
async def clear_violations():
    try:
        # 1. Xóa sạch SQLite database
        _db.clear_all_violations()
        
        # 2. Xóa các file JPEG lưu ảnh bằng chứng cục bộ trong thư mục
        from Config import Config
        import glob
        import os
        cfg = Config()
        files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.webm", "*.avi", "*.mov"):
            files.extend(glob.glob(os.path.join(cfg.OUTPUT_DIR, pattern)))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
                
        # 3. Reset lịch sử phiên gộp ở backend
        reset_session_violations(clear_db=False)
        
        # 4. Xóa sạch bucket MinIO để đồng bộ
        try:
            objects = _storage.client.list_objects(_storage.bucket_name, recursive=True)
            for obj in objects:
                _storage.client.remove_object(_storage.bucket_name, obj.object_name)
        except Exception as e:
            print(f"⚠️  Không thể xóa hoàn toàn MinIO bucket: {e}")
            
        return {
            "status" : "success",
            "message": "Đã dọn dẹp và xóa toàn bộ dữ liệu lịch sử vi phạm thành công.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server khi dọn dẹp lịch sử: {e}")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _upload_optional_video(person_id: int, trash_id: int, evidence_video: UploadFile | None) -> str | None:
    if evidence_video is None:
        return None
    video_data = await evidence_video.read()
    if not video_data:
        return None
    filename = _build_filename(person_id, trash_id, evidence_video.filename or "clip.mp4")
    return _storage.upload(filename, video_data, evidence_video.content_type or "video/mp4")

def _build_filename(person_id: int, trash_id: int, original_name: str) -> str:
    ext = original_name.rsplit(".", 1)[-1] if "." in original_name else "jpg"
    return f"violation_P{person_id}_T{trash_id}_{uuid.uuid4().hex[:6]}.{ext}"

def _log_violation(person_id, trash_id, v_type, score, timestamp, image_url, video_url=None):
    sep = "=" * 50
    print(f"\n{sep}")
    print("🚨 ĐÃ LƯU VI PHẠM MỚI!")
    print(f"👤 Người vi phạm : {person_id}")
    print(f"🗑️  Rác ID        : {trash_id}")
    print(f"📋 Loại vi phạm  : {v_type}")
    print(f"📊 Score         : {score:.4f}")
    print(f"🕐 Thời gian     : {timestamp}")
    print(f"🔗 URL Bằng chứng: {image_url}")
    if video_url:
        print(f"URL Video       : {video_url}")
    print(f"{sep}\n")
