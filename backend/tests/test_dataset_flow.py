from __future__ import annotations

import io
import zipfile

from PIL import Image


def png_bytes(color: str = "#4477aa") -> bytes:
    image = Image.new("RGB", (100, 80), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def create_dataset(client, task_type: str = "detect") -> str:
    response = client.post("/api/datasets", json={"name": "demo", "task_type": task_type})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_detect_dataset_annotation_validation_and_yolo_round_trip(client):
    dataset_id = create_dataset(client)
    class_response = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat", "color": "#ff0000"})
    assert class_response.status_code == 201, class_response.text
    class_id = class_response.json()["id"]

    upload = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        data={"split": "val"},
        files={"files": ("cat.png", png_bytes(), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    image = upload.json()["items"][0]
    assert image["split"] == "val"

    annotations = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={"annotations": [{"type": "bbox", "class_id": class_id, "bbox": {"x": 10, "y": 12, "width": 30, "height": 25}}]},
    )
    assert annotations.status_code == 200, annotations.text
    assert annotations.json()[0]["bbox"] == {"x": 10.0, "y": 12.0, "width": 30.0, "height": 25.0}

    validation = client.post(f"/api/datasets/{dataset_id}/validate")
    assert validation.status_code == 200, validation.text
    assert validation.json()["valid"] is True

    image_response = client.get(image["file_url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    exported = client.get(f"/api/datasets/{dataset_id}/export/yolo")
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = set(archive.namelist())
        assert "data.yaml" in names
        assert any(name.startswith("images/val/") for name in names)
        label_name = next(name for name in names if name.startswith("labels/val/"))
        assert label_name.endswith(".txt")
        assert archive.read(label_name).decode().startswith("0 0.25")

    imported = client.post(
        "/api/datasets/import/yolo",
        data={"name": "round-trip", "task_type": "detect"},
        files={"file": ("export.zip", exported.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported_images"] == 1
    assert imported.json()["imported_annotations"] == 1


def test_segment_annotations_require_polygon_and_reject_out_of_bounds(client):
    dataset_id = create_dataset(client, "segment")
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "shape"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("shape.png", png_bytes(), "image/png")},
    ).json()["items"][0]

    rejected = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={"annotations": [{"type": "polygon", "class_id": class_id, "polygon": [[-1, 1], [20, 1], [20, 20]]}]},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "polygon_out_of_bounds"

    saved = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={"annotations": [{"type": "polygon", "class_id": class_id, "polygon": [[10, 10], [60, 10], [60, 50], [10, 50]]}]},
    )
    assert saved.status_code == 200, saved.text


def test_scan_path_is_restricted_to_import_root(client, tmp_path):
    dataset_id = create_dataset(client)
    response = client.post(f"/api/datasets/{dataset_id}/images/scan", json={"path": "../", "recursive": True, "split": "train"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "import_path_outside_root"
