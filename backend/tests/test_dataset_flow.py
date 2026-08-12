from __future__ import annotations

import io
import zipfile
from dataclasses import replace

import pytest
from PIL import Image


def png_bytes(color: str = "#4477aa", size: tuple[int, int] = (100, 80)) -> bytes:
    image = Image.new("RGB", size, color)
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


def test_quality_report_and_deterministic_bulk_splits(client):
    dataset_id = create_dataset(client)
    class_a = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "a"}).json()["id"]
    client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "b"})
    images = []
    for index in range(5):
        response = client.post(f"/api/datasets/{dataset_id}/images/upload", files={"files": (f"{index}.png", png_bytes(), "image/png")})
        images.append(response.json()["items"][0])
    client.put(f"/api/datasets/{dataset_id}/images/{images[0]['id']}/annotations", json={"annotations": [
        {"type": "bbox", "class_id": class_a, "bbox": {"x": 10, "y": 10, "width": 10, "height": 10}},
        {"type": "bbox", "class_id": class_a, "bbox": {"x": 10.2, "y": 10.2, "width": 10, "height": 10}},
    ]})
    report = client.get(f"/api/datasets/{dataset_id}/quality/report")
    assert report.status_code == 200
    assert report.json()["summary"]["image_count"] == 5
    assert report.json()["summary"]["small_object_count"] == 2
    assert {item["type"] for item in report.json()["issues"]} >= {"small_object", "similar_bbox"}

    bulk = client.post(f"/api/datasets/{dataset_id}/images/bulk-split", json={"image_ids": [images[0]["id"], images[1]["id"]], "split": "val"})
    assert bulk.status_code == 200
    assert bulk.json()["updated"] == 2
    split = client.post(f"/api/datasets/{dataset_id}/images/auto-split", json={"train_ratio": 0.8, "val_ratio": 0.2, "test_ratio": 0, "seed": 42})
    assert split.status_code == 200
    assert split.json()["split_counts"] == {"train": 4, "val": 1, "test": 0}


def test_coco_round_trip_and_derived_tiling_dataset(client):
    dataset_id = create_dataset(client)
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        data={"split": "val"},
        files={"files": ("large.png", png_bytes(size=(300, 200)), "image/png")},
    ).json()["items"][0]
    client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={"annotations": [{"type": "bbox", "class_id": class_id, "bbox": {"x": 10, "y": 12, "width": 30, "height": 25}}]},
    )
    coco = client.get(f"/api/datasets/{dataset_id}/export/coco")
    assert coco.status_code == 200, coco.text
    with zipfile.ZipFile(io.BytesIO(coco.content)) as archive:
        manifest = __import__("json").loads(archive.read("annotations.json"))
        assert manifest["images"][0]["split"] == "val"
        assert manifest["annotations"][0]["bbox"] == [10.0, 12.0, 30.0, 25.0]
    imported = client.post(
        "/api/datasets/import/coco",
        data={"name": "coco-round-trip", "task_type": "detect"},
        files={"file": ("coco.zip", coco.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported_annotations"] == 1

    tiled = client.post(
        f"/api/datasets/{dataset_id}/tile",
        json={"name": "tiled", "tile_size": 128, "overlap": 0, "keep_empty_tiles": False},
    )
    assert tiled.status_code == 201, tiled.text
    assert tiled.json()["generated_images"] == 1
    detail = client.get(f"/api/datasets/{tiled.json()['dataset_id']}")
    assert detail.status_code == 200
    assert detail.json()["task_type"] == "detect"


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


def test_obb_annotations_validate_and_round_trip_through_yolo(client):
    dataset_id = create_dataset(client, "obb")
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "vehicle"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        data={"split": "val"},
        files={"files": ("vehicle.png", png_bytes(), "image/png")},
    ).json()["items"][0]

    saved = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={
            "annotations": [
                {"type": "obb", "class_id": class_id, "obb": {"cx": 45, "cy": 35, "width": 30, "height": 20, "angle": 20}}
            ]
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()[0]["obb"] == {"cx": 45.0, "cy": 35.0, "width": 30.0, "height": 20.0, "angle": 20.0}

    exported = client.get(f"/api/datasets/{dataset_id}/export/yolo")
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        label_name = next(name for name in archive.namelist() if name.startswith("labels/val/"))
        assert len(archive.read(label_name).decode().strip().split()) == 9

    imported = client.post(
        "/api/datasets/import/yolo",
        data={"name": "obb-round-trip", "task_type": "obb"},
        files={"file": ("obb.zip", exported.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported_annotations"] == 1


def test_obb_annotation_allows_right_angle_image_boundary_without_float_error(client):
    dataset_id = create_dataset(client, "obb")
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "vehicle"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("vehicle.png", png_bytes(), "image/png")},
    ).json()["items"][0]

    response = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={
            "annotations": [
                {"type": "obb", "class_id": class_id, "obb": {"cx": 30, "cy": 40, "width": 80, "height": 60, "angle": -90}}
            ]
        },
    )

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("0 0.10 0.10 0.50 0.10 0.40 0.50 0.10 0.40", "an ordered rectangle"),
        ("0 -0.01 0.10 0.50 0.10 0.50 0.50 0.10 0.50", "normalized to the [0, 1] range"),
    ],
)
def test_obb_yolo_import_rejects_non_rectangular_or_out_of_range_corners(client, label, message):
    archive_payload = io.BytesIO()
    with zipfile.ZipFile(archive_payload, "w") as archive:
        archive.writestr("data.yaml", "names: [vehicle]\n")
        archive.writestr("images/train/vehicle.png", png_bytes())
        archive.writestr("labels/train/vehicle.txt", f"{label}\n")

    response = client.post(
        "/api/datasets/import/yolo",
        data={"name": "invalid-obb", "task_type": "obb"},
        files={"file": ("invalid-obb.zip", archive_payload.getvalue(), "application/zip")},
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_obb_label"
    assert message in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("settings", "members", "expected_code"),
    [
        ({"max_yolo_archive_members": 2}, [("data.yaml", b"names: [object]\n"), ("images/train/object.png", png_bytes()), ("labels/train/object.txt", b"0 0.5 0.5 0.2 0.2\n")], "archive_too_many_members"),
        ({"max_yolo_archive_member_bytes": 32}, [("payload.bin", b"x" * 33)], "archive_member_too_large"),
        ({"max_yolo_archive_uncompressed_bytes": 64}, [("first.bin", b"x" * 32), ("second.bin", b"x" * 33)], "archive_total_too_large"),
    ],
)
def test_yolo_import_rejects_archive_count_and_size_limits(client, settings, members, expected_code):
    client.app.state.settings = replace(client.app.state.settings, **settings)
    archive_payload = io.BytesIO()
    with zipfile.ZipFile(archive_payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)

    response = client.post(
        "/api/datasets/import/yolo",
        data={"name": "limited", "task_type": "detect"},
        files={"file": ("limited.zip", archive_payload.getvalue(), "application/zip")},
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == expected_code


def test_yolo_import_rejects_excessive_compression_ratio_before_extracting_members(client):
    archive_payload = io.BytesIO()
    with zipfile.ZipFile(archive_payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("highly-compressible.bin", b"0" * (256 * 1024))

    response = client.post(
        "/api/datasets/import/yolo",
        data={"name": "zip-bomb", "task_type": "detect"},
        files={"file": ("zip-bomb.zip", archive_payload.getvalue(), "application/zip")},
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "archive_compression_ratio_exceeded"


def test_classification_uses_yolo_class_directories(client):
    dataset_id = create_dataset(client, "classify")
    cat_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "cat"}).json()["id"]
    dog_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "dog"}).json()["id"]
    images = []
    for split in ("train", "val"):
        response = client.post(
            f"/api/datasets/{dataset_id}/images/upload",
            data={"split": split},
            files={"files": (f"{split}.png", png_bytes(), "image/png")},
        )
        images.append(response.json()["items"][0])
    for image, class_id in zip(images, (cat_id, dog_id), strict=True):
        response = client.put(
            f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
            json={"annotations": [{"type": "classify", "class_id": class_id}]},
        )
        assert response.status_code == 200, response.text

    exported = client.get(f"/api/datasets/{dataset_id}/export/yolo")
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = set(archive.namelist())
        assert "data.yaml" not in names
        assert any(name.startswith("train/cat/") for name in names)
        assert any(name.startswith("val/dog/") for name in names)

    imported = client.post(
        "/api/datasets/import/yolo",
        data={"name": "classify-round-trip", "task_type": "classify"},
        files={"file": ("classify.zip", exported.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported_images"] == 2
    assert imported.json()["imported_annotations"] == 2


def test_sam_box_assist_returns_a_reviewable_polygon(client):
    dataset_id = create_dataset(client, "segment")
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "object"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("object.png", png_bytes(), "image/png")},
    ).json()["items"][0]
    suggestion = client.post(
        "/api/sam/predict",
        json={"image_id": image["id"], "class_id": class_id, "prompt_type": "box", "box": {"x": 10, "y": 12, "width": 30, "height": 25}},
    )
    assert suggestion.status_code == 200, suggestion.text
    payload = suggestion.json()
    assert payload["backend_used"] == "box_stub"
    assert payload["polygon"] == [[10.0, 12.0], [40.0, 12.0], [40.0, 37.0], [10.0, 37.0]]

    saved = client.put(
        f"/api/datasets/{dataset_id}/images/{image['id']}/annotations",
        json={"annotations": [{"type": "polygon", "class_id": class_id, "polygon": payload["polygon"], "source": "sam"}]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()[0]["source"] == "sam"


def test_system_info_reports_sam_capabilities_without_exposing_model_reference(client):
    unavailable = client.get("/api/system/info")
    assert unavailable.status_code == 200, unavailable.text
    assert unavailable.json()["sam"] == {
        "model_configured": False,
        "box_prompt_available": True,
        "point_prompt_available": False,
        "box_backend": "box_stub",
    }

    client.app.state.settings = replace(client.app.state.settings, sam_model="/private/checkpoints/sam_b.pt")
    configured = client.get("/api/system/info")
    assert configured.status_code == 200, configured.text
    assert configured.json()["sam"] == {
        "model_configured": True,
        "box_prompt_available": True,
        "point_prompt_available": True,
        "box_backend": "ultralytics_sam",
    }
    assert "/private/checkpoints/sam_b.pt" not in configured.text


def test_sam_uses_configured_ultralytics_model(client, monkeypatch):
    dataset_id = create_dataset(client, "segment")
    class_id = client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "object"}).json()["id"]
    image = client.post(
        f"/api/datasets/{dataset_id}/images/upload",
        files={"files": ("object.png", png_bytes(), "image/png")},
    ).json()["items"][0]

    class FakePolygon:
        def tolist(self):
            return [[12, 12], [40, 12], [40, 35], [12, 35]]

    class FakeMasks:
        xy = [FakePolygon()]

        class data:
            device = "mps"

    class FakeResults:
        masks = FakeMasks()

    class FakeSam:
        def predict(self, **kwargs):
            assert kwargs["bboxes"] == [[10, 12, 40, 37]]
            return [FakeResults()]

    client.app.state.settings = replace(client.app.state.settings, sam_model="fake-sam.pt")
    monkeypatch.setattr("app.sam.service._model_for", lambda _: FakeSam())
    suggestion = client.post(
        "/api/sam/predict",
        json={"image_id": image["id"], "class_id": class_id, "prompt_type": "box", "box": {"x": 10, "y": 12, "width": 30, "height": 25}},
    )
    assert suggestion.status_code == 200, suggestion.text
    assert suggestion.json()["backend_used"] == "ultralytics_sam"
    assert suggestion.json()["polygon"] == [[12.0, 12.0], [40.0, 12.0], [40.0, 35.0], [12.0, 35.0]]
    assert suggestion.json()["device"] == "mps"


def test_scan_path_is_restricted_to_import_root(client, tmp_path):
    dataset_id = create_dataset(client)
    response = client.post(f"/api/datasets/{dataset_id}/images/scan", json={"path": "../", "recursive": True, "split": "train"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "import_path_outside_root"


def test_scan_does_not_follow_file_symlink_outside_import_root(client, tmp_path):
    dataset_id = create_dataset(client)
    external_image = tmp_path / "outside.png"
    external_image.write_bytes(png_bytes())
    import_root = tmp_path / "imports"
    import_root.mkdir(exist_ok=True)
    link = import_root / "linked-outside.png"
    try:
        link.symlink_to(external_image)
    except OSError as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")

    response = client.post(
        f"/api/datasets/{dataset_id}/images/scan",
        json={"path": ".", "recursive": True, "split": "train"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"total_found": 1, "imported": 0, "skipped": 0, "invalid": 1}
    assert client.get(f"/api/datasets/{dataset_id}/images").json()["total"] == 0
