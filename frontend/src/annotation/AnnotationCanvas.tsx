import { useEffect, useMemo, useState } from "react";
import { Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";
import type { Annotation, BBox, ClassLabel, ImageItem } from "../types";
import { clamp, normalizeBBox, toOriginalPoint } from "../utils/geometry";

export type AnnotationDraft = {
  id?: string;
  class_id: string;
  type: "bbox" | "polygon";
  bbox?: BBox;
  polygon?: [number, number][];
  source: "manual" | "imported";
};

interface Props {
  image: ImageItem;
  classes: ClassLabel[];
  annotations: Annotation[];
  activeClassId: string;
  onChange: (annotations: AnnotationDraft[]) => void;
}

const MAX_WIDTH = 900;
const MAX_HEIGHT = 620;

export function AnnotationCanvas({ image, classes, annotations, activeClassId, onChange }: Props) {
  const [imageElement, setImageElement] = useState<HTMLImageElement>();
  const [tool, setTool] = useState<"bbox" | "polygon">(image.dataset_id ? "bbox" : "bbox");
  const [dragStart, setDragStart] = useState<[number, number]>();
  const [dragEnd, setDragEnd] = useState<[number, number]>();
  const [polygonPoints, setPolygonPoints] = useState<[number, number][]>([]);
  const [displaySize, setDisplaySize] = useState({ width: MAX_WIDTH, height: MAX_HEIGHT });

  useEffect(() => {
    const element = new window.Image();
    element.onload = () => {
      setImageElement(element);
      const ratio = Math.min(MAX_WIDTH / image.width, MAX_HEIGHT / image.height, 1);
      setDisplaySize({ width: Math.max(1, Math.round(image.width * ratio)), height: Math.max(1, Math.round(image.height * ratio)) });
    };
    element.src = image.file_url.startsWith("http") ? image.file_url : `${(import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "")}${image.file_url}`;
  }, [image]);

  const scale = useMemo(() => ({ x: displaySize.width / image.width, y: displaySize.height / image.height }), [displaySize, image]);
  const toOriginal = (point: [number, number]) => toOriginalPoint(point, displaySize.width, displaySize.height, image.width, image.height);
  const pointer = (event: { target: { getStage: () => { getPointerPosition: () => { x: number; y: number } | null } | null } }): [number, number] | undefined => {
    const stage = event.target.getStage();
    const position = stage?.getPointerPosition();
    if (!position) return undefined;
    return [clamp(position.x, 0, displaySize.width), clamp(position.y, 0, displaySize.height)];
  };

  const finishPolygon = () => {
    if (polygonPoints.length < 3) return;
    onChange([...annotations.map(toDraft), { class_id: activeClassId, type: "polygon", polygon: polygonPoints.map(toOriginal), source: "manual" }]);
    setPolygonPoints([]);
  };

  return (
    <section className="canvas-card">
      <div className="canvas-toolbar">
        <div>
          <span className="eyebrow">ANNOTATION</span>
          <h2>{image.file_name}</h2>
        </div>
        <div className="tool-group">
          <button className={tool === "bbox" ? "button active" : "button"} onClick={() => { setTool("bbox"); setPolygonPoints([]); }}>BBox</button>
          <button className={tool === "polygon" ? "button active" : "button"} onClick={() => { setTool("polygon"); setDragStart(undefined); }}>Polygon</button>
          {tool === "polygon" && <button className="button primary" onClick={finishPolygon}>完成多边形</button>}
          <button className="button" onClick={() => { setPolygonPoints([]); setDragStart(undefined); setDragEnd(undefined); }}>清空草稿</button>
        </div>
      </div>
      <p className="hint">当前类别：{classes.find((item) => item.id === activeClassId)?.name ?? "未选择"}。坐标以原图像素保存。</p>
      <div className="stage-wrap">
        {imageElement ? (
          <Stage
            width={displaySize.width}
            height={displaySize.height}
            onMouseDown={(event) => {
              const point = pointer(event);
              if (!point) return;
              if (tool === "bbox") setDragStart(point);
              else setPolygonPoints((current) => [...current, point]);
            }}
            onMouseMove={(event) => {
              if (tool !== "bbox" || !dragStart) return;
              const point = pointer(event);
              if (point) setDragEnd(point);
            }}
            onMouseUp={(event) => {
              if (tool !== "bbox" || !dragStart) return;
              const point = pointer(event);
              if (!point) return;
              const displayBox = normalizeBBox(dragStart, point);
              const box = { x: displayBox.x / scale.x, y: displayBox.y / scale.y, width: displayBox.width / scale.x, height: displayBox.height / scale.y };
              if (box.width >= 3 && box.height >= 3) onChange([...annotations.map(toDraft), { class_id: activeClassId, type: "bbox", bbox: box, source: "manual" }]);
              setDragStart(undefined);
              setDragEnd(undefined);
            }}
          >
            <Layer>
              <KonvaImage image={imageElement} width={displaySize.width} height={displaySize.height} listening={false} />
              {annotations.map((annotation) => <AnnotationShape key={annotation.id} annotation={annotation} scale={scale} />)}
              {dragStart && dragEnd && <Rect {...displayBox(normalizeBBox(dragStart, dragEnd))} stroke="#f97316" dash={[6, 4]} />}
              {polygonPoints.length > 0 && <Line points={polygonPoints.flat()} stroke="#f97316" strokeWidth={2} closed={false} dash={[6, 4]} />}
            </Layer>
          </Stage>
        ) : <div className="loading">正在载入图片…</div>}
      </div>
      <div className="canvas-meta">
        <span>{image.width} × {image.height}px</span>
        <span>{annotations.length} 个已保存标注</span>
        {polygonPoints.length > 0 && <span>{polygonPoints.length} 个多边形点</span>}
      </div>
    </section>
  );
}

function displayBox(box: BBox) {
  return { x: box.x, y: box.y, width: box.width, height: box.height };
}

function toDraft(annotation: Annotation): AnnotationDraft {
  return { id: annotation.id, class_id: annotation.class_id, type: annotation.type, bbox: annotation.bbox ?? undefined, polygon: annotation.polygon ?? undefined, source: annotation.source };
}

function AnnotationShape({ annotation, scale }: { annotation: Annotation; scale: { x: number; y: number } }) {
  if (annotation.type === "bbox" && annotation.bbox) {
    return <Rect x={annotation.bbox.x * scale.x} y={annotation.bbox.y * scale.y} width={annotation.bbox.width * scale.x} height={annotation.bbox.height * scale.y} stroke={annotation.color} strokeWidth={2} />;
  }
  if (annotation.polygon) {
    return <Line points={annotation.polygon.flatMap(([x, y]) => [x * scale.x, y * scale.y])} closed stroke={annotation.color} fill={`${annotation.color}33`} strokeWidth={2} />;
  }
  return null;
}
