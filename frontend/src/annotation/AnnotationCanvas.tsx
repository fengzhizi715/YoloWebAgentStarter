import { useEffect, useMemo, useRef, useState } from "react";
import Konva from "konva";
import { Image as KonvaImage, Layer, Line, Rect, Stage, Transformer } from "react-konva";
import type { Annotation, BBox, ClassLabel, ImageItem, OBB, SamCapabilities, TaskType } from "../types";
import { clamp, clampBBoxToImage, clampOBBToImage, MIN_SHAPE_SIZE, normalizeAngle, normalizeBBox, toOriginalPoint } from "../utils/geometry";

export type AnnotationDraft = {
  id?: string;
  class_id: string;
  type: "bbox" | "polygon" | "obb" | "classify";
  bbox?: BBox;
  polygon?: [number, number][];
  obb?: OBB;
  source: "manual" | "imported" | "sam" | "auto";
};

interface Props {
  image: ImageItem;
  classes: ClassLabel[];
  annotations: Annotation[];
  activeClassId: string;
  selectedAnnotationId?: string | null;
  onSelectAnnotation?: (id: string | null) => void;
  taskType: TaskType;
  onChange: (annotations: AnnotationDraft[]) => void;
  onSamBox?: (box: BBox) => Promise<void>;
  onSamPoints?: (points: [number, number][]) => Promise<void>;
  samBusy?: boolean;
  samCapabilities?: SamCapabilities;
}

const MAX_WIDTH = 900;
const MAX_HEIGHT = 620;

type DrawingTool = "bbox" | "polygon" | "obb" | "sam" | "sam_point";

const taskTool = (taskType: TaskType): DrawingTool => ({ detect: "bbox", segment: "polygon", obb: "obb", classify: "bbox" } satisfies Record<TaskType, DrawingTool>)[taskType];

export function AnnotationCanvas({ image, classes, annotations, activeClassId, selectedAnnotationId, onSelectAnnotation, taskType, onChange, onSamBox, onSamPoints, samBusy = false, samCapabilities }: Props) {
  const [imageElement, setImageElement] = useState<HTMLImageElement>();
  const [tool, setTool] = useState<DrawingTool>(() => taskTool(taskType));
  const [dragStart, setDragStart] = useState<[number, number]>();
  const [dragEnd, setDragEnd] = useState<[number, number]>();
  const [polygonPoints, setPolygonPoints] = useState<[number, number][]>([]);
  const [angle, setAngle] = useState(0);
  const [selectedObbId, setSelectedObbId] = useState<string>();
  const [selectedBboxId, setSelectedBboxId] = useState<string>();
  const [displaySize, setDisplaySize] = useState({ width: MAX_WIDTH, height: MAX_HEIGHT });
  const transformerRef = useRef<Konva.Transformer>(null);
  const obbRefs = useRef<Record<string, Konva.Rect | null>>({});
  const bboxRefs = useRef<Record<string, Konva.Rect | null>>({});

  useEffect(() => {
    const element = new window.Image();
    element.onload = () => {
      setImageElement(element);
      const ratio = Math.min(MAX_WIDTH / image.width, MAX_HEIGHT / image.height, 1);
      setDisplaySize({ width: Math.max(1, Math.round(image.width * ratio)), height: Math.max(1, Math.round(image.height * ratio)) });
    };
    element.src = image.file_url.startsWith("http") ? image.file_url : `${(import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "")}${image.file_url}`;
  }, [image]);

  useEffect(() => {
    setTool(taskTool(taskType));
    setPolygonPoints([]);
    setDragStart(undefined);
    setDragEnd(undefined);
    setSelectedObbId(undefined);
    setSelectedBboxId(undefined);
  }, [taskType, image.id]);

  useEffect(() => {
    if (tool === "sam_point" && !samCapabilities?.point_prompt_available) setTool("polygon");
  }, [samCapabilities?.point_prompt_available, tool]);

  const scale = useMemo(() => ({ x: displaySize.width / image.width, y: displaySize.height / image.height }), [displaySize, image]);
  const effectiveSelectedAnnotationId = selectedAnnotationId !== undefined ? selectedAnnotationId : selectedObbId ?? selectedBboxId;
  const selectedObb = annotations.find((item) => item.id === effectiveSelectedAnnotationId && item.type === "obb");
  const selectedBbox = annotations.find((item) => item.id === effectiveSelectedAnnotationId && item.type === "bbox");

  useEffect(() => {
    const node = selectedObb ? obbRefs.current[selectedObb.id] : selectedBbox ? bboxRefs.current[selectedBbox.id] : null;
    transformerRef.current?.nodes(node ? [node] : []);
    transformerRef.current?.getLayer()?.batchDraw();
  }, [annotations, selectedBbox, selectedObb]);

  useEffect(() => {
    if (!dragStart) return;
    const cancelDrawing = () => {
      setDragStart(undefined);
      setDragEnd(undefined);
    };
    window.addEventListener("mouseup", cancelDrawing);
    return () => {
      window.removeEventListener("mouseup", cancelDrawing);
    };
  }, [dragStart]);

  const toOriginal = (point: [number, number]) => toOriginalPoint(point, displaySize.width, displaySize.height, image.width, image.height);
  const pointer = (event: { target: { getStage: () => { getPointerPosition: () => { x: number; y: number } | null } | null } }): [number, number] | undefined => {
    const stage = event.target.getStage();
    const position = stage?.getPointerPosition();
    if (!position) return undefined;
    return [clamp(position.x, 0, displaySize.width), clamp(position.y, 0, displaySize.height)];
  };
  const toStageObb = (obb: OBB): OBB => ({ ...obb, cx: obb.cx * scale.x, cy: obb.cy * scale.y, width: obb.width * scale.x, height: obb.height * scale.y });
  const toImageObb = (obb: OBB): OBB => clampOBBToImage(
    { ...obb, cx: obb.cx / scale.x, cy: obb.cy / scale.y, width: obb.width / scale.x, height: obb.height / scale.y, angle: normalizeAngle(obb.angle) },
    image.width,
    image.height,
  );

  const replaceBbox = (id: string, stageBox: BBox) => {
    const updated = clampBBoxToImage(
      { x: stageBox.x / scale.x, y: stageBox.y / scale.y, width: stageBox.width / scale.x, height: stageBox.height / scale.y },
      image.width,
      image.height,
    );
    onChange(annotations.map((item) => item.id === id && item.type === "bbox" ? { ...toDraft(item), bbox: updated } : toDraft(item)));
  };

  const replaceObb = (id: string, stageObb: OBB) => {
    const updated = toImageObb(stageObb);
    onChange(annotations.map((item) => item.id === id && item.type === "obb" ? { ...toDraft(item), obb: updated } : toDraft(item)));
  };
  const removeSelectedObb = () => {
    if (!selectedObb?.id) return;
    onChange(annotations.filter((item) => item.id !== selectedObb.id).map(toDraft));
    setSelectedObbId(undefined);
    onSelectAnnotation?.(null);
  };
  const finishPolygon = () => {
    if (polygonPoints.length < 3) return;
    onChange([...annotations.map(toDraft), { id: newDraftId(), class_id: activeClassId, type: "polygon", polygon: polygonPoints.map(toOriginal), source: "manual" }]);
    setPolygonPoints([]);
  };
  const samConfigured = samCapabilities?.model_configured ?? false;
  const pointPromptAvailable = samCapabilities?.point_prompt_available ?? false;
  const canAnnotate = Boolean(activeClassId);
  const selectedAngle = selectedObb?.obb?.angle ?? angle;
  const draftObb = dragStart && dragEnd && tool === "obb" ? normalizeBBox(dragStart, dragEnd) : undefined;

  return (
    <section className="canvas-card">
      <div className="canvas-toolbar">
        <div>
          <span className="eyebrow">ANNOTATION</span>
          <h2>{image.file_name}</h2>
        </div>
        <div className="tool-group">
          {taskType === "detect" && <button className={tool === "bbox" ? "button active" : "button"} disabled={!canAnnotate} onClick={() => { setTool("bbox"); setPolygonPoints([]); }}>BBox</button>}
          {taskType === "segment" && <><button className={tool === "polygon" ? "button active" : "button"} disabled={!canAnnotate} onClick={() => { setTool("polygon"); setDragStart(undefined); }}>Polygon</button><button className={tool === "sam" ? "button active" : "button"} disabled={!canAnnotate || samBusy || !samCapabilities?.box_prompt_available} onClick={() => { setTool("sam"); setPolygonPoints([]); }}>{samConfigured ? "SAM 框选" : "框形建议"}</button><button className={tool === "sam_point" ? "button active" : "button"} disabled={!canAnnotate || samBusy || !pointPromptAvailable} title={pointPromptAvailable ? undefined : "配置 YWA_SAM_MODEL 后可用"} onClick={() => { setTool("sam_point"); setPolygonPoints([]); }}>{pointPromptAvailable ? "SAM 点选" : "SAM 点选（需配置）"}</button></>}
          {taskType === "obb" && <><button className={tool === "obb" ? "button active" : "button"} disabled={!canAnnotate} onClick={() => { setTool("obb"); setPolygonPoints([]); }}>OBB</button><label className="angle-input">角度<input aria-label="OBB angle" type="number" value={selectedAngle} onChange={(event) => { const nextAngle = Number(event.target.value); if (!Number.isFinite(nextAngle)) return; if (selectedObb?.obb) replaceObb(selectedObb.id, { ...toStageObb(selectedObb.obb), angle: nextAngle }); else setAngle(nextAngle); }} /></label>{selectedObb && <button className="button danger" onClick={removeSelectedObb}>删除所选 OBB</button>}</>}
          {tool === "polygon" && <button className="button primary" disabled={!canAnnotate} onClick={finishPolygon}>完成多边形</button>}
          <button className="button" onClick={() => { setPolygonPoints([]); setDragStart(undefined); setDragEnd(undefined); setSelectedObbId(undefined); }}>清空草稿</button>
        </div>
      </div>
      <p className="hint">当前类别：{classes.find((item) => item.id === activeClassId)?.name ?? "未选择"}。{!canAnnotate ? "请先在右侧新增一个类别，再开始绘制。" : taskType === "obb" ? selectedObb ? "已选中 OBB：拖动可移动，四角手柄可缩放，顶部圆形手柄可旋转。" : "拖拽创建 OBB；点击既有框即可选择、旋转、缩放或删除。" : taskType === "detect" && selectedBbox ? "已选中框：拖动可移动，四角手柄可调整大小；BBox 不支持旋转。" : tool === "sam" ? samConfigured ? "拖拽一个提示框，SAM 会返回可确认的多边形建议。" : "未配置 SAM：框选只会生成可审阅的矩形建议，不会运行模型推理。" : tool === "sam_point" ? "点击一个前景点，SAM 会返回可确认的多边形建议。" : taskType === "segment" && !samConfigured ? "SAM 未配置；可使用 Polygon，或使用仅供审阅的框形建议。" : "坐标以原图像素保存。"}</p>
      <div className="stage-wrap">
        {imageElement ? (
          <Stage
            width={displaySize.width}
            height={displaySize.height}
            onMouseDown={(event) => {
              if (!activeClassId) return;
              const point = pointer(event);
              if (!point) return;
              if (tool !== "polygon" && !["Stage", "Layer"].includes(event.target.getClassName())) return;
              if (tool === "sam_point" && onSamPoints) {
                void onSamPoints([toOriginal(point)]);
                return;
              }
              if (tool !== "polygon") setDragStart(point);
              else setPolygonPoints((current) => [...current, point]);
            }}
            onMouseMove={(event) => {
              if (tool === "polygon" || tool === "sam_point" || !dragStart) return;
              const point = pointer(event);
              if (point) setDragEnd(point);
            }}
            onMouseUp={(event) => {
              if (!activeClassId || tool === "polygon" || tool === "sam_point" || !dragStart) return;
              const point = pointer(event);
              if (!point) {
                setDragStart(undefined);
                setDragEnd(undefined);
                return;
              }
              const displayBox = normalizeBBox(dragStart, point);
              const box = { x: displayBox.x / scale.x, y: displayBox.y / scale.y, width: displayBox.width / scale.x, height: displayBox.height / scale.y };
              if (box.width >= MIN_SHAPE_SIZE && box.height >= MIN_SHAPE_SIZE) {
                if (tool === "bbox") onChange([...annotations.map(toDraft), { id: newDraftId(), class_id: activeClassId, type: "bbox", bbox: box, source: "manual" }]);
                if (tool === "obb") {
                  const id = `draft-obb-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                  onChange([...annotations.map(toDraft), { id, class_id: activeClassId, type: "obb", obb: clampOBBToImage({ cx: box.x + box.width / 2, cy: box.y + box.height / 2, width: box.width, height: box.height, angle }, image.width, image.height), source: "manual" }]);
                  setSelectedObbId(id);
                }
                if (tool === "sam" && onSamBox) void onSamBox(box);
              }
              setDragStart(undefined);
              setDragEnd(undefined);
            }}
            onMouseLeave={() => {
              if (tool !== "polygon" && tool !== "sam_point") {
                setDragStart(undefined);
                setDragEnd(undefined);
              }
            }}
          >
            <Layer>
              <KonvaImage image={imageElement} width={displaySize.width} height={displaySize.height} listening={false} />
              {annotations.map((annotation) => annotation.type === "obb" && annotation.obb ? <EditableObb key={annotation.id} annotation={annotation} stageObb={toStageObb(annotation.obb)} selected={annotation.id === effectiveSelectedAnnotationId} setNodeRef={(node) => { obbRefs.current[annotation.id] = node; }} onSelect={() => { setSelectedObbId(annotation.id); setSelectedBboxId(undefined); onSelectAnnotation?.(annotation.id); }} onChange={(updated) => replaceObb(annotation.id, updated)} /> : annotation.type === "bbox" && annotation.bbox ? <EditableBbox key={annotation.id} annotation={annotation} scale={scale} selected={annotation.id === effectiveSelectedAnnotationId} setNodeRef={(node) => { bboxRefs.current[annotation.id] = node; }} onSelect={() => { setSelectedBboxId(annotation.id); setSelectedObbId(undefined); onSelectAnnotation?.(annotation.id); }} onChange={(updated) => replaceBbox(annotation.id, updated)} /> : <AnnotationShape key={annotation.id} annotation={annotation} scale={scale} onSelect={() => onSelectAnnotation?.(annotation.id)} />)}
              {dragStart && dragEnd && tool !== "obb" && <Rect {...displayBox(normalizeBBox(dragStart, dragEnd))} stroke="#f97316" dash={[6, 4]} />}
              {draftObb && <Rect x={draftObb.x + draftObb.width / 2} y={draftObb.y + draftObb.height / 2} width={draftObb.width} height={draftObb.height} offsetX={draftObb.width / 2} offsetY={draftObb.height / 2} rotation={angle} stroke="#f97316" dash={[6, 4]} />}
              {polygonPoints.length > 0 && <Line points={polygonPoints.flat()} stroke="#f97316" strokeWidth={2} closed={false} dash={[6, 4]} />}
              <Transformer ref={transformerRef} rotateEnabled={Boolean(selectedObb)} flipEnabled={false} enabledAnchors={["top-left", "top-right", "bottom-left", "bottom-right"]} rotateAnchorOffset={28} boundBoxFunc={(previous, next) => next.width < MIN_SHAPE_SIZE * scale.x || next.height < MIN_SHAPE_SIZE * scale.y ? previous : next} />
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

function EditableObb({ annotation, stageObb, selected, setNodeRef, onSelect, onChange }: { annotation: Annotation; stageObb: OBB; selected: boolean; setNodeRef: (node: Konva.Rect | null) => void; onSelect: () => void; onChange: (obb: OBB) => void }) {
  return <Rect ref={setNodeRef} x={stageObb.cx} y={stageObb.cy} width={stageObb.width} height={stageObb.height} offsetX={stageObb.width / 2} offsetY={stageObb.height / 2} rotation={stageObb.angle} stroke={annotation.color} strokeWidth={selected ? 3 : 2} fill={selected ? `${annotation.color}22` : "transparent"} draggable onMouseDown={(event) => { event.cancelBubble = true; onSelect(); }} onClick={onSelect} onTap={onSelect} onDragEnd={(event) => onChange({ ...stageObb, cx: event.target.x(), cy: event.target.y() })} onTransformEnd={(event) => { const node = event.target; const scaleX = node.scaleX(); const scaleY = node.scaleY(); node.scaleX(1); node.scaleY(1); onChange({ cx: node.x(), cy: node.y(), width: Math.abs(node.width() * scaleX), height: Math.abs(node.height() * scaleY), angle: node.rotation() }); }} />;
}

function EditableBbox({ annotation, scale, selected, setNodeRef, onSelect, onChange }: { annotation: Annotation; scale: { x: number; y: number }; selected: boolean; setNodeRef: (node: Konva.Rect | null) => void; onSelect: () => void; onChange: (bbox: BBox) => void }) {
  const bbox = annotation.bbox;
  if (!bbox) return null;
  const stageBox = { x: bbox.x * scale.x, y: bbox.y * scale.y, width: bbox.width * scale.x, height: bbox.height * scale.y };
  return <Rect ref={setNodeRef} {...stageBox} stroke={annotation.color} strokeWidth={selected ? 3 : 2} fill={selected ? `${annotation.color}22` : "transparent"} draggable onMouseDown={(event) => { event.cancelBubble = true; onSelect(); }} onClick={onSelect} onTap={onSelect} onDragEnd={(event) => onChange({ ...stageBox, x: event.target.x(), y: event.target.y() })} onTransformEnd={(event) => { const node = event.target; const scaleX = node.scaleX(); const scaleY = node.scaleY(); node.scaleX(1); node.scaleY(1); onChange({ x: node.x(), y: node.y(), width: Math.abs(node.width() * scaleX), height: Math.abs(node.height() * scaleY) }); }} />;
}

function displayBox(box: BBox) {
  return { x: box.x, y: box.y, width: box.width, height: box.height };
}

function toDraft(annotation: Annotation): AnnotationDraft {
  return { id: annotation.id, class_id: annotation.class_id, type: annotation.type, bbox: annotation.bbox ?? undefined, polygon: annotation.polygon ?? undefined, obb: annotation.obb ?? undefined, source: annotation.source };
}

function AnnotationShape({ annotation, scale, onSelect }: { annotation: Annotation; scale: { x: number; y: number }; onSelect: () => void }) {
  if (annotation.type === "bbox" && annotation.bbox) {
    return <Rect x={annotation.bbox.x * scale.x} y={annotation.bbox.y * scale.y} width={annotation.bbox.width * scale.x} height={annotation.bbox.height * scale.y} stroke={annotation.color} strokeWidth={2} />;
  }
  if (annotation.polygon) {
    return <Line points={annotation.polygon.flatMap(([x, y]) => [x * scale.x, y * scale.y])} closed stroke={annotation.color} fill={`${annotation.color}33`} strokeWidth={2} onMouseDown={(event) => { event.cancelBubble = true; onSelect(); }} onClick={onSelect} onTap={onSelect} />;
  }
  return null;
}

function newDraftId(): string { return `draft-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
