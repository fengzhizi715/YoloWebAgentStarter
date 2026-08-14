interface Props {
  total: number;
  ptCount: number;
  onnxCount: number;
  bestMap50: string;
}

export function ModelLibraryStats({ total, ptCount, onnxCount, bestMap50 }: Props) {
  return (
    <section className="models-stats-strip" aria-label="模型库摘要">
      <div>
        <small>模型总数</small>
        <strong>{total}</strong>
      </div>
      <div>
        <small>PT 权重</small>
        <strong>{ptCount}</strong>
      </div>
      <div>
        <small>ONNX 产物</small>
        <strong>{onnxCount}</strong>
      </div>
      <div>
        <small>最佳 mAP50</small>
        <strong>{bestMap50}</strong>
      </div>
    </section>
  );
}
