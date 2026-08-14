/** Best-effort parsing of Ultralytics-style training logs for live UI. */

export type EpochLossPoint = { epoch: number; boxLoss: number };

export type ParsedTrainingLogs = {
  epochRows: EpochLossPoint[];
  speedItPerSec?: string;
};

/** Epoch row: leading epoch/TOTAL, GPU mem token, then box / cls / dfl losses. */
const EPOCH_TABLE_ROW =
  /^\s*(\d+)\/(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/;

export function parseTrainingLogs(logs: string, configuredEpochs: number): ParsedTrainingLogs {
  const lines = logs.split("\n");
  const epochRows: EpochLossPoint[] = [];
  let speedItPerSec: string | undefined;

  for (const line of lines) {
    const speedMatch = line.match(/([\d.]+)\s*it\/s/i);
    if (speedMatch) speedItPerSec = `${speedMatch[1]} it/s`;

    const m = line.match(EPOCH_TABLE_ROW);
    if (!m) continue;
    const cur = Number.parseInt(m[1], 10);
    const total = Number.parseInt(m[2], 10);
    const boxLoss = Number.parseFloat(m[3]);
    if (!Number.isFinite(cur) || !Number.isFinite(total) || !Number.isFinite(boxLoss)) continue;
    if (configuredEpochs > 0 && total !== configuredEpochs) continue;
    epochRows.push({ epoch: cur, boxLoss });
  }

  const byEpoch = new Map<number, number>();
  for (const row of epochRows) {
    byEpoch.set(row.epoch, row.boxLoss);
  }
  const deduped: EpochLossPoint[] = [...byEpoch.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([epoch, boxLoss]) => ({ epoch, boxLoss }));

  return { epochRows: deduped, speedItPerSec };
}

export function formatEtaSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}
