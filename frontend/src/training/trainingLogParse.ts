/** Best-effort parsing of Ultralytics-style training logs for live UI. */

export type EpochLossPoint = { epoch: number; boxLoss: number; clsLoss?: number; dflLoss?: number };

export type ParsedTrainingLogs = {
  epochRows: EpochLossPoint[];
  speedItPerSec?: string;
  batchTimeSeconds?: number;
};

/** Epoch row: leading epoch/TOTAL, GPU mem token, then box / cls / dfl losses. */
const EPOCH_TABLE_ROW =
  /^\s*(\d+)\/(\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/;

export function parseTrainingLogs(logs: string, configuredEpochs: number): ParsedTrainingLogs {
  const lines = logs.split("\n");
  const epochRows: EpochLossPoint[] = [];
  let speedItPerSec: string | undefined;
  let batchTimeSeconds: number | undefined;

  for (const line of lines) {
    const speedMatch = line.match(/([\d.]+)\s*it\/s/i);
    if (speedMatch) {
      const speed = Number.parseFloat(speedMatch[1]);
      if (Number.isFinite(speed) && speed > 0) {
        speedItPerSec = `${speedMatch[1]} it/s`;
        batchTimeSeconds = 1 / speed;
      }
    }

    const m = line.match(EPOCH_TABLE_ROW);
    if (!m) continue;
    const cur = Number.parseInt(m[1], 10);
    const total = Number.parseInt(m[2], 10);
    const boxLoss = Number.parseFloat(m[3]);
    const clsLoss = Number.parseFloat(m[4]);
    const dflLoss = Number.parseFloat(m[5]);
    if (!Number.isFinite(cur) || !Number.isFinite(total) || !Number.isFinite(boxLoss)) continue;
    if (configuredEpochs > 0 && total !== configuredEpochs) continue;
    epochRows.push({
      epoch: cur,
      boxLoss,
      ...(Number.isFinite(clsLoss) ? { clsLoss } : {}),
      ...(Number.isFinite(dflLoss) ? { dflLoss } : {}),
    });
  }

  const byEpoch = new Map<number, number>();
  for (const row of epochRows) {
    byEpoch.set(row.epoch, row.boxLoss);
  }
  const deduped: EpochLossPoint[] = [...byEpoch.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([epoch, boxLoss]) => ({ epoch, boxLoss }));

  return { epochRows: deduped, speedItPerSec, batchTimeSeconds };
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
