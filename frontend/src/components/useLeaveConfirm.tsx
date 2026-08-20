import { useCallback, useEffect, useState } from "react";

type LeaveAction = { onConfirm: () => void };

/**
 * Keep the annotation page from silently discarding local edits.
 * This follows YoloWebAgent's leave-confirmation pattern and also covers
 * browser refresh/close through beforeunload.
 */
export function useLeaveConfirm(isDirty: boolean) {
  const [pending, setPending] = useState<LeaveAction | null>(null);

  const requestLeave = useCallback((onProceed: () => void) => {
    if (!isDirty) {
      onProceed();
      return;
    }
    setPending({ onConfirm: onProceed });
  }, [isDirty]);

  const confirmLeave = useCallback(() => {
    pending?.onConfirm();
    setPending(null);
  }, [pending]);

  const cancelLeave = useCallback(() => setPending(null), []);

  useEffect(() => {
    if (!isDirty) return undefined;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [isDirty]);

  const dialog = pending ? (
    <div className="modal-backdrop leave-confirm-backdrop" role="presentation" onMouseDown={cancelLeave}>
      <section
        className="dataset-dialog leave-confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="leave-confirm-title"
        aria-describedby="leave-confirm-description"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <span className="eyebrow">UNSAVED ANNOTATION</span>
            <h2 id="leave-confirm-title">未保存的标注</h2>
            <p id="leave-confirm-description">当前图片有未保存标注，确定放弃修改并继续吗？</p>
          </div>
        </header>
        <footer>
          <button className="button" type="button" onClick={cancelLeave}>继续编辑</button>
          <button className="button danger" type="button" onClick={confirmLeave}>放弃修改</button>
        </footer>
      </section>
    </div>
  ) : null;

  return { requestLeave, dialog };
}
