'use client';

import { useReactFlow } from 'reactflow';
import { LayoutGrid, Maximize2, Minus, Plus } from 'lucide-react';

/** Zoom/fit/auto-layout toolbar — must render inside a <ReactFlowProvider>. */
export function CanvasToolbar({ onAutoLayout }: { onAutoLayout: () => void }) {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  return (
    <div className="pipeline-toolbar">
      <button className="pipeline-toolbar-btn" title="Zoom in" onClick={() => zoomIn()}>
        <Plus size={15} />
      </button>
      <button className="pipeline-toolbar-btn" title="Zoom out" onClick={() => zoomOut()}>
        <Minus size={15} />
      </button>
      <button className="pipeline-toolbar-btn" title="Fit view" onClick={() => fitView({ padding: 0.2, duration: 200 })}>
        <Maximize2 size={15} />
      </button>
      <div className="pipeline-toolbar-divider" />
      <button className="pipeline-toolbar-btn" title="Auto-layout" onClick={onAutoLayout}>
        <LayoutGrid size={15} />
      </button>
    </div>
  );
}
