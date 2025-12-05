import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { Card } from "../ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { Maximize2 } from "lucide-react";

interface MermaidDiagramProps {
  chart: string;
}

export default function MermaidDiagram({ chart }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [renderedSvg, setRenderedSvg] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragState = useRef<{ dragging: boolean; startX: number; startY: number; originX: number; originY: number }>({
    dragging: false,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      theme: "dark",
      securityLevel: "loose",
      fontFamily: "monospace",
    });

    if (containerRef.current && chart) {
      const id = `mermaid-diagram-${Math.random().toString(36).slice(2, 9)}`;

      mermaid
        .render(id, chart)
        .then(({ svg }: any) => {
          if (containerRef.current) {
            containerRef.current.innerHTML = svg;
          }
          setRenderedSvg(svg);
        })
        .catch((_) => {
          if (containerRef.current) {
            containerRef.current.innerHTML = `
            <div class="text-red-400 text-sm p-4">
              Failed to render diagram
            </div>
          `;
          }
        });
    }
  }, [chart]);

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    // Respect user intent: only zoom when holding ctrl (touchpad pinch) to avoid accidental zooms.
    if (!event.ctrlKey) return;
    event.preventDefault();
    const delta = -event.deltaY;
    setZoom((prev) => {
      const next = Math.min(3, Math.max(0.4, prev + delta * 0.0015));
      return Number(next.toFixed(2));
    });
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    dragState.current = {
      dragging: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: pan.x,
      originY: pan.y,
    };
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragState.current.dragging) return;
    const dx = event.clientX - dragState.current.startX;
    const dy = event.clientY - dragState.current.startY;
    setPan({ x: dragState.current.originX + dx, y: dragState.current.originY + dy });
  };

  const handleMouseUp = () => {
    dragState.current.dragging = false;
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <>
      <Card className="p-4 bg-background border-border overflow-x-auto relative">
        <button
          type="button"
          onClick={() => setIsOpen(true)}
          className="absolute top-2 right-2 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition"
          aria-label="Expand diagram"
        >
          <Maximize2 className="h-4 w-4" />
          Expand
        </button>
        <div ref={containerRef} className="flex items-center justify-center" />
      </Card>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-6xl h-[85vh]">
          <DialogHeader>
            <DialogTitle>Diagram</DialogTitle>
            <DialogDescription>
              Use Ctrl + scroll/touchpad to zoom, drag to pan, or use the +/- controls. Click outside or press ESC to close.
            </DialogDescription>
          </DialogHeader>
          <div className="flex items-center gap-3 mb-3 text-xs text-muted-foreground">
            <label className="flex items-center gap-2">
              Zoom
              <span className="tabular-nums text-foreground">{Math.round(zoom * 100)}%</span>
            </label>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(0.4, Number((z - 0.1).toFixed(2))))}
                className="px-2 py-1 rounded border border-border text-foreground hover:bg-accent/50"
              >
                –
              </button>
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(3, Number((z + 0.1).toFixed(2))))}
                className="px-2 py-1 rounded border border-border text-foreground hover:bg-accent/50"
              >
                +
              </button>
            </div>
            <button
              type="button"
              onClick={resetView}
              className="px-2 py-1 rounded border border-border text-foreground hover:bg-accent/50"
            >
              Reset
            </button>
          </div>

          <div
            className="h-[70vh] overflow-hidden border border-border rounded-md bg-background"
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            role="presentation"
          >
            {renderedSvg ? (
              <div
                className="flex items-center justify-center w-full h-full"
                style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "center center" }}
                dangerouslySetInnerHTML={{ __html: renderedSvg }}
              />
            ) : (
              <div className="text-sm text-muted-foreground p-6">Rendering diagram...</div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
