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
  const [isDragging, setIsDragging] = useState(false);
  const dragState = useRef<{
    dragging: boolean;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  }>({
    dragging: false,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      securityLevel: "loose",
      theme: "dark",
      fontFamily: "IBM Plex Mono, monospace",
      themeVariables: {
        primaryColor: "#4F46E5",
        primaryTextColor: "#FFFFFF",
        primaryBorderColor: "#4338CA",
        secondaryColor: "#E0E7FF",
        secondaryTextColor: "#111827",
        secondaryBorderColor: "#C7D2FE",
        tertiaryColor: "#111827",
        lineColor: "#4338CA",
        textColor: "#E5E7EB",
        background: "#0B1120",
      },
    });

    if (containerRef.current && chart) {
      const id = `mermaid-diagram-${Math.random().toString(36).slice(2, 9)}`;

      mermaid
        .render(id, chart)
        .then(({ svg }: any) => {
          const responsiveSvg = svg.includes("data-responsive-mermaid")
            ? svg
            : svg.replace(
                "<svg",
                '<svg data-responsive-mermaid="true" preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto;display:block"',
              );

          if (containerRef.current) {
            containerRef.current.innerHTML = responsiveSvg;
          }
          setRenderedSvg(responsiveSvg);
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
    event.preventDefault();
    const delta = -event.deltaY;
    setZoom((prev) => {
      const next = Math.min(3, Math.max(0.4, prev + delta * 0.0015));
      return Number(next.toFixed(2));
    });
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
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
    setPan({
      x: dragState.current.originX + dx,
      y: dragState.current.originY + dy,
    });
  };

  const handleMouseUp = () => {
    dragState.current.dragging = false;
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <>
      <Card className="p-4 bg-background border-border overflow-hidden relative">
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
              Scroll inside the canvas or use the controls to zoom. Drag while
              holding the mouse button to pan. Double-click anywhere to reset
              the view.
            </DialogDescription>
          </DialogHeader>
          
          <div
            className={`h-[70vh] overflow-hidden border border-border rounded-md bg-background ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onDoubleClick={resetView}
            role="presentation"
          >
            {renderedSvg ? (
              <div
                className="flex items-center justify-center w-full h-full"
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: "center center",
                }}
                dangerouslySetInnerHTML={{ __html: renderedSvg }}
              />
            ) : (
              <div className="text-sm text-muted-foreground p-6">
                Rendering diagram...
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
