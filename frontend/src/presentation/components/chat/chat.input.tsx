import type React from "react";

import { useState, useEffect, useRef } from "react";
import type { ChatInputProps } from "../../../lib/types";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { Card } from "../ui/card";
import { Send } from "lucide-react";

export default function ChatInput({ onSubmit, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const minTextareaHeight = 48;

  useEffect(() => {
    if (!textareaRef.current) return;
    const el = textareaRef.current;
    el.style.height = "0px";
    const next = Math.max(minTextareaHeight, el.scrollHeight);
    el.style.height = `${next}px`;
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || disabled) return;

    onSubmit(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="sticky bottom-0 z-30 w-full shadow-lg shadow-cyan-500/70 border-border/60 bg-card/90 backdrop-blur">
      <Card className="mx-auto max-w-4xl bg-transparent border-0 shadow-none px-4 py-3">
        <form onSubmit={handleSubmit} className="space-y-2">
          <div className="flex items-center gap-3 rounded-full bg-muted/10 px-4 py-2">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe or paste your algorithm..."
              className="flex-1 resize-none border-none shadow-none bg-white/5 px-2 py-2 text-sm leading-6 text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:ring-0 max-h-48 overflow-y-auto"
              disabled={disabled}
              style={{ minHeight: minTextareaHeight }}
            />

            <Button
              type="submit"
              disabled={disabled || !input.trim()}
              className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-primary hover:bg-primary/20 border border-primary/60"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>

          <p className="text-[11px] text-muted-foreground text-center">
            Press <kbd className="px-1 py-0.5 rounded bg-muted">Enter</kbd> to
            send - <kbd className="px-1 py-0.5 rounded bg-muted">Shift+Enter</kbd> for new line
          </p>
        </form>
      </Card>
    </div>
  );
}
