import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownRendererProps {
  content: string;
}

// Render markdown chat content with dark-mode friendly theming and semantic components.
const markdownComponents: Components = {
  p: ({ children }) => (
    <p className="text-sm leading-relaxed text-foreground">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="text-foreground font-semibold">{children}</strong>
  ),
  em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
  a: ({ children, href }) => (
    <a
      href={href}
      className="text-primary underline underline-offset-4 hover:text-primary/80"
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  ),
  ul: ({ children }) => (
    <ul className="list-disc list-inside space-y-1 text-sm text-foreground">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-inside space-y-1 text-sm text-foreground">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="ml-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-primary/40 pl-3 text-sm text-muted-foreground">
      {children}
    </blockquote>
  ),
  code: ({ children, className, ...props }) => {
    // Check if code is inline by inspecting props or node structure
    const isInline = !className?.includes('language-');
    
    if (isInline) {
      return (
        <code
          className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-xs text-foreground"
          {...props}
        >
          {children}
        </code>
      );
    }

    return (
      <pre className="overflow-x-auto rounded-lg border border-border/60 bg-muted/20 p-3">
        <code className={`font-mono text-xs leading-relaxed text-foreground ${className || ""}`} {...props}>
          {children}
        </code>
      </pre>
    );
  },
};

// Provide a reusable markdown renderer to keep markdown logic isolated from chat components.
export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      className="space-y-3 text-sm leading-relaxed text-foreground"
      components={markdownComponents}
      remarkPlugins={[remarkGfm]}
    >
      {content}
    </ReactMarkdown>
  );
}
