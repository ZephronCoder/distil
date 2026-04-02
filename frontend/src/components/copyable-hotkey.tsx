"use client";

import { useState, useCallback } from "react";

interface CopyableHotkeyProps {
  hotkey: string;
  /** How many chars to show at start and end */
  chars?: number;
  className?: string;
}

export function CopyableHotkey({ hotkey, chars = 6, className = "" }: CopyableHotkeyProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(hotkey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Fallback for older browsers
      const ta = document.createElement("textarea");
      ta.value = hotkey;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  }, [hotkey]);

  const truncated = `${hotkey.slice(0, chars)}…${hotkey.slice(-chars)}`;

  return (
    <button
      onClick={handleCopy}
      title={`${hotkey}\n\nClick to copy`}
      className={`font-mono text-[11px] cursor-pointer transition-colors inline-flex items-center gap-1 ${
        copied
          ? "text-emerald-400"
          : "text-muted-foreground/50 hover:text-muted-foreground/80"
      } ${className}`}
    >
      <span>{copied ? "✓ copied" : truncated}</span>
      {!copied && (
        <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" className="opacity-30 hover:opacity-60 shrink-0">
          <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z" />
          <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z" />
        </svg>
      )}
    </button>
  );
}
