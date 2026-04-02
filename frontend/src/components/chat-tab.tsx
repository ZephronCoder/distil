"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.arbos.life";

interface Message {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  tps?: number;
  genTime?: number;
  tokens?: number;
}

interface ChatStatus {
  available: boolean;
  king_uid: number | null;
  king_model: string | null;
  server_running: boolean;
  note: string;
}

function ThinkingBlock({ thinking, elapsed }: { thinking: string; elapsed?: number }) {
  const [expanded, setExpanded] = useState(false);
  if (!thinking) return null;

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-[10px] text-muted-foreground/40 hover:text-muted-foreground/60 transition-colors"
      >
        <span className={`transition-transform ${expanded ? "rotate-90" : ""}`}>▶</span>
        <span>Thought{elapsed != null ? ` for ${elapsed.toFixed(1)}s` : ""}</span>
      </button>
      {expanded && (
        <div className="mt-1 pl-3 border-l border-muted-foreground/10 text-xs text-muted-foreground/30 whitespace-pre-wrap max-h-[200px] overflow-y-auto">
          {thinking}
        </div>
      )}
    </div>
  );
}

export function ChatTab() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [streamTps, setStreamTps] = useState(0);
  // Streaming state
  const [streamThinking, setStreamThinking] = useState("");
  const [streamContent, setStreamContent] = useState("");
  const [streamPhase, setStreamPhase] = useState<"thinking" | "answer">("thinking");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 8000);
      const res = await fetch(`${API_BASE}/api/chat/status`, {
        cache: "no-store",
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (res.ok) {
        setStatus(await res.json());
        setError(null);
      } else {
        setStatus({ available: false, king_uid: null, king_model: null, server_running: false, note: `API returned ${res.status}` });
      }
    } catch {
      setStatus({ available: false, king_uid: null, king_model: null, server_running: false, note: "Cannot reach chat API" });
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 30000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, streamContent, streamThinking]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = { role: "user", content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    setError(null);
    setElapsed(0);
    setStreamTps(0);
    setStreamThinking("");
    setStreamContent("");
    setStreamPhase("thinking");

    const t0 = Date.now();
    timerRef.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
          max_tokens: 2048,
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.error || `HTTP ${res.status}`);
        return;
      }

      const contentType = res.headers.get("content-type") || "";

      if (contentType.includes("text/event-stream")) {
        // SSE streaming
        const reader = res.body?.getReader();
        if (!reader) { setError("No response stream"); return; }

        const decoder = new TextDecoder();
        let buffer = "";
        let thinking = "";
        let content = "";
        let phase: "thinking" | "answer" = "thinking";
        let lastTps = 0;
        let lastTokens = 0;
        let lastGenTime = 0;
        let thinkEndTime: number | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") break;

            try {
              const evt = JSON.parse(raw);
              if (evt.error) { setError(evt.error); continue; }

              const delta = evt.choices?.[0]?.delta;
              const finishReason = evt.choices?.[0]?.finish_reason;
              const tps = evt.usage?.tokens_per_second;
              if (tps) { lastTps = tps; setStreamTps(tps); }
              if (evt.usage?.completion_tokens) lastTokens = evt.usage.completion_tokens;
              if (evt.usage?.generation_time_s) lastGenTime = evt.usage.generation_time_s;

              if (delta?.phase === "answer" && phase === "thinking") {
                phase = "answer";
                thinkEndTime = (Date.now() - t0) / 1000;
                setStreamPhase("answer");
              }

              if (delta?.content) {
                if (phase === "thinking") {
                  thinking += delta.content;
                  setStreamThinking(thinking);
                } else {
                  content += delta.content;
                  setStreamContent(content);
                }
              }

              if (finishReason === "stop") {
                // Server sends final thinking/answer split for models without <think> tags
                const finalThinking = evt.thinking || thinking || undefined;
                const finalAnswer = evt.answer || content || "(empty response)";

                const assistantMsg: Message = {
                  role: "assistant",
                  content: finalAnswer,
                  thinking: finalThinking,
                  tps: lastTps || undefined,
                  genTime: lastGenTime || undefined,
                  tokens: lastTokens || undefined,
                };
                setMessages(prev => [...prev, assistantMsg]);
                setStreamThinking("");
                setStreamContent("");
              }
            } catch {}
          }
        }
      } else {
        // Fallback: non-streaming JSON response
        const data = await res.json();
        if (data.error) {
          setError(data.error);
        } else if (data.response) {
          setMessages(prev => [...prev, {
            role: "assistant",
            content: data.response,
            thinking: data.thinking || undefined,
            tps: data.usage?.tokens_per_second,
            genTime: data.usage?.generation_time_s,
            tokens: data.usage?.completion_tokens,
          }]);
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(`Failed to reach API: ${e instanceof Error ? e.message : "unknown"}`);
      }
    } finally {
      setLoading(false);
      if (timerRef.current) clearInterval(timerRef.current);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    // Finalize whatever we have
    if (streamContent || streamThinking) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: streamContent || "(stopped)",
        thinking: streamThinking || undefined,
        tps: streamTps || undefined,
      }]);
      setStreamThinking("");
      setStreamContent("");
    }
    setLoading(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const available = status?.available ?? false;
  const kingModel = status?.king_model;

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] max-h-[600px]">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border/20">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Chat with the King</span>
          {kingModel && (
            <a
              href={`https://huggingface.co/${kingModel}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] font-mono text-blue-400/50 hover:text-blue-400/80 transition-colors"
            >
              {kingModel} ↗
            </a>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full ${available ? "bg-emerald-400" : status?.server_running === false ? "bg-red-400" : "bg-yellow-400"}`} />
          <span className="text-[10px] font-mono text-muted-foreground/50">
            {available ? "ready" : status?.server_running === false ? "offline" : "loading..."}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-4 space-y-4 min-h-0">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            {!available && status?.server_running === false ? (
              <>
                <span className="text-4xl">🔌</span>
                <p className="text-sm text-red-400/80 font-medium">Chat is currently offline</p>
                <p className="text-xs text-muted-foreground/40 max-w-sm">
                  The king model server is not running. It will be available when an evaluation completes and the king model is loaded.
                </p>
                {status?.note && (
                  <p className="text-xs text-orange-400/50 mt-1">{status.note}</p>
                )}
              </>
            ) : (
              <>
                <span className="text-4xl">👑</span>
                <p className="text-sm text-muted-foreground/60">Talk to the current king model</p>
                <p className="text-xs text-muted-foreground/40 max-w-sm">
                  This {kingModel ? <span className="font-mono">{kingModel}</span> : "distilled"} model
                  is running on a dedicated GPU. Responses stream in real-time.
                </p>
                {!available && status?.note && (
                  <p className="text-xs text-orange-400/60 mt-2">{status.note}</p>
                )}
              </>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm ${
              msg.role === "user"
                ? "bg-blue-500/15 border border-blue-500/20 text-foreground"
                : "bg-card/30 border border-border/20 text-foreground/90"
            }`}>
              {msg.role === "assistant" && msg.thinking && (
                <ThinkingBlock thinking={msg.thinking} />
              )}
              <div className="whitespace-pre-wrap break-words">{msg.content}</div>
              {msg.role === "assistant" && msg.tps != null && (
                <div className="flex items-center gap-3 mt-2 pt-1.5 border-t border-border/10 text-[9px] text-muted-foreground/30 font-mono">
                  <span>⚡ {msg.tps} tok/s</span>
                  {msg.tokens != null && <span>{msg.tokens} tokens</span>}
                  {msg.genTime != null && <span>{msg.genTime}s</span>}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Streaming in-progress message */}
        {loading && (streamContent || streamThinking) && (
          <div className="flex justify-start">
            <div className="max-w-[80%] bg-card/30 border border-border/20 rounded-xl px-4 py-2.5 text-sm text-foreground/90">
              {streamThinking && streamPhase === "thinking" && (
                <div className="mb-2">
                  <div className="flex items-center gap-1.5 text-[10px] text-blue-400/50 mb-1">
                    <span className="animate-pulse">●</span>
                    <span>Thinking...</span>
                    <span className="font-mono tabular-nums">{elapsed.toFixed(1)}s</span>
                  </div>
                  <div className="pl-3 border-l border-blue-400/10 text-xs text-muted-foreground/30 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
                    {streamThinking}
                  </div>
                </div>
              )}
              {streamThinking && streamPhase === "answer" && (
                <ThinkingBlock thinking={streamThinking} elapsed={elapsed} />
              )}
              {streamContent && (
                <div className="whitespace-pre-wrap break-words">{streamContent}</div>
              )}
              <div className="flex items-center gap-3 mt-2 pt-1.5 border-t border-border/10 text-[9px] text-muted-foreground/30 font-mono">
                {streamTps > 0 && <span>⚡ {streamTps} tok/s</span>}
                <span className="tabular-nums">{elapsed.toFixed(1)}s</span>
              </div>
            </div>
          </div>
        )}

        {/* Loading dots (before any content arrives) */}
        {loading && !streamContent && !streamThinking && (
          <div className="flex justify-start">
            <div className="bg-card/30 border border-border/20 rounded-xl px-4 py-2.5 text-sm">
              <div className="flex items-center gap-2">
                <span className="flex gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: "0.2s" }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: "0.4s" }} />
                </span>
                <span className="text-[10px] text-muted-foreground/40 font-mono tabular-nums">{elapsed.toFixed(1)}s</span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-center">
            <div className="text-xs text-red-400/70 bg-red-400/5 border border-red-400/20 rounded-lg px-3 py-2 max-w-sm">
              {error}
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="pt-3 border-t border-border/20">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder={available ? "Message the king model..." : "Chat server starting..."}
            disabled={!available || loading}
            className="flex-1 h-10 rounded-xl border border-border/30 bg-card/20 px-4 text-sm font-mono text-foreground placeholder:text-muted-foreground/30 focus:outline-none focus:border-blue-400/40 disabled:opacity-40"
          />
          {loading ? (
            <button
              onClick={stopGeneration}
              className="h-10 px-5 rounded-xl bg-red-500/15 border border-red-500/25 text-sm text-red-400 font-medium hover:bg-red-500/25 transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={sendMessage}
              disabled={!available || !input.trim()}
              className="h-10 px-5 rounded-xl bg-blue-500/15 border border-blue-500/25 text-sm text-blue-400 font-medium hover:bg-blue-500/25 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Send
            </button>
          )}
        </div>
        {messages.length > 0 && (
          <button
            onClick={() => { setMessages([]); setError(null); }}
            className="mt-2 text-[10px] text-muted-foreground/40 hover:text-muted-foreground/60 transition-colors"
          >
            Clear conversation
          </button>
        )}
      </div>
    </div>
  );
}
