"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

/** Custom event name dispatched on each auto-refresh cycle */
export const REFRESH_EVENT = "dashboard-refresh";

/**
 * Auto-refresh the page by triggering a Next.js router refresh every `intervalMs`.
 * This re-runs server components without a full browser reload.
 * Also dispatches a custom "dashboard-refresh" event so client components
 * (charts, tables) can re-fetch their own data.
 */
export function AutoRefresh({ intervalMs = 30000 }: { intervalMs?: number }) {
  const router = useRouter();

  useEffect(() => {
    const id = setInterval(() => {
      router.refresh();
      window.dispatchEvent(new CustomEvent(REFRESH_EVENT));
    }, intervalMs);
    return () => clearInterval(id);
  }, [router, intervalMs]);

  return null;
}

/**
 * Hook that returns a refreshKey that increments on each auto-refresh cycle.
 * Use as a dependency in useEffect to trigger re-fetches.
 */
export function useRefreshKey(): number {
  const [key, setKey] = useState(0);
  useEffect(() => {
    const handler = () => setKey((k: number) => k + 1);
    window.addEventListener(REFRESH_EVENT, handler);
    return () => window.removeEventListener(REFRESH_EVENT, handler);
  }, []);
  return key;
}
