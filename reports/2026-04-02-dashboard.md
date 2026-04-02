# Dashboard Improvements — 2026-04-02

## Summary
Frontend dashboard improvements for the Distil SN97 dashboard (Next.js/TypeScript/Tailwind/shadcn/Recharts).

## Changes Made

### 1. Chart Auto-Refresh Fix
- **`auto-refresh.tsx`**: Now dispatches a custom `"dashboard-refresh"` event on each refresh cycle alongside `router.refresh()`
- **`useRefreshKey()` hook**: New export — returns an incrementing number on each refresh cycle. Client components use this as a `useEffect` dependency to trigger re-fetches
- **`h2h-round-table.tsx`**: H2H History component now uses `useRefreshKey()` to re-fetch round data when auto-refresh fires
- Note: `score-trend.tsx` receives `history` as a prop from the server component, so it auto-updates via `router.refresh()` already (server-side re-fetch)

### 2. Chat Fixes
- **Offline detection**: `fetchStatus` now has an 8-second timeout and proper error handling — if the API is unreachable or returns an error, status is set to `server_running: false` with a descriptive note
- **Offline UI**: When `server_running === false`, shows a "🔌 Chat is currently offline" message with explanation instead of just a disabled input
- **Error propagation**: API errors during status check are captured and shown to the user

### 3. Performance — History Limit
- **`api.ts`**: `fetchHistory()` now accepts a `limit` parameter (default 50), appends `?limit=` to the API URL, and also caps client-side to prevent rendering too many data points
- The page already uses `Promise.all` for all initial data fetches (was already parallelized)

### 4. Global vs H2H Score Clarity
- **Miners tab**: Added an info box explaining the difference between H2H Score (same-prompt, fair comparison) and Global Score (different prompts, not directly comparable)
- **Miner row**: Each score now shows a "Global" label underneath
- **H2H round table**: Header column labeled "H2H KL Score" with title tooltip

### 5. King History Timeline
- **New component**: `king-history.tsx` — fetches from `/api/king-history`, with fallback to extracting dethronements from `/api/h2h-history`
- Shows up to 10 most recent king changes with timeline UI (dots, connecting line)
- Displays: new king UID, model name, KL score, timestamp, time-ago
- Loading skeleton while fetching
- Integrated into the Chart tab (below the score trend chart)
- Listens to `useRefreshKey()` for auto-refresh

## Files Modified
- `src/components/auto-refresh.tsx` — Custom event dispatch + `useRefreshKey` hook
- `src/components/h2h-round-table.tsx` — Refresh key integration + H2H label
- `src/components/chat-tab.tsx` — Offline detection, error handling
- `src/components/miners-tab.tsx` — Score type info box + "Global" label
- `src/components/dashboard-tabs.tsx` — King history integration
- `src/lib/api.ts` — History limit parameter

## Files Created
- `src/components/king-history.tsx` — King history timeline component

## Build
Build passes cleanly: `npm run build` — no TypeScript or compilation errors.

## Commit
`c2216e4` on branch `improvements/validator-fixes-v2`
