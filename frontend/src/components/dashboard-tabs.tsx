"use client";

import { useState } from "react";
import type { MinerEntry, ModelInfo, ScoresResponse, ScoreHistoryEntry, EvalProgress } from "@/lib/api";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { MinersTab } from "@/components/miners-tab";
import { H2hHistory } from "@/components/h2h-round-table";
import { GpuLogs } from "@/components/gpu-logs";
import { EvalProgressBar } from "@/components/eval-progress";
import { ScoreTrend } from "@/components/score-trend";
import { ValidatorStatus } from "@/components/validator-status";
import { DocsTab } from "@/components/docs-tab";
import { ChatTab } from "@/components/chat-tab";
import { BenchmarksTab } from "@/components/benchmarks-tab";
import { KingHistory } from "@/components/king-history";

interface DashboardTabsProps {
  miners: MinerEntry[];
  scores: ScoresResponse | null;
  modelInfoMap: Record<string, ModelInfo>;
  currentBlock: number;
  taoUsd: number;
  minersTaoDay: number;
  pendingMiners: MinerEntry[];
  evalProgress: EvalProgress | null;
  lastEvalTime: number;
  tempoSeconds: number;
  history: ScoreHistoryEntry[];
  kingUid: number | null;
  kingH2hKl: number | null;
}

export function DashboardTabs({
  miners,
  scores,
  modelInfoMap,
  currentBlock,
  taoUsd,
  minersTaoDay,
  pendingMiners,
  evalProgress,
  lastEvalTime,
  tempoSeconds,
  history,
  kingUid,
  kingH2hKl,
}: DashboardTabsProps) {
  const [activeTab, setActiveTab] = useState("eval-rounds");
  const kingMiner = kingUid != null ? miners.find(m => m.uid === kingUid) : null;

  const EPSILON = 0.01;
  const scoreToBeat = kingH2hKl != null ? kingH2hKl * (1 - EPSILON) : null;

  return (
    <div className="space-y-3">
      <ValidatorStatus
        kingUid={kingUid}
        kingModel={kingMiner?.model}
        onViewDetails={() => setActiveTab("live")}
      />

      <Tabs value={activeTab} onValueChange={(v: unknown) => setActiveTab(String(v))}>
        <TabsList variant="line" className="w-full justify-start border-b border-border/20 pb-0">
          <TabsTrigger value="eval-rounds" className="text-sm px-4 py-2">
            ⚔️ Rounds
          </TabsTrigger>
          <TabsTrigger value="live" className="text-sm px-4 py-2">
            📡 Live
          </TabsTrigger>
          <TabsTrigger value="miners" className="text-sm px-4 py-2">
            ⛏️ Miners
          </TabsTrigger>
          <TabsTrigger value="chart" className="text-sm px-4 py-2">
            📊 Chart
          </TabsTrigger>
          <TabsTrigger value="benchmarks" className="text-sm px-4 py-2">
            🏆 Benchmarks
          </TabsTrigger>
          <TabsTrigger value="chat" className="text-sm px-4 py-2">
            💬 Chat
          </TabsTrigger>
          <TabsTrigger value="docs" className="text-sm px-4 py-2">
            📖 Docs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="eval-rounds" className="pt-4">
          <H2hHistory />
        </TabsContent>

        <TabsContent value="live" className="pt-4 space-y-4">
          <EvalProgressBar />
          <GpuLogs />
        </TabsContent>

        <TabsContent value="miners" className="pt-4">
          <MinersTab
            miners={miners}
            scores={scores}
            modelInfoMap={modelInfoMap}
            currentBlock={currentBlock}
            taoUsd={taoUsd}
            minersTaoDay={minersTaoDay}
          />
        </TabsContent>

        <TabsContent value="chart" className="pt-4 space-y-4">
          {history.length > 0 ? (
            <ScoreTrend history={history} />
          ) : (
            <div className="text-center text-sm text-muted-foreground py-8">
              No score history available yet.
            </div>
          )}
          <KingHistory />
        </TabsContent>

        <TabsContent value="benchmarks" className="pt-4">
          <BenchmarksTab />
        </TabsContent>

        <TabsContent value="chat" className="pt-4">
          <ChatTab />
        </TabsContent>

        <TabsContent value="docs" className="pt-4">
          <DocsTab scoreToBeat={scoreToBeat} kingKl={kingH2hKl} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
