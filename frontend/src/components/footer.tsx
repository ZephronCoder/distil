import { TEACHER } from "@/lib/api";

export function Footer() {
  return (
    <footer className="border-t border-border/30 py-1.5 mt-auto">
      <div className="mx-auto max-w-7xl px-3 sm:px-4 flex items-center justify-between text-[11px] text-muted-foreground/50 font-mono">
        <span>💧 Distil · SN97 · Bittensor</span>
        <div className="flex items-center gap-4">
          <a href="https://x.com/arbos_born" target="_blank" rel="noopener noreferrer" className="hover:text-foreground/70 transition-colors">
            𝕏 ↗
          </a>
          <a href="https://github.com/unarbos/distil" target="_blank" rel="noopener noreferrer" className="hover:text-foreground/70 transition-colors">
            GitHub ↗
          </a>
          <a href={`https://huggingface.co/${TEACHER.model}`} target="_blank" rel="noopener noreferrer" className="hover:text-foreground/70 transition-colors">
            Teacher ↗
          </a>
          <a href="https://api.arbos.life/docs" target="_blank" rel="noopener noreferrer" className="hover:text-foreground/70 transition-colors">
            API Docs ↗
          </a>
          <a href="https://taomarketcap.com/subnets/97" target="_blank" rel="noopener noreferrer" className="hover:text-foreground/70 transition-colors">
            TaoMarketCap ↗
          </a>
        </div>
      </div>
    </footer>
  );
}
