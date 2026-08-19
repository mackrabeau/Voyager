import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  comparisonIssue,
  loadCatalog,
  loadReplayInitial,
} from "./replay/ReplaySource";
import type { CatalogReplay } from "./replay/platformTypes";
import type { ReplayStatus } from "./replay/types";
import type { ReplayPlayerHandle } from "./components/ReplayPlayer";
import {
  comparisonSyncTarget,
  PERFECT_RUN_ID,
  technologyStage,
} from "./replay/islandPresentation";
import { appPath, currentRoutePath } from "./paths";

const ReplayPlayer = lazy(() =>
  import("./components/ReplayPlayer").then((module) => ({
    default: module.ReplayPlayer,
  })),
);
const DEFAULT_REPLAY = "island_feed_forward_progression_v1";
const MATCHED_RANDOM_REPLAY = "island_legal_random_matched_v1";

export function App() {
  const path = currentRoutePath();
  if (path === "/compare") return <ComparisonPage />;
  if (path.startsWith("/present/")) {
    return (
      <PresentationPage replayId={decodeURIComponent(path.slice("/present/".length))} />
    );
  }
  if (path.startsWith("/replays/")) {
    return <ReplayPage replayId={decodeURIComponent(path.slice("/replays/".length))} />;
  }
  return <ShowcasePage />;
}

function ShowcasePage() {
  const [catalogOpen, setCatalogOpen] = useState(false);
  return (
    <main className="page-shell">
      <header className="page-intro">
        <h1>VOYAGER</h1>
        <div className="intro-actions">
          <a className="perfect-run-action" href={appPath(`replays/${PERFECT_RUN_ID}`)}>
            <strong>WATCH PERFECT RUN</strong>
            <span>SCRIPTED PUBLIC-OBSERVATION ORACLE · 15 / 15 · 2 / 2 RESCUED</span>
          </a>
          <button type="button" onClick={() => setCatalogOpen((open) => !open)}>
            {catalogOpen ? "CLOSE RUNS" : "EXPLORE RUNS"}
          </button>
        </div>
      </header>
      {catalogOpen && <RunLibrary />}
      <PlayerFallback>
        <ReplayPlayer replayId={DEFAULT_REPLAY} />
      </PlayerFallback>
      <footer>
        <span>OFFICIAL STAGE 7 PPO · CURATED HELD-OUT REPLAY · NO RESCUE</span>
        <span>DRAG TO EXPLORE · PRESS I TO INSPECT</span>
      </footer>
    </main>
  );
}

function ReplayPage({ replayId }: { replayId: string }) {
  return (
    <main className="page-shell general-page">
      <CompactHeader replayId={replayId} />
      <PlayerFallback>
        <ReplayPlayer replayId={replayId} />
      </PlayerFallback>
      <footer>
        <a href={appPath("")}>SHOWCASE</a>
        <a href={appPath(`present/${encodeURIComponent(replayId)}`)}>PRESENTATION MODE</a>
      </footer>
    </main>
  );
}

function PresentationPage({ replayId }: { replayId: string }) {
  return (
    <main className="presentation-page">
      <PlayerFallback>
        <ReplayPlayer replayId={replayId} presentation />
      </PlayerFallback>
    </main>
  );
}

function ComparisonPage() {
  const params = new URLSearchParams(window.location.search);
  const leftId = params.get("left") ?? MATCHED_RANDOM_REPLAY;
  const rightId = params.get("right") ?? DEFAULT_REPLAY;
  const left = useRef<ReplayPlayerHandle>(null);
  const right = useRef<ReplayPlayerHandle>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [status, setStatus] = useState<Record<string, ReplayStatus>>({});
  const [compatible, setCompatible] = useState<string | null>(null);
  const syncing = useRef(false);

  useEffect(() => {
    comparisonIssue(leftId, rightId)
      .then((issue) => setCompatible(issue ?? ""))
      .catch((error: unknown) =>
        setCompatible(error instanceof Error ? error.message : "Runs are incompatible."),
      );
  }, [leftId, rightId]);

  useEffect(() => {
    const target = comparisonSyncTarget(status.left, status.right);
    if (
      target === null ||
      syncing.current
    ) {
      return;
    }
    syncing.current = true;
    left.current?.setPlaying(false);
    right.current?.setPlaying(false);
    left.current?.seekTo(target);
    right.current?.seekTo(target);
    window.setTimeout(() => {
      if (!status.left?.ended) left.current?.setPlaying(playing);
      if (!status.right?.ended) right.current?.setPlaying(playing);
      syncing.current = false;
    }, 120);
  }, [playing, status]);

  const both = (action: (player: ReplayPlayerHandle) => void) => {
    if (left.current) action(left.current);
    if (right.current) action(right.current);
  };
  const leftStatus = status.left;
  const rightStatus = status.right;
  const achievementDelta =
    (rightStatus?.achievements?.length ?? 0) - (leftStatus?.achievements?.length ?? 0);
  return (
    <main className="compare-page">
      <header className="compare-header">
        <a href={appPath("")}>← VOYAGER</a>
        <div>
          <span>SEED-MATCHED POLICY COMPARISON</span>
          <h1>{leftId.replaceAll("_", " ")} <i>VS</i> {rightId.replaceAll("_", " ")}</h1>
        </div>
        <div className="compare-deltas">
          <strong>{achievementDelta >= 0 ? "+" : ""}{achievementDelta} ACHIEVEMENTS</strong>
          <span>{rightStatus?.technologyStage ?? "STRANDED"} VS {leftStatus?.technologyStage ?? "STRANDED"}</span>
          <span>{rightStatus?.alive ?? 0} VS {leftStatus?.alive ?? 0} SURVIVORS</span>
        </div>
      </header>
      {compatible ? (
        <div className="comparison-error">
          <strong>THESE WORLDS CANNOT BE SYNCHRONIZED</strong>
          <p>{compatible}</p>
          <a href={appPath("")}>RETURN TO SHOWCASE</a>
        </div>
      ) : (
        <>
          <div className="comparison-grid">
            <div>
              <span className="side-label">A · {leftId.replaceAll("_", " ")}{leftStatus?.ended ? " · RUN ENDED" : ""}</span>
              <PlayerFallback>
                <ReplayPlayer
                  ref={left}
                  replayId={leftId}
                  compact
                  hideControls
                  onStatus={(value) =>
                    setStatus((current) => ({ ...current, left: value }))
                  }
                />
              </PlayerFallback>
            </div>
            <div>
              <span className="side-label">B · {rightId.replaceAll("_", " ")}{rightStatus?.ended ? " · RUN ENDED" : ""}</span>
              <PlayerFallback>
                <ReplayPlayer
                  ref={right}
                  replayId={rightId}
                  compact
                  hideControls
                  onStatus={(value) =>
                    setStatus((current) => ({ ...current, right: value }))
                  }
                />
              </PlayerFallback>
            </div>
          </div>
          <div className="comparison-controls">
            <button
              type="button"
              onClick={() => {
                const next = !playing;
                setPlaying(next);
                if (!leftStatus?.ended) left.current?.setPlaying(next);
                if (!rightStatus?.ended) right.current?.setPlaying(next);
              }}
            >{playing ? "Ⅱ PAUSE BOTH" : "▶ PLAY BOTH"}</button>
            <button type="button" onClick={() => both((player) => player.stepBy(-1))}>‹ STEP</button>
            <button type="button" onClick={() => both((player) => player.stepBy(1))}>STEP ›</button>
            <button type="button" onClick={() => both((player) => player.restart())}>↺ RESTART</button>
            {[0.5, 1, 2, 4].map((value) => (
              <button
                type="button"
                key={value}
                className={speed === value ? "active" : ""}
                onClick={() => {
                  setSpeed(value);
                  both((player) => player.setSpeed(value));
                }}
              >{value}×</button>
            ))}
          </div>
        </>
      )}
    </main>
  );
}

function RunLibrary() {
  const [runs, setRuns] = useState<CatalogReplay[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    loadCatalog().then(setRuns).catch(() => setError("START THE VOYAGER SERVER TO BROWSE RUNS"));
  }, []);
  const visible = runs.filter((run) =>
    `${run.policy_id} ${run.seed} ${run.tags.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <section className="run-library">
      <div className="library-heading">
        <div><span>FROZEN BENCHMARK REPLAYS</span><h2>CHOOSE A CIVILIZATION</h2></div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="SEARCH POLICY, SEED, TAG"
          aria-label="Search replays"
        />
      </div>
      {error && <p className="library-error">{error}</p>}
      <div className="run-cards">
        {visible.map((run) => (
          <a
            href={appPath(`replays/${encodeURIComponent(run.replay_id)}`)}
            key={run.replay_id}
            className={`${run.tags.includes("showcase") ? "showcase-card" : ""} ${run.replay_id === PERFECT_RUN_ID ? "perfect-card" : ""}`}
          >
            <IslandThumbnail replayId={run.replay_id} />
            <small>{run.policy_kind.toUpperCase()} · SEED {run.seed.toLocaleString()}</small>
            <strong>{run.policy_id.replaceAll("_", " ")}</strong>
            <RunSummary run={run} />
            <em>
              {run.replay_id === PERFECT_RUN_ID
                ? "PERFECT RUN · SCRIPTED PUBLIC-OBSERVATION ORACLE"
                : run.tags.includes("showcase")
                ? "DEFAULT SHOWCASE"
                : run.tags.includes("failure-case")
                  ? "HONEST FAILURE"
                  : run.inference_mode?.toUpperCase() ?? "BASELINE"}
            </em>
          </a>
        ))}
      </div>
    </section>
  );
}

function RunSummary({ run }: { run: CatalogReplay }) {
  if (!run.tags.includes("island-benchmark")) {
    return (
      <div className="run-stats">
        <span>{run.terminal_summary.survivors} SURVIVED</span>
        <span>{run.terminal_summary.achievements.length} ACHIEVEMENTS</span>
      </div>
    );
  }
  return (
    <div className="run-stats">
      <span>{technologyStage(run.terminal_summary.achievements)}</span>
      <span>{run.terminal_summary.achievements.length} / 15 ACHIEVEMENTS</span>
      <span>{run.terminal_summary.rescue_success
        ? `${run.terminal_summary.survivors} / 2 RESCUED`
        : run.terminal_summary.survivors > 0
          ? `${run.terminal_summary.survivors} / 2 SURVIVED`
          : "EXPEDITION FAILED"}</span>
    </div>
  );
}

interface ThumbnailInitial {
  terrain: string[][];
  camp: { x: number; y: number };
}

const thumbnailInitialCache = new Map<string, Promise<ThumbnailInitial>>();

function loadThumbnailInitial(replayId: string) {
  const cached = thumbnailInitialCache.get(replayId);
  if (cached) return cached;
  const request = loadReplayInitial(replayId)
    .then((initial) => initial as ThumbnailInitial)
    .catch((error: unknown) => {
      thumbnailInitialCache.delete(replayId);
      throw error;
    });
  thumbnailInitialCache.set(replayId, request);
  return request;
}

function IslandThumbnail({ replayId }: { replayId: string }) {
  const drawThumbnail = useCallback(
    (target: HTMLCanvasElement | null) => {
      if (!target) return;
      void loadThumbnailInitial(replayId)
        .then((initial) => {
          if (!target.isConnected) return;
          const context = target.getContext("2d");
          if (!context) return;
          context.imageSmoothingEnabled = false;
          const colors: Record<string, string> = {
            water: "#287d86",
            beach: "#dba45e",
            grass: "#4a9346",
            forest: "#286a39",
            quarry: "#68736a",
          };
          initial.terrain.forEach((row, y) =>
            row.forEach((terrain, x) => {
              context.fillStyle = colors[terrain] ?? "#8a5ca0";
              context.fillRect(x, y, 1, 1);
            }),
          );
          context.fillStyle = "#f0c65d";
          context.fillRect(initial.camp.x - 1, initial.camp.y - 1, 3, 3);
          target.dataset.ready = "true";
        })
        .catch((error: unknown) => {
          target.dataset.error =
            error instanceof Error ? error.message : "Thumbnail unavailable";
        });
    },
    [replayId],
  );
  return (
    <canvas
      ref={drawThumbnail}
      className="island-thumbnail"
      width={48}
      height={48}
      aria-hidden="true"
    />
  );
}

function CompactHeader({ replayId }: { replayId: string }) {
  const perfect = replayId === PERFECT_RUN_ID;
  return (
    <header className="compact-header">
      <a href={appPath("")}><span className="voyager-mark">V</span> VOYAGER</a>
      <div>
        <span>{perfect ? "PERFECT RUN · SCRIPTED PUBLIC-OBSERVATION ORACLE" : "SAVED REPLAY"}</span>
        <h1>{perfect ? "15 / 15 ACHIEVEMENTS · 2 / 2 RESCUED" : replayId.replaceAll("_", " ")}</h1>
      </div>
      <nav><a href={appPath("")}>RUNS</a><a href={appPath(`compare?left=${MATCHED_RANDOM_REPLAY}&right=${encodeURIComponent(replayId)}`)}>COMPARE</a></nav>
    </header>
  );
}

function PlayerFallback({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="replay-frame app-loading" aria-label="Loading replay player">
          <span>LOADING REPLAY...</span>
        </div>
      }
    >{children}</Suspense>
  );
}
