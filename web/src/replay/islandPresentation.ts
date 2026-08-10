import type { ReplayEvent } from "./platformTypes";
import type {
  ReplayArtifact,
  ReplayStructure,
  ReplayTime,
} from "./types";

export const ISLAND_BENCHMARK_ID = "voyager_island_v1_procedural_baselines_v1";
export const PERFECT_RUN_ID = "island_oracle_rescue_v1";

const TECHNOLOGY_ORDER = [
  ["collect_food", "collect_wood", "collect_stone"],
  ["deposit_wood", "deposit_stone"],
  ["build_workbench"],
  ["craft_axe", "craft_spear"],
  ["build_campfire"],
  ["hunt_deer", "cook_meat"],
  ["build_shelter"],
  ["build_beacon"],
  ["rescue_both"],
] as const;

const STAGE_LABELS = [
  "STRANDED",
  "GATHERING",
  "DELIVERY",
  "WORKBENCH",
  "TOOLS",
  "CAMPFIRE",
  "SURVIVAL",
  "SHELTER",
  "BEACON",
  "RESCUED",
];

const MARKER_ACHIEVEMENTS = [
  "build_workbench",
  "craft_axe",
  "craft_spear",
  "build_campfire",
  "hunt_deer",
  "cook_meat",
  "build_shelter",
  "both_survive_first_night",
  "build_beacon",
  "rescue_both",
] as const;
const MARKER_ACHIEVEMENT_SET = new Set<string>(MARKER_ACHIEVEMENTS);

const ACHIEVEMENT_LABELS: Record<string, string> = {
  build_workbench: "WORKBENCH COMPLETE",
  craft_axe: "AXE CRAFTED",
  craft_spear: "SPEAR CRAFTED",
  build_campfire: "CAMPFIRE COMPLETE",
  hunt_deer: "FIRST HUNT",
  cook_meat: "FIRST COOKED MEAL",
  build_shelter: "SHELTER COMPLETE",
  both_survive_first_night: "FIRST NIGHT SURVIVED",
  build_beacon: "BEACON COMPLETE",
  rescue_both: "RESCUE COMPLETE",
};

export interface IslandObjective {
  id: "workbench" | "campfire" | "shelter" | "beacon" | "extraction" | "rescued" | "failed";
  label: string;
  progress: number;
}

export interface TimelineMarker {
  tick: number;
  type: string;
  label: string;
}

export interface StructureVisualState {
  stage: 0 | 1 | 2;
  condition: "normal" | "degraded" | "disabled";
}

export interface FinalePresentation {
  title: string;
  caption: string;
  success: boolean;
}

export function isIslandReplay(replay: ReplayArtifact | null): boolean {
  return Boolean(
    replay &&
      (replay.source.benchmark_id === ISLAND_BENCHMARK_ID ||
        replay.schema_version === "stage7_replay_2.3.0"),
  );
}

export function technologyStage(achievements: readonly string[]): string {
  const unlocked = new Set(achievements);
  let level = 0;
  TECHNOLOGY_ORDER.forEach((requirements, index) => {
    if (requirements.some((achievement) => unlocked.has(achievement))) level = index + 1;
  });
  return STAGE_LABELS[level];
}

export function structureVisualState(structure: ReplayStructure): StructureVisualState {
  const condition =
    (structure.condition ?? 100) <= 0
      ? "disabled"
      : (structure.condition ?? 100) < 50
        ? "degraded"
        : "normal";
  const stage = structure.complete ? 2 : structure.progress > 0 ? 1 : 0;
  return { stage, condition };
}

export function islandObjective(
  structures: readonly ReplayStructure[],
  achievementSteps: Readonly<Record<string, number>>,
  tick: number,
  ended: boolean,
  rescueSuccess: boolean,
): IslandObjective {
  if (rescueSuccess) return { id: "rescued", label: "RESCUED", progress: 1 };
  if (ended) return { id: "failed", label: "EXPEDITION FAILED", progress: 0 };
  for (const [id, label] of [
    ["workbench", "WORKBENCH"],
    ["campfire", "CAMPFIRE"],
    ["shelter", "SHELTER"],
    ["beacon", "BEACON"],
  ] as const) {
    const structure = structures.find((value) => value.type === id);
    if (!structure?.complete) {
      return { id, label, progress: Math.max(0, Math.min(1, structure?.progress ?? 0)) };
    }
  }
  const beaconTick = achievementSteps.build_beacon ?? tick;
  return {
    id: "extraction",
    label: "EXTRACTION",
    progress: Math.max(0, Math.min(1, (tick - beaconTick) / 100)),
  };
}

export function phaseLabel(time: ReplayTime | null | undefined): string | null {
  if (!time) return null;
  return time.phase === "night"
    ? `NIGHT ${time.day}`
    : `DAY ${time.day} · ${time.phase.toUpperCase()}`;
}

export function buildIslandMarkers(
  events: readonly ReplayEvent[],
  achievementSteps: Readonly<Record<string, number>>,
  maxSteps: number,
): TimelineMarker[] {
  const candidates: Array<TimelineMarker & { priority: number }> = [];
  MARKER_ACHIEVEMENTS.forEach((achievement, index) => {
    const tick = achievementSteps[achievement];
    if (tick !== undefined) {
      candidates.push({
        tick,
        type: achievement,
        label: ACHIEVEMENT_LABELS[achievement],
        priority: index + 20,
      });
    }
  });
  events.forEach((event) => {
    if (event.type === "agent_died" || event.type === "stalkers_spawned") {
      candidates.push({
        tick: event.tick,
        type: event.type,
        label: eventLabel(event),
        priority: event.type === "agent_died" ? 50 : 18,
      });
    }
  });
  candidates.push({ tick: maxSteps, type: "finale", label: "FINALE", priority: 100 });
  candidates.sort((left, right) => left.tick - right.tick || right.priority - left.priority);
  const clustered: typeof candidates = [];
  candidates.forEach((candidate) => {
    const previous = clustered.at(-1);
    if (previous && candidate.tick === previous.tick && candidate.type !== previous.type) {
      previous.label = previous.type === "finale"
        ? `${candidate.label} · ${previous.label}`
        : `${previous.label} · ${candidate.label}`;
      if (candidate.priority > previous.priority) {
        previous.type = candidate.type;
        previous.priority = candidate.priority;
      }
      return;
    }
    const previousIsAchievement = previous && MARKER_ACHIEVEMENT_SET.has(previous.type);
    const candidateIsAchievement = MARKER_ACHIEVEMENT_SET.has(candidate.type);
    if (previous && previousIsAchievement && candidateIsAchievement && candidate.tick - previous.tick <= 8) {
      if (candidate.priority > previous.priority) clustered[clustered.length - 1] = candidate;
      return;
    }
    clustered.push(candidate);
  });
  return clustered.map(({ priority: _priority, ...marker }) => marker);
}

export function eventLabel(event: ReplayEvent): string {
  const achievement = event.payload.achievement ?? event.payload.achievement_id;
  if (achievement) return ACHIEVEMENT_LABELS[String(achievement)] ?? humanize(String(achievement));
  if (event.type === "structure_complete" && event.targets[0]) {
    return `${humanize(event.targets[0])} COMPLETE`;
  }
  if (event.type === "gather" && event.payload.item) {
    return `${humanize(String(event.payload.item))} GATHERED`;
  }
  const labels: Record<string, string> = {
    agent_died: "ISLANDER LOST",
    stalkers_spawned: "STALKERS EMERGE",
    stalker_attack: "STALKER ATTACK",
    creature_defeated: "CREATURE DEFEATED",
    ground_pile_created: "RAW MEAT DROPPED",
    craft_axe: "AXE CRAFTED",
    craft_spear: "SPEAR CRAFTED",
    cook_meat: "MEAT COOKED",
  };
  return labels[event.type] ?? humanize(event.type);
}

export function islandFinale(
  summary: ReplayArtifact["summary"],
  population: number,
): FinalePresentation {
  const achievements = summary.achievements.length;
  if (summary.rescue_success) {
    return {
      title: "RESCUE COMPLETE",
      caption: `${summary.survivors} / ${population} RESCUED · ${achievements} / 15 ACHIEVEMENTS`,
      success: true,
    };
  }
  return {
    title: summary.survivors > 0
      ? `${summary.survivors} / ${population} SURVIVED`
      : "EXPEDITION FAILED",
    caption: `${technologyStage(summary.achievements)} · ${achievements} / 15 ACHIEVEMENTS`,
    success: false,
  };
}

export function comparisonSyncTarget(
  left: { step: number; ended: boolean } | undefined,
  right: { step: number; ended: boolean } | undefined,
): number | null {
  if (!left || !right || left.ended || right.ended || Math.abs(left.step - right.step) <= 1) {
    return null;
  }
  return Math.max(left.step, right.step);
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").toUpperCase();
}
