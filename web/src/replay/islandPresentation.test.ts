// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import type { ReplayEvent } from "./platformTypes";
import type { ReplayArtifact, ReplayStructure } from "./types";
import {
  buildIslandMarkers,
  comparisonSyncTarget,
  eventLabel,
  islandObjective,
  islandFinale,
  isIslandReplay,
  phaseLabel,
  structureVisualState,
  technologyStage,
} from "./islandPresentation";
import { islandTextureKeys } from "./pixelArt";

const structure = (
  type: string,
  progress: number,
  complete = progress >= 1,
  condition = 100,
): ReplayStructure => ({ id: type, type, x: 1, y: 1, progress, complete, condition });

describe("island presentation state", () => {
  it("advances the contextual objective through extraction and terminal outcomes", () => {
    const structures = [
      structure("workbench", 1),
      structure("campfire", 1),
      structure("shelter", 1),
      structure("beacon", 1),
    ];
    expect(islandObjective([structure("workbench", 0)], {}, 0, false, false)).toMatchObject({
      id: "workbench", progress: 0,
    });
    expect(islandObjective([structure("workbench", 0.45)], {}, 10, false, false)).toMatchObject({
      id: "workbench", progress: 0.45,
    });
    expect(islandObjective(structures, { build_beacon: 327 }, 377, false, false)).toEqual({
      id: "extraction", label: "EXTRACTION", progress: 0.5,
    });
    expect(islandObjective(structures, { build_beacon: 327 }, 427, true, true).id).toBe("rescued");
    expect(islandObjective(structures, { build_beacon: 327 }, 427, true, false).id).toBe("failed");
  });

  it("selects foundation, partial, complete, degraded, and disabled structure art", () => {
    expect(structureVisualState(structure("shelter", 0))).toEqual({ stage: 0, condition: "normal" });
    expect(structureVisualState(structure("shelter", 0.5, false))).toEqual({ stage: 1, condition: "normal" });
    expect(structureVisualState(structure("shelter", 1, true))).toEqual({ stage: 2, condition: "normal" });
    expect(structureVisualState(structure("shelter", 1, true, 40)).condition).toBe("degraded");
    expect(structureVisualState(structure("shelter", 1, true, 0)).condition).toBe("disabled");
  });

  it("derives technology and recorded island time", () => {
    expect(technologyStage([])).toBe("STRANDED");
    expect(technologyStage(["collect_wood", "build_workbench", "build_beacon"])).toBe("BEACON");
    expect(phaseLabel({ day: 2, tick_in_day: 210, phase: "night", ambient_light: 0.3 })).toBe("NIGHT 2");
    expect(phaseLabel({ day: 3, tick_in_day: 40, phase: "morning", ambient_light: 1 })).toBe("DAY 3 · MORNING");
  });

  it("presents the frozen oracle and learned failure finales honestly", () => {
    const base = {
      agent_steps: 0, dense_return: 0, deaths: 0, shelter_completion_step: 0,
      camp_stockpile: { food: 0, wood: 0, stone: 0 }, achievement_steps: {}, world_steps: 427,
    };
    const oracle = islandFinale({
      ...base, survivors: 2, rescue_success: true,
      achievements: Array.from({ length: 15 }, (_, index) => `achievement_${index}`),
    }, 2);
    expect(oracle).toEqual({
      title: "RESCUE COMPLETE",
      caption: "2 / 2 RESCUED · 15 / 15 ACHIEVEMENTS",
      success: true,
    });
    const learned = islandFinale({
      ...base, world_steps: 562, survivors: 0, rescue_success: false,
      achievements: ["collect_food", "build_workbench", "build_campfire"],
    }, 2);
    expect(learned.title).toBe("EXPEDITION FAILED");
    expect(learned.caption).toBe("CAMPFIRE · 3 / 15 ACHIEVEMENTS");
  });
});

describe("island timeline and labels", () => {
  const event = (tick: number, type: string, payload: Record<string, unknown> = {}): ReplayEvent => ({
    tick, type, payload, importance: 0, actors: [], targets: [], tags: [],
  });

  it("uses Replay 2.3 payload names and clusters nearby milestones", () => {
    const markers = buildIslandMarkers(
      [event(200, "stalkers_spawned"), event(562, "agent_died")],
      { build_workbench: 29, craft_axe: 30, build_campfire: 64 },
      562,
    );
    expect(markers).toEqual([
      { tick: 30, type: "craft_axe", label: "AXE CRAFTED" },
      { tick: 64, type: "build_campfire", label: "CAMPFIRE COMPLETE" },
      { tick: 200, type: "stalkers_spawned", label: "STALKERS EMERGE" },
      { tick: 562, type: "finale", label: "ISLANDER LOST · FINALE" },
    ]);
    expect(eventLabel(event(4, "achievement_unlocked", { achievement: "collect_food" }))).toBe("COLLECT FOOD");
    expect(eventLabel(event(200, "stalker_attack"))).toBe("STALKER ATTACK");
    expect(eventLabel({ ...event(64, "structure_complete"), targets: ["campfire"] })).toBe("CAMPFIRE COMPLETE");
  });
});

describe("renderer selection and texture completeness", () => {
  it("selects the island renderer from the manifest benchmark id", () => {
    const artifact = { source: { benchmark_id: "voyager_island_v1_procedural_baselines_v1" } };
    expect(isIslandReplay(artifact as ReplayArtifact)).toBe(true);
    expect(isIslandReplay({ source: { benchmark_id: "stage5_6_final_v1" } } as ReplayArtifact)).toBe(false);
  });

  it("freezes a shorter terminal comparison run while the longer run continues", () => {
    expect(comparisonSyncTarget({ step: 100, ended: false }, { step: 104, ended: false })).toBe(104);
    expect(comparisonSyncTarget({ step: 272, ended: true }, { step: 400, ended: false })).toBeNull();
    expect(comparisonSyncTarget({ step: 427, ended: true }, { step: 562, ended: true })).toBeNull();
  });

  it("declares every Stage 7 island texture without a placeholder key", () => {
    const keys = islandTextureKeys();
    expect(new Set(keys).size).toBe(keys.length);
    for (const required of [
      "workbench-0", "workbench-1", "workbench-2",
      "island-campfire-2-1", "beacon-2-1",
      "creature-island_deer-left-0", "creature-night_stalker-right-1",
      "ground-raw-meat-2", "stock-cooked-meat", "tool-axe", "tool-spear",
      "terrain-rocky_highland-0", "terrain-cave-1",
    ]) {
      expect(keys).toContain(required);
    }
    expect(keys.some((key) => key.includes("placeholder"))).toBe(false);
  });
});
