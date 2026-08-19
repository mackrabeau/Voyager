import type {
  AgentRole,
  ReplayAgent,
  ReplayArtifact,
  ReplayResource,
  ReplayStructure,
  ReplayCreature,
  ReplayGroundPile,
  ReplayTime,
  ResourceType,
  TerrainType,
} from "./types";
import type {
  CatalogReplay,
  LoadedReplay,
  ReplayAction,
  ReplayEvent,
  ReplayManifestV2,
} from "./platformTypes";
import { appPath, STATIC_REPLAYS } from "../paths";

const LEGACY_PATH = `${import.meta.env.BASE_URL}replays/stage6_vertical_slice_v1.json`;
const knownTerrain = new Set(["water", "beach", "grass", "forest", "quarry", "rocky_highland", "cave"]);
const knownResources = new Set(["none", "food", "wood", "stone", "raw_meat", "cooked_meat"]);
const knownRoles = new Set(["forager", "woodcutter", "builder"]);

interface V2Initial {
  width: number;
  height: number;
  terrain: string[][];
  resources: Array<ReplayResource & { id: string }>;
  camp: ReplayArtifact["world"]["initial"]["camp"];
  agents: Array<Omit<ReplayAgent, "action" | "event">>;
  structures?: ReplayStructure[];
  creatures?: ReplayCreature[];
  ground_piles?: ReplayGroundPile[];
  time?: ReplayTime | null;
}

interface TimelineRecord {
  tick: number;
  actions: ReplayAction[];
  events: ReplayEvent[];
  state_delta: {
    camp: ReplayArtifact["world"]["initial"]["camp"];
    agents: Array<Omit<ReplayAgent, "action" | "event">>;
    resource_changes: Array<ReplayResource & { id: string }>;
    structures?: ReplayStructure[];
    creatures?: ReplayCreature[];
    ground_piles?: ReplayGroundPile[];
    time?: ReplayTime | null;
    extensions?: Record<string, unknown>;
  };
  achievements: string[];
  weather: { storm_active: boolean };
}

export interface ReplaySource {
  load(): Promise<LoadedReplay>;
  cancel?(): void;
}

export class LegacyReplaySource implements ReplaySource {
  async load(): Promise<LoadedReplay> {
    const artifact = await fetchJson<ReplayArtifact>(LEGACY_PATH);
    return {
      artifact,
      manifest: null,
      events: [],
      actions: [],
      metrics: {},
      camera: { version: "legacy", mode: "showcase", cues: [] },
      legacy: true,
    };
  }
}

export class ApiReplaySource implements ReplaySource {
  private readonly controller = new AbortController();

  constructor(private readonly replayId: string) {}

  cancel(): void {
    this.controller.abort();
  }

  async load(): Promise<LoadedReplay> {
    const base = STATIC_REPLAYS
      ? appPath(`static-api/replays/${encodeURIComponent(this.replayId)}`)
      : appPath(`api/v1/replays/${encodeURIComponent(this.replayId)}`);
    const signal = this.controller.signal;
    const manifestUrl = STATIC_REPLAYS ? `${base}/manifest.json` : base;
    const manifest = validateManifestV2(await fetchJson<unknown>(manifestUrl, signal));
    const [initial, timelinePayload, eventsPayload, metrics, camera] = await Promise.all([
      fetchJson<V2Initial>(`${base}/initial${STATIC_REPLAYS ? ".json" : ""}`, signal),
      STATIC_REPLAYS
        ? fetchJson<{ records: TimelineRecord[] }>(`${base}/timeline.json`, signal)
        : fetchTimelineChunks(base, manifest.world_steps, signal),
      fetchJson<{ events: ReplayEvent[] }>(
        STATIC_REPLAYS
          ? `${base}/events.json`
          : `${base}/events?start=0&end=${manifest.world_steps}`,
        signal,
      ),
      fetchJson<Record<string, unknown>>(
        `${base}/metrics${STATIC_REPLAYS ? ".json" : ""}`,
        signal,
      ),
      fetchJson<{ version: string; mode: string; cues: LoadedReplay["camera"]["cues"] }>(
        `${base}/camera${STATIC_REPLAYS ? ".json" : ""}`,
        signal,
      ).catch(() => ({ version: "automatic", mode: "automatic", cues: [] })),
    ]);
    const artifact = adaptV2ToRenderer(manifest, initial, timelinePayload.records);
    return {
      artifact,
      manifest,
      events: eventsPayload.events,
      actions: timelinePayload.records.flatMap((record) =>
        record.actions.map((action) => ({ tick: record.tick, ...action })),
      ),
      metrics,
      camera,
      legacy: false,
    };
  }
}

export function validateManifestV2(value: unknown): ReplayManifestV2 {
  if (!value || typeof value !== "object") {
    throw new Error("Replay manifest must be an object.");
  }
  const manifest = value as Partial<ReplayManifestV2>;
  if (
    typeof manifest.replay_id !== "string" ||
    typeof manifest.world_steps !== "number" ||
    typeof manifest.tick_rate !== "number" ||
    !manifest.versions ||
    !/^stage[67]_replay_2\./.test(String(manifest.versions.replay ?? ""))
  ) {
    throw new Error("Unsupported or malformed Stage 6 replay manifest.");
  }
  if (!manifest.source || !manifest.terminal_summary || !manifest.registries) {
    throw new Error("Replay manifest is missing source, summary, or registries.");
  }
  return manifest as ReplayManifestV2;
}

export async function loadCatalog(): Promise<CatalogReplay[]> {
  const url = STATIC_REPLAYS
    ? appPath("static-api/catalog.json")
    : appPath("api/v1/replays?limit=100");
  const response = await fetchJson<{ items: CatalogReplay[] }>(url);
  return response.items;
}

export async function loadReplayInitial(replayId: string): Promise<V2Initial> {
  const encoded = encodeURIComponent(replayId);
  const url = STATIC_REPLAYS
    ? appPath(`static-api/replays/${encoded}/initial.json`)
    : appPath(`api/v1/replays/${encoded}/initial`);
  return fetchJson<V2Initial>(url);
}

export async function comparisonIssue(leftId: string, rightId: string): Promise<string | null> {
  if (!STATIC_REPLAYS) {
    const response = await fetch(
      appPath(
        `api/v1/compare?left=${encodeURIComponent(leftId)}&right=${encodeURIComponent(rightId)}`,
      ),
    );
    if (response.ok) return null;
    const payload = (await response.json()) as {
      error?: { reasons?: string[]; message?: string };
      detail?: string;
    };
    return (
      payload.error?.reasons?.join(", ") ??
      payload.error?.message ??
      payload.detail ??
      "Runs are unavailable or incompatible."
    );
  }

  const [leftManifest, rightManifest, leftInitial, rightInitial] = await Promise.all([
    loadStaticManifest(leftId),
    loadStaticManifest(rightId),
    loadReplayInitial(leftId),
    loadReplayInitial(rightId),
  ]);
  const reasons: string[] = [];
  if (leftManifest.source.evaluation_seed !== rightManifest.source.evaluation_seed) {
    reasons.push("evaluation seed differs");
  }
  if (leftManifest.versions.scenario !== rightManifest.versions.scenario) {
    reasons.push("scenario differs");
  }
  if (JSON.stringify(leftManifest.environment_config) !== JSON.stringify(rightManifest.environment_config)) {
    reasons.push("environment configuration differs");
  }
  if (JSON.stringify(leftInitial.terrain) !== JSON.stringify(rightInitial.terrain)) {
    reasons.push("map terrain differs");
  }
  return reasons.length ? reasons.join(", ") : null;
}

async function loadStaticManifest(replayId: string): Promise<ReplayManifestV2> {
  const url = appPath(
    `static-api/replays/${encodeURIComponent(replayId)}/manifest.json`,
  );
  return validateManifestV2(await fetchJson<unknown>(url));
}

export function adaptV2ToRenderer(
  manifest: ReplayManifestV2,
  initial: V2Initial,
  records: TimelineRecord[],
): ReplayArtifact {
  const initialResources = initial.resources.map(normalizeResource);
  const initialAgents = initial.agents.map((agent) =>
    normalizeAgent(agent, "noop", "reset"),
  );
  const frames = records.map((record) => {
    const actionByAgent = new Map(
      record.actions.map((action) => [action.agent_id, action.selected_action]),
    );
    const eventByAgent = new Map<string, string>();
    record.events.forEach((event) => {
      event.actors.forEach((agent) => {
        const message = String(event.payload.message ?? event.type);
        eventByAgent.set(agent, message);
      });
    });
    return {
      step: record.tick,
      storm: record.weather.storm_active,
      camp: record.state_delta.camp,
      agents: record.state_delta.agents.map((agent) =>
        normalizeAgent(
          agent,
          actionByAgent.get(agent.id) ?? "noop",
          eventByAgent.get(agent.id) ?? actionByAgent.get(agent.id) ?? "noop",
        ),
      ),
      resource_changes: record.state_delta.resource_changes.map(normalizeResource),
      new_achievements: record.achievements,
      structures: record.state_delta.structures,
      creatures: record.state_delta.creatures,
      ground_piles: record.state_delta.ground_piles,
      time: record.state_delta.time,
      extensions: record.state_delta.extensions,
    };
  });
  return {
    schema_version: "stage6_replay_2.0.0",
    replay_id: manifest.replay_id,
    tick_rate: manifest.tick_rate,
    duration_seconds: manifest.world_steps / manifest.tick_rate,
    source: {
      benchmark_id: manifest.source.benchmark_id ?? "stage5_6_final_v1",
      scenario_id: manifest.versions.scenario,
      policy_id: manifest.source.policy_id,
      policy_kind: manifest.source.policy_kind,
      inference_mode: manifest.source.inference_mode ?? "default",
      training_seed: manifest.source.training_seed ?? 0,
      evaluation_seed: manifest.source.evaluation_seed,
      checkpoint: "",
      checkpoint_sha256: manifest.source.checkpoint_sha256 ?? "",
      manifest_sha256: manifest.manifest_sha256,
    },
    world: {
      width: initial.width,
      height: initial.height,
      initial: {
        terrain: initial.terrain.map((row) =>
          row.map((terrain) =>
            knownTerrain.has(terrain) ? (terrain as TerrainType) : "quarry",
          ),
        ),
        resources: initialResources,
        camp: initial.camp,
        agents: initialAgents,
        structures: initial.structures,
        creatures: initial.creatures,
        ground_piles: initial.ground_piles,
        time: initial.time,
      },
    },
    frames,
    summary: {
      world_steps: manifest.world_steps,
      agent_steps: manifest.agent_steps,
      dense_return: manifest.terminal_summary.dense_return ?? 0,
      survivors: manifest.terminal_summary.survivors,
      deaths: manifest.terminal_summary.deaths,
      shelter_completion_step:
        manifest.terminal_summary.shelter_completion_step ?? manifest.world_steps,
      camp_stockpile: manifest.terminal_summary.camp_stockpile,
      achievements: manifest.terminal_summary.achievements,
      achievement_steps: manifest.terminal_summary.achievement_steps,
      rescue_success: manifest.terminal_summary.rescue_success ?? false,
    },
  };
}

function normalizeResource(resource: ReplayResource): ReplayResource {
  return {
    ...resource,
    type: knownResources.has(resource.type) ? resource.type : ("stone" as ResourceType),
  };
}

function normalizeAgent(
  agent: Omit<ReplayAgent, "action" | "event">,
  action: string,
  event: string,
): ReplayAgent {
  return {
    ...agent,
    role: knownRoles.has(agent.role) ? agent.role : ("builder" as AgentRole),
    action,
    event,
  };
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return (await response.json()) as T;
}

async function fetchTimelineChunks(
  base: string,
  worldSteps: number,
  signal?: AbortSignal,
): Promise<{ records: TimelineRecord[] }> {
  const records: TimelineRecord[] = [];
  for (let start = 1; start <= worldSteps; start += 100) {
    const end = Math.min(worldSteps, start + 99);
    const chunk = await fetchJson<{ records: TimelineRecord[] }>(
      `${base}/timeline?start=${start}&end=${end}`,
      signal,
    );
    records.push(...chunk.records);
  }
  return { records };
}
