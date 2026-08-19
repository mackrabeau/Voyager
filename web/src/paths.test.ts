// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { currentRoutePath } from "./paths";

describe("hosted viewer routes", () => {
  it("normalizes GitHub Pages directory-index trailing slashes", () => {
    window.history.replaceState({}, "", "/replays/island_oracle_rescue_v1/");

    expect(currentRoutePath()).toBe("/replays/island_oracle_rescue_v1");
  });

  it("preserves the root route", () => {
    window.history.replaceState({}, "", "/");

    expect(currentRoutePath()).toBe("/");
  });
});
