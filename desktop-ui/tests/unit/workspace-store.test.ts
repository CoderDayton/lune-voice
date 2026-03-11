import { beforeEach, describe, expect, test } from "bun:test";
import { createInitialWorkspaceState, workspaceActions, workspaceStore } from "../../src/store/workspace-store";

describe("workspace store", () => {
  beforeEach(() => {
    workspaceStore.setState(() => createInitialWorkspaceState());
  });

  test("keeps dictionary entry ids unique for duplicate phrases", () => {
    const addedFirst = workspaceActions.addDictionaryEntry("github", "GitHub");
    const addedSecond = workspaceActions.addDictionaryEntry("github", "GitHub Enterprise");

    expect(addedFirst).toBe(true);
    expect(addedSecond).toBe(true);

    const githubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");

    expect(githubEntries).toHaveLength(3);
    expect(new Set(githubEntries.map((entry) => entry.id)).size).toBe(3);

    workspaceActions.removeDictionaryEntry(githubEntries[1]!.id);

    const remainingGithubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");
    expect(remainingGithubEntries).toHaveLength(2);
  });

  test("rejects incomplete dictionary entries without mutating state", () => {
    const before = workspaceStore.state.dictionaryEntries.length;

    expect(workspaceActions.addDictionaryEntry("github", "")).toBe(false);
    expect(workspaceActions.addDictionaryEntry("", "GitHub")).toBe(false);
    expect(workspaceStore.state.dictionaryEntries).toHaveLength(before);
  });
});
