import { beforeEach, describe, expect, test } from "bun:test";
import { resetMockBackendClient } from "../../src/lib/backend/mock";
import {
  addDictionaryEntry,
  createInitialWorkspaceState,
  loadWorkspaceSnapshot,
  removeDictionaryEntry,
  workspaceStore,
} from "../../src/store/workspace-store";

describe("workspace store", () => {
  beforeEach(async () => {
    resetMockBackendClient();
    workspaceStore.setState(() => createInitialWorkspaceState());
    await loadWorkspaceSnapshot();
  });

  test("keeps dictionary entry ids unique for duplicate phrases", async () => {
    const addedFirst = await addDictionaryEntry("github", "GitHub");
    const addedSecond = await addDictionaryEntry("github", "GitHub Enterprise");

    expect(addedFirst).toBe(true);
    expect(addedSecond).toBe(true);

    const githubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");

    expect(githubEntries).toHaveLength(3);
    expect(new Set(githubEntries.map((entry) => entry.id)).size).toBe(3);

    await removeDictionaryEntry(githubEntries[1]!.id);

    const remainingGithubEntries = workspaceStore.state.dictionaryEntries.filter((entry) => entry.phrase === "github");
    expect(remainingGithubEntries).toHaveLength(2);
  });

  test("rejects incomplete dictionary entries without mutating state", async () => {
    const before = workspaceStore.state.dictionaryEntries.length;

    expect(await addDictionaryEntry("github", "")).toBe(false);
    expect(await addDictionaryEntry("", "GitHub")).toBe(false);
    expect(workspaceStore.state.dictionaryEntries).toHaveLength(before);
  });
});
