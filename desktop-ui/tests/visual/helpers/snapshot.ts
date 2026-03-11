import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect } from "bun:test";
import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";
import type { Locator } from "playwright";

const SNAPSHOT_ROOT = fileURLToPath(new URL("../__snapshots__/", import.meta.url));
const DIFF_ROOT = fileURLToPath(new URL("../__diffs__/", import.meta.url));
const UPDATE_SNAPSHOTS = process.env.UPDATE_VISUAL_SNAPSHOTS === "1";

export async function expectLocatorToMatchSnapshot(locator: Locator, snapshotName: string) {
  const screenshot = await locator.screenshot({ animations: "disabled" });
  assertSnapshot(snapshotName, screenshot);
}

function assertSnapshot(snapshotName: string, actualBuffer: Buffer<ArrayBufferLike>) {
  mkdirSync(SNAPSHOT_ROOT, { recursive: true });
  mkdirSync(DIFF_ROOT, { recursive: true });

  const snapshotPath = join(SNAPSHOT_ROOT, `${snapshotName}.png`);
  const diffPath = join(DIFF_ROOT, `${snapshotName}.png`);

  if (UPDATE_SNAPSHOTS) {
    writeFileSync(snapshotPath, actualBuffer);
    rmSync(diffPath, { force: true });
    return;
  }

  let expectedBuffer: Buffer;

  try {
    expectedBuffer = readFileSync(snapshotPath);
  } catch {
    throw new Error(`Missing visual snapshot ${snapshotName}. Run \`bun run test:visual:update\` first.`);
  }

  const expectedImage = PNG.sync.read(expectedBuffer);
  const actualImage = PNG.sync.read(actualBuffer);

  if (expectedImage.width !== actualImage.width || expectedImage.height !== actualImage.height) {
    writeFileSync(diffPath, actualBuffer);
    throw new Error(
      `Snapshot ${snapshotName} changed size from ${expectedImage.width}x${expectedImage.height} to ${actualImage.width}x${actualImage.height}.`,
    );
  }

  const diffImage = new PNG({ width: actualImage.width, height: actualImage.height });
  const mismatchedPixels = pixelmatch(
    expectedImage.data,
    actualImage.data,
    diffImage.data,
    actualImage.width,
    actualImage.height,
    { threshold: 0.1 },
  );

  if (mismatchedPixels > 0) {
    writeFileSync(diffPath, PNG.sync.write(diffImage));
  } else {
    rmSync(diffPath, { force: true });
  }

  expect(
    mismatchedPixels,
    `${snapshotName} does not match its baseline. Inspect ${diffPath} or update snapshots if the change is intentional.`,
  ).toBe(0);
}
