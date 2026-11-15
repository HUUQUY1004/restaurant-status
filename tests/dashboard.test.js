import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("Restaurant Status Dashboard", () => {
  it("loads the dashboard and navigates tabs", async (context) => {
    const testdriver = TestDriver(context);

    const targetUrl =
      process.env.APP_URL ||
      "https://missing-chains-floor-capabilities.trycloudflare.com";

    await testdriver.provision.chrome({ url: targetUrl });

    const assertTitle = await testdriver.assert(
      "the dashboard title 'Restaurant table status from camera footage' is visible"
    );
    expect(assertTitle).toBeTruthy();

    await testdriver.find("Timeline & metrics tab").click();

    const assertTimeline = await testdriver.assert(
      "the timeline chart displaying table status is visible"
    );
    expect(assertTimeline).toBeTruthy();

    await testdriver.find("Accuracy check tab").click();

    const assertAccuracy = await testdriver.assert(
      "the metric 'Model agrees with human labels' is visible"
    );
    expect(assertAccuracy).toBeTruthy();
  });
});
