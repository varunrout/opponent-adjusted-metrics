import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";

// A signed-in user whose /v1/me lookup never resolves (hung/unresponsive
// backend) must still settle roleResolved to true, via the guest fallback,
// rather than leaving pages that gate on roleResolved (e.g. /analysis)
// stuck on a loading skeleton forever.
const fakeUser = {
  uid: "fake-uid",
  email: "fake@example.invalid",
  getIdToken: vi.fn().mockResolvedValue("fake-id-token"),
};

let authStateCallback: ((user: unknown) => void) | null = null;

vi.mock("firebase/auth", () => ({
  onAuthStateChanged: (_auth: unknown, callback: (user: unknown) => void) => {
    authStateCallback = callback;
    return () => {};
  },
}));

vi.mock("@/lib/firebase", () => ({
  auth: {},
  firebaseIsConfigured: true,
}));

vi.mock("@/lib/api", () => ({
  // Never resolves — simulates a hung backend.
  getMe: vi.fn(() => new Promise(() => {})),
}));

import { RoleProvider, useRole } from "@/components/shell/RoleProvider";

function RoleProbe() {
  const { role, roleResolved } = useRole();
  return (
    <div>
      <div data-testid="role">{role}</div>
      <div data-testid="resolved">{String(roleResolved)}</div>
    </div>
  );
}

describe("RoleProvider getMe timeout", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    authStateCallback = null;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("falls back to guest and resolves roleResolved after a hung getMe call times out", async () => {
    render(
      <RoleProvider defaultRole="admin">
        <RoleProbe />
      </RoleProvider>
    );

    // Fire the auth-state change with a signed-in user, then let the
    // getIdToken() microtask resolve so the getMe() call is actually made.
    expect(authStateCallback).not.toBeNull();
    await act(async () => {
      authStateCallback!(fakeUser);
      await vi.advanceTimersByTimeAsync(0);
    });

    // Still pending — getMe never resolves and the timeout hasn't fired yet.
    expect(screen.getByTestId("resolved")).toHaveTextContent("false");

    // Advance well past the timeout (9s) — the race should reject and the
    // catch/finally should settle role=guest, roleResolved=true.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });

    expect(screen.getByTestId("resolved")).toHaveTextContent("true");
    expect(screen.getByTestId("role")).toHaveTextContent("guest");
  });
});
