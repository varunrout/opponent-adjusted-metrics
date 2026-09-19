import { describe, expect, it, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";

// RoleGate must not redirect until the real role has resolved.
//
// Regression test for a live bug: an admin loading /analysis directly was
// bounced to /overview because RoleGate ran its check against RoleProvider's
// initial defaultRole ("guest") before /v1/me had returned. The admin API
// calls were never attempted at all, which made a client-side race look like
// a backend auth failure.

const replace = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/analysis",
  useRouter: () => ({ replace }),
}));

let mockRole = "guest";
let mockResolved = false;

vi.mock("@/components/shell/RoleProvider", () => ({
  useRole: () => ({
    role: mockRole,
    roleResolved: mockResolved,
    setRole: vi.fn(),
    user: null,
  }),
}));

import { RoleGate } from "@/components/shell/RoleGate";

describe("RoleGate", () => {
  beforeEach(() => {
    replace.mockClear();
  });

  it("does not redirect while the role is still unresolved", () => {
    mockRole = "guest";
    mockResolved = false;

    render(<RoleGate />);

    // This is the bug: without the roleResolved guard, the pre-resolution
    // "guest" default would trigger an immediate redirect off /analysis.
    expect(replace).not.toHaveBeenCalled();
  });

  it("does not redirect an admin from /analysis once the role resolves", () => {
    mockRole = "admin";
    mockResolved = true;

    render(<RoleGate />);

    expect(replace).not.toHaveBeenCalled();
  });

  it("redirects a resolved guest away from /analysis", () => {
    mockRole = "guest";
    mockResolved = true;

    render(<RoleGate />);

    expect(replace).toHaveBeenCalledWith("/overview");
  });

  it("redirects a resolved viewer away from /analysis", () => {
    mockRole = "viewer";
    mockResolved = true;

    render(<RoleGate />);

    expect(replace).toHaveBeenCalledWith("/overview");
  });
});
