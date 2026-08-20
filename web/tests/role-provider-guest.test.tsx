import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// When Firebase isn't configured (no NEXT_PUBLIC_FIREBASE_* env vars —
// the actual state of the CI/build environment), @/lib/firebase exports
// `auth: null`. RoleProvider must settle on "guest" in that case rather
// than trying to reach a real Firebase project.
vi.mock("@/lib/firebase", () => ({
  auth: null,
  firebaseIsConfigured: false,
}));

import { RoleProvider, useRole } from "@/components/shell/RoleProvider";

function RoleProbe() {
  const { role } = useRole();
  return <div data-testid="role">{role}</div>;
}

describe("RoleProvider without configured Firebase auth", () => {
  it("settles on guest even when defaultRole is admin", async () => {
    render(
      <RoleProvider defaultRole="admin">
        <RoleProbe />
      </RoleProvider>
    );

    await waitFor(() => expect(screen.getByTestId("role")).toHaveTextContent("guest"));
  });
});
