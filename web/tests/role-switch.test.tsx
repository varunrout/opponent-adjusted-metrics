import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// PrimaryNav uses next/navigation's usePathname, which requires an app-router
// context that isn't present under vitest/jsdom. Mock it to a fixed pathname.
vi.mock("next/navigation", () => ({
  usePathname: () => "/overview",
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));

// RoleProvider now derives its role from Firebase auth state. Mock
// firebase/auth's onAuthStateChanged to never fire (simulating an
// indefinitely-pending auth check) so the provider's `defaultRole` prop
// stays authoritative for these nav-gating tests, which only exercise the
// manual "view as" override — not the real auth-derived flow. Pair with a
// non-null `auth` stub from @/lib/firebase so RoleProvider takes the
// "subscribe and wait" branch rather than its "auth is null -> guest"
// immediate fallback.
vi.mock("firebase/auth", () => ({
  onAuthStateChanged: () => () => {},
}));
vi.mock("@/lib/firebase", () => ({
  auth: {},
  firebaseIsConfigured: false,
}));

import { RoleProvider } from "@/components/shell/RoleProvider";
import { PrimaryNav } from "@/components/shell/PrimaryNav";
import { RoleSwitch } from "@/components/shell/RoleSwitch";

function renderShellNav() {
  return render(
    <RoleProvider defaultRole="admin">
      <PrimaryNav />
      <RoleSwitch />
    </RoleProvider>
  );
}

describe("role-based nav gating", () => {
  it("shows admin-only links (Analysis) and public Models when defaulted to admin", () => {
    renderShellNav();

    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Analysis" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument();
  });

  it("hides Analysis but keeps Models visible after switching role to guest via the select", () => {
    renderShellNav();

    const select = screen.getByLabelText("View as role");
    fireEvent.change(select, { target: { value: "guest" } });

    // Models is guest-visible by design (docs/dashboard_content_ideation.md:
    // it's a public registry layer, not an admin tool) — only Analysis is
    // actually admin-gated.
    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Analysis" })).not.toBeInTheDocument();
    // Guest-visible links remain
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Stories" })).toBeInTheDocument();
  });
});
