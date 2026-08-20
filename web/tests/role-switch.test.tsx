import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// PrimaryNav uses next/navigation's usePathname, which requires an app-router
// context that isn't present under vitest/jsdom. Mock it to a fixed pathname.
vi.mock("next/navigation", () => ({
  usePathname: () => "/overview",
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
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
  it("shows admin-only links (Models, Analysis) when defaulted to admin", () => {
    renderShellNav();

    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Analysis" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument();
  });

  it("hides Models and Analysis after switching role to guest via the select", () => {
    renderShellNav();

    const select = screen.getByLabelText("View as role");
    fireEvent.change(select, { target: { value: "guest" } });

    expect(screen.queryByRole("link", { name: "Models" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Analysis" })).not.toBeInTheDocument();
    // Guest-visible links remain
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Stories" })).toBeInTheDocument();
  });
});
