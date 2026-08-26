import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

type Row = { id: number; name: string; shots: number };

const ROWS: Row[] = [
  { id: 1, name: "Charlie", shots: 5 },
  { id: 2, name: "Alice", shots: 20 },
  { id: 3, name: "Bob", shots: 10 },
];

const COLUMNS: DataTableColumn<Row>[] = [
  { key: "name", label: "Name", sortable: true },
  { key: "shots", label: "Shots", align: "right", sortable: true },
];

describe("DataTable", () => {
  it("renders loading skeletons, then rows", () => {
    const { rerender } = render(<DataTable columns={COLUMNS} rows={[]} rowKey={(r) => r.id} loading />);
    expect(screen.queryByText("Alice")).not.toBeInTheDocument();
    rerender(<DataTable columns={COLUMNS} rows={ROWS} rowKey={(r) => r.id} />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
  });

  it("shows the empty message when there are no rows", () => {
    render(<DataTable columns={COLUMNS} rows={[]} rowKey={(r) => r.id} emptyMessage="Nothing here." />);
    expect(screen.getByText("Nothing here.")).toBeInTheDocument();
  });

  it("shows an error state with a working retry button", () => {
    const onRetry = vi.fn();
    render(<DataTable columns={COLUMNS} rows={[]} rowKey={(r) => r.id} error onRetry={onRetry} />);
    fireEvent.click(screen.getByText("Retry"));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("sorts descending on first header click, ascending on second", () => {
    render(<DataTable columns={COLUMNS} rows={ROWS} rowKey={(r) => r.id} />);
    const header = screen.getByText("Shots");

    fireEvent.click(header);
    let cells = screen.getAllByText(/^(5|10|20)$/);
    expect(cells.map((c) => c.textContent)).toEqual(["20", "10", "5"]);

    fireEvent.click(header);
    cells = screen.getAllByText(/^(5|10|20)$/);
    expect(cells.map((c) => c.textContent)).toEqual(["5", "10", "20"]);
  });

  it("paginates rows client-side according to pageSize", () => {
    const manyRows: Row[] = Array.from({ length: 5 }, (_, i) => ({ id: i, name: `Row ${i}`, shots: i }));
    render(<DataTable columns={COLUMNS} rows={manyRows} rowKey={(r) => r.id} pageSize={2} />);

    expect(screen.getByText("Row 0")).toBeInTheDocument();
    expect(screen.queryByText("Row 2")).not.toBeInTheDocument();
    expect(screen.getByText(/Page 1 of 3/)).toBeInTheDocument();

    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("Row 2")).toBeInTheDocument();
    expect(screen.queryByText("Row 0")).not.toBeInTheDocument();
  });
});
