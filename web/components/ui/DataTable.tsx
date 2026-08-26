"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ClickableRow } from "@/components/ui/ClickableRow";

export type DataTableColumn<T> = {
  key: string;
  label: string;
  align?: "left" | "right" | "center";
  sortable?: boolean;
  // Custom cell renderer. `index` is the row's 0-based position in the
  // current sorted+paginated view (e.g. for a "#" rank column). Defaults to
  // String(row[key]).
  render?: (row: T, index: number) => ReactNode;
  // Value used for sorting when the column is sortable. Defaults to row[key].
  sortValue?: (row: T) => number | string | null;
  // CSS grid track size for this column, e.g. "48px" or "1fr". Defaults to "1fr".
  width?: string;
};

type DataTableProps<T> = {
  columns: DataTableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string | number;
  rowHref?: (row: T) => string;
  loading?: boolean;
  error?: boolean;
  onRetry?: () => void;
  emptyMessage?: string;
  pageSize?: number;
};

const ALIGN_CLASS: Record<"left" | "right" | "center", string> = {
  left: "text-left justify-start",
  right: "text-right justify-end",
  center: "text-center justify-center",
};

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  rowHref,
  loading = false,
  error = false,
  onRetry,
  emptyMessage = "No results found.",
  pageSize = 50,
}: DataTableProps<T>) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(null);
  const [page, setPage] = useState(0);

  const gridTemplateColumns = columns.map((c) => c.width ?? "1fr").join(" ");

  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((c) => c.key === sort.key);
    if (!column) return rows;
    const getValue =
      column.sortValue ?? ((row: T) => (row as Record<string, unknown>)[column.key] as number | string | null);
    const withValues = rows.map((row) => ({ row, value: getValue(row) }));
    withValues.sort((a, b) => {
      const av = a.value;
      const bv = b.value;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (av < bv) return sort.dir === "asc" ? -1 : 1;
      if (av > bv) return sort.dir === "asc" ? 1 : -1;
      return 0;
    });
    return withValues.map((w) => w.row);
  }, [rows, sort, columns]);

  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const currentPage = Math.min(page, totalPages - 1);
  const pageRows = sortedRows.slice(currentPage * pageSize, currentPage * pageSize + pageSize);

  function toggleSort(column: DataTableColumn<T>) {
    if (!column.sortable) return;
    setPage(0);
    setSort((prev) => {
      if (!prev || prev.key !== column.key) return { key: column.key, dir: "desc" };
      return { key: column.key, dir: prev.dir === "desc" ? "asc" : "desc" };
    });
  }

  if (loading) {
    return (
      <div className="bg-card border border-border rounded overflow-hidden">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="px-4 py-[10px] border-b border-border last:border-b-0">
            <Skeleton style={{ height: 14 }} />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-l-[3px]" style={{ borderLeftColor: "var(--amber)" }}>
        <p className="text-[12.5px] text-muted m-0 mb-2">Couldn&apos;t load this data. Try again shortly.</p>
        {onRetry && (
          <button type="button" onClick={onRetry} className="text-[12px] text-teal hover:underline">
            Retry
          </button>
        )}
      </Card>
    );
  }

  if (rows.length === 0) {
    return (
      <Card>
        <p className="text-[12.5px] text-muted m-0">{emptyMessage}</p>
      </Card>
    );
  }

  return (
    <div>
      <div className="bg-card border border-border rounded overflow-hidden">
        <div
          className="grid gap-3 items-center px-4 py-[9px] border-b border-border text-[11.5px] text-muted"
          style={{ gridTemplateColumns }}
        >
          {columns.map((column) => (
            <button
              key={column.key}
              type="button"
              onClick={() => toggleSort(column)}
              className={`flex items-center gap-1 bg-transparent border-0 p-0 m-0 text-[11.5px] text-muted ${ALIGN_CLASS[column.align ?? "left"]} ${
                column.sortable ? "cursor-pointer hover:text-text2" : "cursor-default"
              }`}
              disabled={!column.sortable}
            >
              {column.label}
              {column.sortable && sort?.key === column.key && <span>{sort.dir === "asc" ? "↑" : "↓"}</span>}
            </button>
          ))}
        </div>

        {pageRows.map((row, i) => {
          const index = currentPage * pageSize + i;
          const rowContent = columns.map((column) => (
            <div key={column.key} className={`font-data text-text2 truncate ${ALIGN_CLASS[column.align ?? "left"]}`}>
              {column.render
                ? column.render(row, index)
                : String((row as Record<string, unknown>)[column.key] ?? "")}
            </div>
          ));

          const rowClassName =
            "grid gap-3 items-center px-4 py-[10px] border-b border-border last:border-b-0 text-[12.5px] hover:bg-card-hi";

          if (rowHref) {
            return (
              <ClickableRow
                key={rowKey(row)}
                href={rowHref(row)}
                className={`${rowClassName} cursor-pointer`}
                style={{ gridTemplateColumns }}
              >
                {rowContent}
              </ClickableRow>
            );
          }
          return (
            <div key={rowKey(row)} className={rowClassName} style={{ gridTemplateColumns }}>
              {rowContent}
            </div>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-2.5 text-[12px] text-muted">
          <span>
            Page {currentPage + 1} of {totalPages} · {sortedRows.length} rows
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className="px-2 py-1 rounded border border-border disabled:opacity-40 hover:text-text"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={currentPage >= totalPages - 1}
              className="px-2 py-1 rounded border border-border disabled:opacity-40 hover:text-text"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
