// Small static data table — same bordered-row language as DataTable,
// without sorting/pagination. For a handful of fixed rows (story figures,
// model-results/coefficients tables), not a browsable dataset.
export function SimpleTable({ columns, rows }: { columns: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11.5px] border-collapse" data-testid="simple-table">
        <thead>
          <tr>
            {columns.map((col, i) => (
              <th
                key={col}
                className={`text-muted font-normal pb-2 border-b border-border ${
                  i === 0 ? "text-left" : "text-right"
                }`}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className={`py-2 border-b border-border last:border-b-0 ${
                    ci === 0 ? "text-left text-text2" : "text-right text-text font-data"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
