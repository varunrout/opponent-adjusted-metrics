import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

type MetricTileProps = {
  label: string;
  value?: string;
  delta?: string;
  deltaTone?: "pos" | "neg" | "muted";
  loading?: boolean;
};

export function MetricTile({ label, value, delta, deltaTone = "muted", loading = false }: MetricTileProps) {
  return (
    <Card className="tile">
      <p className="text-[11px] text-muted m-0">{label}</p>
      {loading ? (
        <Skeleton className="mt-1" style={{ height: 22, width: "60%" }} />
      ) : (
        <>
          <p className="font-data text-[22px] font-semibold m-0 mt-1">{value}</p>
          {delta && (
            <p
              className="font-data text-[11.5px] m-0 mt-0.5"
              style={{
                color:
                  deltaTone === "pos" ? "var(--green)" : deltaTone === "neg" ? "var(--red)" : "var(--muted)",
              }}
            >
              {delta}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
