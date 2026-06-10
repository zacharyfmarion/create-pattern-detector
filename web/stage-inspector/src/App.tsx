import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import clsx from "clsx";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Layers,
  RefreshCw,
  Search,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchStage4Diagnostic,
  fetchStage4Examples,
  fetchStage5Diagnostic,
  fetchStage5Examples,
  fetchStages,
  recomputeStage4Diagnostic,
} from "./api";
import { useInspectorStore } from "./store";
import type {
  EntitySelection,
  GraphEdge,
  GraphVertex,
  LayerKey,
  Stage4Diagnostic,
  Stage4Evidence,
  Stage4ExampleRow,
} from "./types";

const STAGE_COPY: Record<string, string> = {
  stage1: "Synthetic dataset composition, generation failures, and fold distribution checks.",
  stage2: "Deterministic vectorizer behavior on clean evidence maps and curated/stress fixtures.",
  stage3: "CPLineNet dense line, junction, angle, and assignment evidence from the Phase 3 checkpoint.",
  stage4: "Square topology decoding, assignment attribution, repair status, and FOLD-readiness diagnostics.",
  stage5: "Production cp-detect runs on real scraped CP images, including rectification, reports, and FOLD export.",
};

const LAYER_LABELS: Record<LayerKey, string> = {
  gtGraph: "GT graph",
  predGraph: "Prediction",
  missingEdges: "Missing GT",
  extraEdges: "Extra pred",
  ambiguousEdges: "Ambiguous edges",
  weakEdges: "Weak edges",
  shortEdges: "Very short",
  crowdedVertices: "Crowded",
  evenDegree: "Even-degree",
  kawasaki: "Kawasaki",
  maekawa: "Maekawa",
  repairs: "Repairs",
  labels: "Labels",
};

type EvidenceOverlayMode =
  | "none"
  | "line"
  | "junction"
  | "contacts"
  | "artifacts"
  | "style"
  | "assignment";

type LegendItem = { label: string; color: string; gradient?: string };

const EVIDENCE_OPTIONS: {
  key: EvidenceOverlayMode;
  label: string;
  field?: keyof Stage4Evidence;
  description: string;
}[] = [
  { key: "none", label: "None", description: "Clean input image." },
  { key: "line", label: "Lines", field: "lineProb", description: "Dense crease-line probability." },
  {
    key: "junction",
    label: "Junctions",
    field: "junctionHeatmap",
    description: "Dense vertex and crossing evidence.",
  },
  {
    key: "contacts",
    label: "Contacts",
    field: "boundaryContact",
    description: "Predicted crease-to-square-boundary contacts.",
  },
  {
    key: "artifacts",
    label: "Artifacts",
    field: "nonCrease",
    description: "Predicted text, grid, watermark, or other non-crease evidence.",
  },
  {
    key: "style",
    label: "Style",
    field: "lineStyle",
    description: "Predicted solid, dashed, faint, and monochrome line style.",
  },
  {
    key: "assignment",
    label: "Assign",
    field: "assignmentLabels",
    description: "Observed dense assignment labels before graph decoding.",
  },
];

export function App() {
  const stagesQuery = useQuery({ queryKey: ["stages"], queryFn: fetchStages });
  const activeStage = useInspectorStore((state) => state.activeStage);
  const setActiveStage = useInspectorStore((state) => state.setActiveStage);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Create Pattern Detector</p>
          <h1>Stage Inspector</h1>
        </div>
        <div className="server-status">
          <Activity size={16} />
          <span>{stagesQuery.isError ? "API unavailable" : "Local inspector"}</span>
        </div>
      </header>

      <nav className="stage-tabs">
        {(stagesQuery.data?.stages ?? []).map((stage) => (
          <button
            className={clsx("stage-tab", activeStage === stage.id && "active")}
            key={stage.id}
            onClick={() => setActiveStage(stage.id)}
          >
            <span>{stage.label}</span>
            <small>{stage.status}</small>
          </button>
        ))}
      </nav>

      <main className="stage-body">
        {activeStage === "stage4" ? (
          <InspectorExplorer stage="stage4" />
        ) : activeStage === "stage5" ? (
          <InspectorExplorer stage="stage5" />
        ) : (
          <ScaffoldStage stageId={activeStage} />
        )}
      </main>
    </div>
  );
}

function ScaffoldStage({ stageId }: { stageId: string }) {
  return (
    <section className="empty-stage">
      <FlaskConical size={32} />
      <div>
        <h2>{stageId.replace("stage", "Stage ")}</h2>
        <p>{STAGE_COPY[stageId] ?? "This stage will be wired into the inspector later."}</p>
      </div>
    </section>
  );
}

function InspectorExplorer({ stage }: { stage: "stage4" | "stage5" }) {
  const queryClient = useQueryClient();
  const selectedKey = useInspectorStore((state) => state.selectedExampleKey);
  const setSelectedKey = useInspectorStore((state) => state.setSelectedExampleKey);
  const [threshold, setThreshold] = useState(0.65);
  const [inferAssignments, setInferAssignments] = useState(false);
  const [snapBorderVertices, setSnapBorderVertices] = useState(false);
  const [evidenceOverlay, setEvidenceOverlay] = useState<EvidenceOverlayMode>("none");
  const examplesQuery = useQuery({
    queryKey: [stage, "examples"],
    queryFn: stage === "stage4" ? fetchStage4Examples : fetchStage5Examples,
  });
  const diagnosticQuery = useQuery({
    queryKey: [stage, "diagnostic", selectedKey],
    queryFn: () =>
      stage === "stage4" ? fetchStage4Diagnostic(selectedKey!) : fetchStage5Diagnostic(selectedKey!),
    enabled: Boolean(selectedKey),
  });
  const recomputeMutation = useMutation({
    mutationFn: recomputeStage4Diagnostic,
    onSuccess: (diagnostic) => {
      queryClient.setQueryData([stage, "diagnostic", diagnostic.key], diagnostic);
    },
  });

  useEffect(() => {
    if (!selectedKey && examplesQuery.data?.rows.length) {
      const preferred =
        stage === "stage4"
          ? examplesQuery.data.rows.find(
              (row) =>
                row.family === "rabbit-ear-fold-program" &&
                row.warnings.includes("very_short_edges") &&
                row.warnings.includes("crowded_junctions"),
            )
          : examplesQuery.data.rows.find((row) => row.status === "ambiguous");
      setSelectedKey((preferred ?? examplesQuery.data.rows[0]).key);
    }
  }, [examplesQuery.data, selectedKey, setSelectedKey, stage]);

  const diagnostic = diagnosticQuery.data;
  const rows = examplesQuery.data?.rows ?? [];

  useEffect(() => {
    if (stage !== "stage4" || evidenceOverlay === "none") return;
    if (diagnostic && !hasEvidenceForMode(diagnostic, evidenceOverlay)) {
      setEvidenceOverlay("none");
    }
  }, [diagnostic, evidenceOverlay, stage]);

  return (
    <section className={clsx("stage4-layout", stage === "stage5" && "stage5-layout")}>
      <aside className="left-rail">
        <ExampleBrowser rows={rows} isLoading={examplesQuery.isLoading} stage={stage} />
      </aside>
      <section className={clsx("main-workspace", stage === "stage5" && "stage5-workspace")}>
        <StageSummary rows={rows} diagnostic={diagnostic} stage={stage} />
        {stage === "stage4" && (
          <div className="controls-band">
            <EvidenceControls
              diagnostic={diagnostic}
              mode={evidenceOverlay}
              onChange={setEvidenceOverlay}
            />
            <div className="recompute-controls">
              <label>
                threshold
                <input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={inferAssignments}
                  onChange={(event) => setInferAssignments(event.target.checked)}
                />
                infer M/V
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={snapBorderVertices}
                  onChange={(event) => setSnapBorderVertices(event.target.checked)}
                />
                snap border
              </label>
              <button
                className="primary-button"
                disabled={!selectedKey || recomputeMutation.isPending}
                onClick={() =>
                  selectedKey &&
                  recomputeMutation.mutate({
                    exampleKey: selectedKey,
                    threshold,
                    inferAssignments,
                    repair: {
                      border_canonicalization_snap_vertices: snapBorderVertices,
                    },
                  })
                }
              >
                <RefreshCw size={16} />
                recompute
              </button>
            </div>
          </div>
        )}
        {diagnosticQuery.isLoading ? (
          <div className="loading-panel">Loading {stage === "stage4" ? "Stage 4" : "Stage 5"} diagnostics...</div>
        ) : diagnosticQuery.isError ? (
          <div className="error-panel">{String(diagnosticQuery.error)}</div>
        ) : diagnostic ? (
          <GraphWorkspace diagnostic={diagnostic} evidenceOverlay={evidenceOverlay} stage={stage} />
        ) : (
          <div className="loading-panel">Select an example to inspect.</div>
        )}
      </section>
    </section>
  );
}

function EvidenceControls({
  diagnostic,
  mode,
  onChange,
}: {
  diagnostic?: Stage4Diagnostic;
  mode: EvidenceOverlayMode;
  onChange: (mode: EvidenceOverlayMode) => void;
}) {
  return (
    <div className="evidence-controls">
      <div className="evidence-label">
        <Layers size={15} />
        <span>Evidence</span>
      </div>
      <div className="segmented-control" role="group" aria-label="Evidence overlay">
        {EVIDENCE_OPTIONS.map((option) => {
          const available = option.key === "none" || Boolean(diagnostic && hasEvidenceForMode(diagnostic, option.key));
          return (
            <button
              className={clsx("segment-button", mode === option.key && "active")}
              disabled={!available}
              key={option.key}
              onClick={() => onChange(option.key)}
              title={available ? option.description : "This diagnostic does not include that evidence head."}
              type="button"
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function StageSummary({
  rows,
  diagnostic,
  stage,
}: {
  rows: Stage4ExampleRow[];
  diagnostic?: Stage4Diagnostic;
  stage: "stage4" | "stage5";
}) {
  const statusData = useMemo(() => {
    const counts = rows.reduce<Record<string, number>>((acc, row) => {
      acc[row.status] = (acc[row.status] ?? 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([status, count]) => ({ status, count }));
  }, [rows]);

  return (
    <section className="summary-strip">
      <MetricTile label="status" value={diagnostic?.status ?? "loading"} />
      {stage === "stage4" ? (
        <>
          <MetricTile label="edge P/R" value={formatPair(diagnostic, "edge_precision", "edge_recall")} />
          <MetricTile
            label="assignment"
            value={formatPercent(metricNumber(diagnostic, "assignment_accuracy"))}
          />
        </>
      ) : (
        <>
          <MetricTile
            label="pred V/E"
            value={`${metricNumber(diagnostic, "pred_vertices").toFixed(0)} / ${metricNumber(
              diagnostic,
              "pred_edges",
            ).toFixed(0)}`}
          />
          <MetricTile
            label="rectifier"
            value={String(diagnostic?.metrics.rectification_mode ?? "loading")}
          />
        </>
      )}
      <MetricTile
        label="structural"
        value={diagnostic?.structuralValidity?.valid ? "valid" : "warning"}
      />
      <div className="status-chart">
        <ResponsiveContainer width="100%" height={74}>
          <BarChart data={statusData}>
            <CartesianGrid vertical={false} strokeDasharray="2 4" />
            <XAxis dataKey="status" hide />
            <YAxis hide allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#2563eb" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ExampleBrowser({
  rows,
  isLoading,
  stage,
}: {
  rows: Stage4ExampleRow[];
  isLoading: boolean;
  stage: "stage4" | "stage5";
}) {
  const selectedKey = useInspectorStore((state) => state.selectedExampleKey);
  const setSelectedKey = useInspectorStore((state) => state.setSelectedExampleKey);
  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState("all");
  const [status, setStatus] = useState("all");
  const [warning, setWarning] = useState("all");

  const profiles = useMemo(() => sortedUnique(rows.map((row) => row.profile)), [rows]);
  const statuses = useMemo(() => sortedUnique(rows.map((row) => row.status)), [rows]);
  const warnings = useMemo(() => sortedUnique(rows.flatMap((row) => row.warnings)), [rows]);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return rows.filter((row) => {
      if (profile !== "all" && row.profile !== profile) return false;
      if (status !== "all" && row.status !== status) return false;
      if (warning !== "all" && !row.warnings.includes(warning)) return false;
      if (!needle) return true;
      return `${row.id} ${row.family} ${row.bucket} ${row.profile}`.toLowerCase().includes(needle);
    });
  }, [profile, query, rows, status, warning]);

  const columns = useMemo<ColumnDef<Stage4ExampleRow>[]>(
    () => [
      {
        header: "Example",
        accessorKey: "id",
        cell: ({ row }) => (
          <div className="example-cell">
            <strong>{row.original.profile}</strong>
            <span>{row.original.family}</span>
            <small>{row.original.id}</small>
          </div>
        ),
      },
      {
        header: "Status",
        accessorKey: "status",
        cell: ({ getValue }) => <StatusPill status={String(getValue())} />,
      },
      {
        header: stage === "stage4" ? "Edge R" : "Pred E",
        accessorKey: "edgeRecall",
        cell: ({ row, getValue }) =>
          stage === "stage4" ? formatPercent(Number(getValue())) : row.original.predEdges,
      },
      {
        header: stage === "stage4" ? "Assign" : "Unknown",
        accessorKey: "assignmentAccuracy",
        cell: ({ row, getValue }) =>
          stage === "stage4" ? formatPercent(Number(getValue())) : row.original.unknownEdges,
      },
    ],
    [stage],
  );
  const table = useReactTable({
    data: filtered,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <section className="browser-panel">
      <div className="panel-title">
        <Search size={16} />
        <h2>Examples</h2>
      </div>
      <input
        className="search-input"
        placeholder="Search id, family, profile..."
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="filter-grid">
        <SelectFilter label="profile" value={profile} options={profiles} onChange={setProfile} />
        <SelectFilter label="status" value={status} options={statuses} onChange={setStatus} />
        <SelectFilter label="warning" value={warning} options={warnings} onChange={setWarning} />
      </div>
      <PresetButtons rows={rows} stage={stage} />
      <div className="table-wrap">
        {isLoading ? (
          <div className="loading-panel">Loading examples...</div>
        ) : (
          <table className="example-table">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th key={header.id}>
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr
                  className={clsx(selectedKey === row.original.key && "selected")}
                  key={row.original.key}
                  onClick={() => setSelectedKey(row.original.key)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function PresetButtons({ rows, stage }: { rows: Stage4ExampleRow[]; stage: "stage4" | "stage5" }) {
  const setSelectedKey = useInspectorStore((state) => state.setSelectedExampleKey);
  const presets =
    stage === "stage4"
      ? [
          {
            label: "worst Rabbit Ear",
            select: () =>
              rows
                .filter((row) => row.family === "rabbit-ear-fold-program")
                .sort((a, b) => a.edgeRecall - b.edgeRecall)[0],
          },
          {
            label: "theorem-heavy",
            select: () =>
              rows.find(
                (row) =>
                  row.warnings.includes("even_degree_failures") &&
                  row.warnings.includes("kawasaki_residuals") &&
                  row.warnings.includes("maekawa_failures"),
              ),
          },
          {
            label: "low confidence",
            select: () => rows.find((row) => row.warnings.includes("low_confidence_assignments")),
          },
        ]
      : [
          {
            label: "ambiguous",
            select: () => rows.find((row) => row.status === "ambiguous"),
          },
          {
            label: "dense output",
            select: () => [...rows].sort((a, b) => b.predEdges - a.predEdges)[0],
          },
          {
            label: "low confidence",
            select: () => rows.find((row) => row.warnings.includes("low_confidence_assignments")),
          },
        ];
  return (
    <div className="preset-row">
      {presets.map((preset) => (
        <button
          key={preset.label}
          onClick={() => {
            const row = preset.select();
            if (row) setSelectedKey(row.key);
          }}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}

function SelectFilter({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">all</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function GraphWorkspace({
  diagnostic,
  evidenceOverlay,
  stage,
}: {
  diagnostic: Stage4Diagnostic;
  evidenceOverlay: EvidenceOverlayMode;
  stage: "stage4" | "stage5";
}) {
  if (stage === "stage5") {
    return (
      <section className="graph-grid stage5-graph-grid">
        <GraphCanvas diagnostic={diagnostic} evidenceOverlay="none" mode="input" title="Input CP image" />
        <GraphCanvas diagnostic={diagnostic} evidenceOverlay="none" mode="pred" title="Output prediction" />
      </section>
    );
  }

  return (
    <section className="graph-grid stage4-graph-grid">
      <GraphCanvas
        diagnostic={diagnostic}
        evidenceOverlay={evidenceOverlay}
        mode="input"
        title="Input CP image"
      />
      <GraphCanvas diagnostic={diagnostic} evidenceOverlay="none" mode="pred" title="Output prediction" />
    </section>
  );
}

function LayerToggles() {
  const layers = useInspectorStore((state) => state.layers);
  const toggleLayer = useInspectorStore((state) => state.toggleLayer);
  const keys = Object.keys(LAYER_LABELS) as LayerKey[];
  return (
    <section className="layer-panel">
      <div className="panel-title compact">
        <Layers size={16} />
        <h2>Layers</h2>
      </div>
      <div className="layer-grid">
        {keys.map((key) => (
          <label key={key} className="checkbox-row">
            <input type="checkbox" checked={layers[key]} onChange={() => toggleLayer(key)} />
            {LAYER_LABELS[key]}
          </label>
        ))}
      </div>
    </section>
  );
}

function GraphCanvas({
  diagnostic,
  evidenceOverlay,
  mode,
  title,
}: {
  diagnostic: Stage4Diagnostic;
  evidenceOverlay: EvidenceOverlayMode;
  mode: "input" | "gt" | "pred" | "overlay";
  title: string;
}) {
  const layers = useInspectorStore((state) => state.layers);
  const selectedWarningCode = useInspectorStore((state) => state.selectedWarningCode);
  const selectedEntity = useInspectorStore((state) => state.selectedEntity);
  const setSelectedEntity = useInspectorStore((state) => state.setSelectedEntity);
  const imageSize = diagnostic.imageSize;
  const gt = diagnostic.graph.groundTruth;
  const pred = diagnostic.graph.prediction;
  const isOverlay = mode === "overlay";
  const showImage = mode === "input" || mode === "overlay";
  const showGtGraph = mode === "gt" || (isOverlay && layers.gtGraph);
  const showPredGraph = mode === "pred" || (isOverlay && layers.predGraph);
  const activeEvidence = mode === "input" && hasEvidenceForMode(diagnostic, evidenceOverlay)
    ? evidenceOverlay
    : "none";
  const evidenceImage = useMemo(
    () => evidenceOverlayImage(diagnostic, activeEvidence),
    [activeEvidence, diagnostic],
  );
  const evidenceLabel = labelForEvidenceMode(activeEvidence);

  return (
    <section className="viewer-panel">
      <div className="viewer-title">
        <h3>{title}</h3>
        <small>{evidenceLabel ? `${diagnostic.row.id as string} · ${evidenceLabel}` : diagnostic.row.id as string}</small>
      </div>
      <div className="graph-frame">
        <svg
          className="graph-svg"
          viewBox={`0 0 ${imageSize} ${imageSize}`}
          role="img"
          aria-label={title}
        >
          <rect width={imageSize} height={imageSize} fill={mode === "input" ? "#f8fafc" : "#ffffff"} />
          {showImage && (
            <image
              href={diagnostic.imageUrl}
              width={imageSize}
              height={imageSize}
              preserveAspectRatio="none"
              opacity={mode === "overlay" ? 0.24 : 1}
            />
          )}
          {evidenceImage && (
            <image
              href={evidenceImage}
              width={imageSize}
              height={imageSize}
              preserveAspectRatio="none"
              className="evidence-raster"
            />
          )}
          {showGtGraph && (
            <g>
              {gt.edges.map((edge) => (
                <GraphLine
                  edge={edge}
                  key={`gt-${edge.id}`}
                  vertices={gt.vertices}
                  kind="gt-edge"
                  selectedEntity={selectedEntity}
                  selectedWarningCode={selectedWarningCode}
                  onSelect={setSelectedEntity}
                  plain={!isOverlay}
                />
              ))}
            </g>
          )}
          {isOverlay && layers.missingEdges && (
            <g>
              {gt.edges
                .filter((edge) => edge.match.state === "missing")
                .map((edge) => (
                  <GraphLine
                    edge={edge}
                    key={`missing-${edge.id}`}
                    vertices={gt.vertices}
                    kind="gt-edge"
                    selectedEntity={selectedEntity}
                    selectedWarningCode={selectedWarningCode}
                    onSelect={setSelectedEntity}
                    variant="missing"
                  />
                ))}
            </g>
          )}
          {showPredGraph && (
            <g>
              {pred.edges.map((edge) => (
                <GraphLine
                  edge={edge}
                  key={`pred-${edge.id}`}
                  vertices={pred.vertices}
                  kind="pred-edge"
                  selectedEntity={selectedEntity}
                  selectedWarningCode={selectedWarningCode}
                  onSelect={setSelectedEntity}
                  plain={!isOverlay}
                />
              ))}
            </g>
          )}
          {showGtGraph &&
            gt.vertices.map((vertex) => (
              <VertexCircle
                key={`gt-v-${vertex.id}`}
                vertex={vertex}
                kind="gt-vertex"
                selectedEntity={selectedEntity}
                selectedWarningCode={selectedWarningCode}
                onSelect={setSelectedEntity}
                plain={!isOverlay}
              />
            ))}
          {showPredGraph &&
            pred.vertices.map((vertex) => (
              <VertexCircle
                key={`pred-v-${vertex.id}`}
                vertex={vertex}
                kind="pred-vertex"
                selectedEntity={selectedEntity}
                selectedWarningCode={selectedWarningCode}
                onSelect={setSelectedEntity}
                plain={!isOverlay}
              />
            ))}
          {isOverlay && layers.labels && <GraphLabels diagnostic={diagnostic} mode={mode} />}
        </svg>
      </div>
      {activeEvidence !== "none" && <EvidenceLegend mode={activeEvidence} />}
    </section>
  );
}

function EvidenceLegend({ mode }: { mode: EvidenceOverlayMode }) {
  const items = legendItems(mode);
  if (!items.length) return null;
  return (
    <div className="evidence-legend" aria-label={`${labelForEvidenceMode(mode)} legend`}>
      {items.map((item) => (
        <span className="legend-item" key={item.label}>
          <span
            className={clsx("legend-swatch", item.gradient && "legend-gradient")}
            style={{ background: item.gradient ?? item.color }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

function hasEvidenceForMode(diagnostic: Stage4Diagnostic | undefined, mode: EvidenceOverlayMode) {
  const field = fieldForEvidenceMode(mode);
  return field ? Boolean(diagnostic?.evidence?.[field]) : true;
}

function fieldForEvidenceMode(mode: EvidenceOverlayMode): keyof Stage4Evidence | null {
  return EVIDENCE_OPTIONS.find((option) => option.key === mode)?.field ?? null;
}

function labelForEvidenceMode(mode: EvidenceOverlayMode) {
  if (mode === "none") return "";
  return EVIDENCE_OPTIONS.find((option) => option.key === mode)?.label ?? "";
}

function evidenceOverlayImage(diagnostic: Stage4Diagnostic, mode: EvidenceOverlayMode) {
  const field = fieldForEvidenceMode(mode);
  if (!field) return null;
  const raster = diagnostic.evidence?.[field];
  if (!raster) return null;

  const canvas = document.createElement("canvas");
  canvas.width = raster.width;
  canvas.height = raster.height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const imageData = context.createImageData(raster.width, raster.height);
  const lineGate = mode === "style" ? matchingLineGate(diagnostic, raster.width, raster.height) : null;

  for (let index = 0; index < raster.width * raster.height; index += 1) {
    const offset = index * 4;
    const pixel =
      raster.kind === "float"
        ? floatEvidencePixel(mode, raster.values[index] ?? 0)
        : classEvidencePixel(mode, raster.labels[raster.values[index]] ?? "none", raster.confidence?.[index], lineGate?.[index]);
    imageData.data[offset] = pixel[0];
    imageData.data[offset + 1] = pixel[1];
    imageData.data[offset + 2] = pixel[2];
    imageData.data[offset + 3] = pixel[3];
  }

  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function matchingLineGate(diagnostic: Stage4Diagnostic, width: number, height: number) {
  const raster = diagnostic.evidence?.lineProb;
  if (!raster || raster.width !== width || raster.height !== height) return null;
  return raster.values;
}

function floatEvidencePixel(mode: EvidenceOverlayMode, rawValue: number): [number, number, number, number] {
  const value = clamp01(rawValue);
  if (value < 0.035) return [0, 0, 0, 0];
  const [red, green, blue] =
    mode === "junction"
      ? [245, 158, 11]
      : mode === "contacts"
        ? [34, 197, 94]
        : mode === "artifacts"
          ? [217, 70, 239]
          : [14, 165, 233];
  return [red, green, blue, Math.round(255 * Math.min(0.78, value * 0.72))];
}

function classEvidencePixel(
  mode: EvidenceOverlayMode,
  label: string,
  confidence = 1,
  lineGate?: number,
): [number, number, number, number] {
  if (label === "none") return [0, 0, 0, 0];
  const gate = mode === "style" ? clamp01(lineGate ?? 0) : 1;
  if (mode === "style" && gate < 0.08) return [0, 0, 0, 0];
  const [red, green, blue] = classColor(mode, label);
  const alphaBase = mode === "style" ? 0.74 * Math.min(1, gate * 1.35) : 0.82;
  const alpha = Math.round(255 * alphaBase * Math.max(0.35, clamp01(confidence)));
  return [red, green, blue, alpha];
}

function classColor(mode: EvidenceOverlayMode, label: string): [number, number, number] {
  if (mode === "assignment") {
    if (label === "M") return [225, 29, 72];
    if (label === "V") return [37, 99, 235];
    if (label === "B") return [17, 24, 39];
    return [107, 114, 128];
  }
  if (label === "dashed") return [8, 145, 178];
  if (label === "faint") return [245, 158, 11];
  if (label === "monochrome") return [124, 58, 237];
  return [34, 197, 94];
}

function legendItems(mode: EvidenceOverlayMode): LegendItem[] {
  if (mode === "style") {
    return [
      { label: "solid", color: "rgb(34, 197, 94)" },
      { label: "dashed", color: "rgb(8, 145, 178)" },
      { label: "faint", color: "rgb(245, 158, 11)" },
      { label: "mono", color: "rgb(124, 58, 237)" },
    ];
  }
  if (mode === "assignment") {
    return [
      { label: "M", color: "rgb(225, 29, 72)" },
      { label: "V", color: "rgb(37, 99, 235)" },
      { label: "B", color: "rgb(17, 24, 39)" },
      { label: "U", color: "rgb(107, 114, 128)" },
    ];
  }
  if (mode === "none") return [];
  return [
    {
      label: "low -> high",
      color: "transparent",
      gradient: `linear-gradient(90deg, rgba(${legendColor(mode).join(", ")}, 0.08), rgba(${legendColor(mode).join(", ")}, 0.78))`,
    },
  ];
}

function legendColor(mode: EvidenceOverlayMode): [number, number, number] {
  if (mode === "junction") return [245, 158, 11];
  if (mode === "contacts") return [34, 197, 94];
  if (mode === "artifacts") return [217, 70, 239];
  return [14, 165, 233];
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function GraphLine({
  edge,
  vertices,
  kind,
  selectedEntity,
  selectedWarningCode,
  onSelect,
  variant,
  plain = false,
}: {
  edge: GraphEdge;
  vertices: GraphVertex[];
  kind: "gt-edge" | "pred-edge";
  selectedEntity: EntitySelection;
  selectedWarningCode: string | null;
  onSelect: (selection: EntitySelection) => void;
  variant?: "missing";
  plain?: boolean;
}) {
  const layers = useInspectorStore((state) => state.layers);
  const [v0, v1] = edge.vertices.map((id) => vertices[id]);
  if (!v0 || !v1) return null;
  const style = plain ? plainEdgeStyle(edge, kind) : edgeStyle(edge, kind, variant, layers);
  const selected = !plain && selectedEntity?.kind === kind && selectedEntity.id === edge.id;
  const highlighted = !plain && selectedWarningCode ? edge.issues.includes(selectedWarningCode) : false;
  return (
    <line
      x1={v0.x}
      y1={v0.y}
      x2={v1.x}
      y2={v1.y}
      className={clsx("graph-edge", selected && "selected", highlighted && "highlighted")}
      stroke={style.stroke}
      strokeWidth={selected || highlighted ? style.width + 3 : style.width}
      strokeDasharray={style.dash}
      opacity={style.opacity}
      onClick={() => onSelect({ kind, id: edge.id })}
    >
      <title>{edgeTitle(edge)}</title>
    </line>
  );
}

function VertexCircle({
  vertex,
  kind,
  selectedEntity,
  selectedWarningCode,
  onSelect,
  plain = false,
}: {
  vertex: GraphVertex;
  kind: "gt-vertex" | "pred-vertex";
  selectedEntity: EntitySelection;
  selectedWarningCode: string | null;
  onSelect: (selection: EntitySelection) => void;
  plain?: boolean;
}) {
  const layers = useInspectorStore((state) => state.layers);
  const selected = !plain && selectedEntity?.kind === kind && selectedEntity.id === vertex.id;
  const highlighted = !plain && selectedWarningCode ? vertex.issues?.includes(selectedWarningCode) : false;
  const fill = kind === "gt-vertex" || plain ? "#f8fafc" : vertexColor(vertex, layers);
  return (
    <circle
      cx={vertex.x}
      cy={vertex.y}
      r={selected || highlighted ? 8 : 5}
      fill={fill}
      stroke={selected || highlighted ? "#111827" : "#475569"}
      strokeWidth={selected || highlighted ? 3 : 1.25}
      opacity={kind === "gt-vertex" ? 0.86 : 0.96}
      onClick={() => onSelect({ kind, id: vertex.id })}
    >
      <title>
        {kind} {vertex.id} degree={vertex.degree} issues={(vertex.issues ?? []).join(", ") || "none"}
      </title>
    </circle>
  );
}

function GraphLabels({
  diagnostic,
  mode,
}: {
  diagnostic: Stage4Diagnostic;
  mode: "gt" | "pred" | "overlay" | "input";
}) {
  const gt = diagnostic.graph.groundTruth;
  const pred = diagnostic.graph.prediction;
  const labels = [];
  if (mode === "gt" || mode === "overlay") {
    labels.push(
      ...gt.edges.map((edge) => ({
        key: `gt-label-${edge.id}`,
        text: `g${edge.id}`,
        point: midpoint(edge, gt.vertices),
      })),
    );
  }
  if (mode === "pred" || mode === "overlay") {
    labels.push(
      ...pred.edges.map((edge) => ({
        key: `pred-label-${edge.id}`,
        text: `p${edge.id}`,
        point: midpoint(edge, pred.vertices),
      })),
    );
  }
  return (
    <g className="graph-labels">
      {labels.map((label) => (
        <text key={label.key} x={label.point.x} y={label.point.y}>
          {label.text}
        </text>
      ))}
    </g>
  );
}

function StatusPill({ status }: { status: string }) {
  const icon =
    status === "valid" || status === "repaired" ? (
      <CheckCircle2 size={14} />
    ) : status === "failed" ? (
      <XCircle size={14} />
    ) : (
      <AlertTriangle size={14} />
    );
  return (
    <span className={clsx("status-pill", status)}>
      {icon}
      {status}
    </span>
  );
}

function edgeStyle(
  edge: GraphEdge,
  kind: "gt-edge" | "pred-edge",
  variant: "missing" | undefined,
  layers: Record<LayerKey, boolean>,
) {
  if (variant === "missing") {
    return { stroke: "#2563eb", width: 7, dash: "12 8", opacity: 0.82 };
  }
  if (kind === "pred-edge" && edge.match.state === "extra" && layers.extraEdges) {
    return { stroke: "#ef4444", width: 6, dash: "", opacity: 0.72 };
  }
  if (edge.issues.includes("weak_edges") && layers.weakEdges) {
    return { stroke: "#f97316", width: 4.5, dash: "", opacity: 0.9 };
  }
  if (edge.source === "unknown" && layers.ambiguousEdges) {
    return { stroke: "#64748b", width: 3, dash: "10 6", opacity: 0.88 };
  }
  if (edge.issues.includes("very_short_edges") && layers.shortEdges) {
    return { stroke: assignmentColor(edge.assignment), width: 4, dash: "3 5", opacity: 0.95 };
  }
  return {
    stroke: assignmentColor(edge.assignment),
    width: edge.assignment === "B" ? 3.6 : 2.6,
    dash: "",
    opacity: kind === "gt-edge" ? 0.5 : 0.92,
  };
}

function plainEdgeStyle(edge: GraphEdge, kind: "gt-edge" | "pred-edge") {
  return {
    stroke: assignmentColor(edge.assignment),
    width: edge.assignment === "B" ? 3.4 : 2.4,
    dash: "",
    opacity: kind === "gt-edge" ? 0.82 : 0.94,
  };
}

function vertexColor(vertex: GraphVertex, layers: Record<LayerKey, boolean>) {
  const issues = vertex.issues ?? [];
  if (issues.includes("illegal_crossings")) return "#ef4444";
  if (issues.includes("maekawa_failures") && layers.maekawa) return "#d946ef";
  if (issues.includes("kawasaki_residuals") && layers.kawasaki) return "#06b6d4";
  if (issues.includes("even_degree_failures") && layers.evenDegree) return "#f59e0b";
  if (issues.includes("crowded_junctions") && layers.crowdedVertices) return "#8b5cf6";
  if ((vertex.repairs ?? []).length && layers.repairs) return "#22c55e";
  return "#94a3b8";
}

function assignmentColor(assignment: string) {
  if (assignment === "M") return "#e11d48";
  if (assignment === "V") return "#2563eb";
  if (assignment === "B") return "#111827";
  return "#6b7280";
}

function edgeTitle(edge: GraphEdge) {
  return [
    `edge ${edge.id}`,
    `assignment=${edge.assignment}`,
    `source=${edge.source ?? "gt"}`,
    `support=${formatNumber(edge.support)}`,
    `confidence=${formatNumber(edge.confidence)}`,
    `match=${edge.match.state}`,
    `issues=${edge.issues.join(", ") || "none"}`,
  ].join("\n");
}

function midpoint(edge: GraphEdge, vertices: GraphVertex[]) {
  const [v0, v1] = edge.vertices.map((id) => vertices[id]);
  return { x: (v0.x + v1.x) / 2, y: (v0.y + v1.y) / 2 };
}

function metricNumber(diagnostic: Stage4Diagnostic | undefined, key: string) {
  const value = diagnostic?.metrics[key];
  return typeof value === "number" ? value : Number(value ?? 0);
}

function formatPair(diagnostic: Stage4Diagnostic | undefined, left: string, right: string) {
  return `${formatPercent(metricNumber(diagnostic, left))} / ${formatPercent(metricNumber(diagnostic, right))}`;
}

function formatPercent(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatNumber(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "n/a";
  return value.toFixed(3);
}

function sortedUnique(values: string[]) {
  return [...new Set(values)].sort();
}
