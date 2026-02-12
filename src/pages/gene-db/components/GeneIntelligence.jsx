import { useMemo, useState, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  BadgeCheck
} from "lucide-react";

const PAGE_SIZE = 10;

/* ================================================================== */
/* MAIN                                                               */
/* ================================================================== */
export default function GeneIntelligence() {
  // State for dynamic data
  const [geneTable, setGeneTable] = useState([]);
  const [selectedGene, setSelectedGene] = useState(null);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);

  // --- FETCH DATA ON MOUNT ---
  useEffect(() => {
    fetch("http://127.0.0.1:8000/gene-intelligence")
      .then((res) => res.json())
      .then((data) => {
        setGeneTable(data);
        // Set default selected gene to the top ranked one
        if (data.length > 0) {
          setSelectedGene(data[0].gene);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load gene intelligence:", err);
        setLoading(false);
      });
  }, []);

  // Derived state
  const selectedRow = geneTable.find(g => g.gene === selectedGene);

  const pagedGenes = useMemo(() => {
    const start = page * PAGE_SIZE;
    return geneTable.slice(start, start + PAGE_SIZE);
  }, [page, geneTable]); // Added geneTable dependency

  // Loading state (preserves layout structure while waiting)
  if (loading) {
    return (
      <section className="min-h-screen px-6 py-20 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100 flex items-center justify-center">
        <div className="text-slate-400 animate-pulse">Loading Biological Intelligence...</div>
      </section>
    );
  }

  return (
    <section className="min-h-screen px-6 py-20 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">

      {/* ================= HEADER ================= */}
      <div className="text-center max-w-4xl mx-auto mb-14">
        <div className="inline-flex items-center gap-2 text-pink-400 text-sm mb-3">
          <BadgeCheck size={14} />
          Peer-Reviewed & Validated
        </div>

        <h1 className="text-3xl font-semibold text-slate-300">
          Gene-Level Biological Intelligence
        </h1>
      </div>

      {/* ================= SUMMARY ================= */}
      <div className="max-w-5xl mx-auto text-sm text-slate-300 leading-relaxed mb-16">
        <h3 className="text-lg font-semibold text-slate-100 mb-3 text-center">
          Gene Contribution Summary
        </h3>

        <p className="mb-4 text-center max-w-3xl mx-auto">
          This gene contributes to the survival model through its learned Cox
          proportional hazards coefficient. The <strong>sign</strong> of the
          coefficient indicates the direction of association with risk, while the
          <strong> absolute magnitude</strong> reflects the strength of its
          contribution relative to other features in the model.
        </p>
      </div>

      {/* ================= MAIN GRID ================= */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 xl:grid-cols-9 gap-8">

        {/* ============ LEFT : TABLE ============ */}
        <div className="xl:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h3 className="text-pink-400 font-medium mb-4 text-center">
            Top Genes by Cox Model Contribution
          </h3>

          <div className="grid grid-cols-4 pb-3 mb-2 text-sm font-semibold text-slate-300 border-b border-slate-800 text-center">
            <div className="text-indigo-400">Rank</div>
            <div>Gene</div>
            <div className="text-indigo-400">Coefficient (loghazard)</div>
            <div>Absolute Magnitude</div>
          </div>

          <div className="divide-y divide-slate-800 text-sm">
            {pagedGenes.map(row => (
              <div
                key={row.gene}
                onClick={() => setSelectedGene(row.gene)}
                className={`grid grid-cols-4 py-2 cursor-pointer text-center transition-colors
                  ${
                    row.gene === selectedGene
                      ? "bg-slate-800 text-white"
                      : "text-slate-300 hover:bg-slate-800/60"
                  }`}
              >
                <div className="text-indigo-400">#{row.rank}</div>
                <div>{row.gene}</div>
                <div className={row.coef < 0 ? "text-emerald-400" : "text-rose-400"}>
                  {row.coef}
                </div>
                <div>{row.abs}</div>
              </div>
            ))}
          </div>

          <Pagination page={page} setPage={setPage} total={geneTable.length} />
        </div>

        {/* ============ RIGHT : MODEL CONTRIBUTION ============ */}
        <div className="xl:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 h-fit sticky top-10">
          <h3 className="font-medium mb-4 text-pink-400">
            Model Contribution (CoxPH)
          </h3>

          <label className="block text-xs text-slate-400 mb-2">
            Select Gene
          </label>

          <select
            value={selectedGene || ""}
            onChange={(e) => setSelectedGene(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-md px-3 py-2 text-sm text-white mb-5 outline-none focus:border-pink-500/50"
          >
            {geneTable.map(g => (
              <option key={g.gene} value={g.gene}>
                {g.gene}
              </option>
            ))}
          </select>

          {/* ===== BIOLOGICAL NARRATIVE ===== */}
          <div className="text-xs text-slate-300 mb-4 leading-relaxed">
            <span className="font-semibold text-slate-400 block mb-1">
              Biological Narrative:
            </span>
            {selectedRow?.narrative || (selectedRow?.coef < 0
              ? "This gene demonstrates a protective association with modeled hazard in the Cox proportional hazards model."
              : "This gene demonstrates an increased hazard association in the Cox proportional hazards model.")
            }
          </div>

          {/* ===== COXPH METRICS (TEXT ONLY) ===== */}
          <div className="border-t border-slate-800 pt-3 space-y-2 text-xs">
            <div className="font-semibold text-slate-400">(CoxPH)</div>

            <div className="flex justify-between">
              <span className="text-slate-400">Coefficient (log-hazard)</span>
              <span className={`font-medium ${selectedRow?.coef < 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {selectedRow?.coef}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">|Coefficient|</span>
              <span className="text-blue-400 font-medium">
                {selectedRow?.abs}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-slate-400">Rank by |Coef|</span>
              <span className="text-blue-400 font-medium">
                {selectedRow?.rank}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ================================================================== */
/* SMALL COMPONENTS                                                    */
/* ================================================================== */
function Pagination({ page, setPage, total }) {
  const start = page * PAGE_SIZE + 1;
  const end = Math.min((page + 1) * PAGE_SIZE, total);
  
  return (
    <div className="flex justify-between items-center mt-4 text-xs text-slate-500">
      <span>
        Showing {total > 0 ? start : 0}–{end} of {total}
      </span>
      <div className="flex gap-2">
        <button 
          onClick={() => setPage(p => Math.max(0, p - 1))}
          disabled={page === 0}
          className="hover:text-white disabled:opacity-30 disabled:hover:text-slate-500"
        >
          <ChevronLeft size={16} />
        </button>
        <button 
          onClick={() => setPage(p => (end < total ? p + 1 : p))}
          disabled={end >= total}
          className="hover:text-white disabled:opacity-30 disabled:hover:text-slate-500"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}