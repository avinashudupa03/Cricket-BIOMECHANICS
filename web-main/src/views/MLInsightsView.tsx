import React, { useEffect, useState } from 'react';
import {
  BrainCircuit,
  Trophy,
  Users,
  Sparkles,
  ArrowDown,
} from 'lucide-react';
import { MLInsights, SHOT_TYPES, SHOT_TYPE_LABELS } from '../types';
import { api } from '../utils/api';
import { Badge, Button, Card, EmptyState, ProgressBar, Spinner, Stat } from '../components/ui';

export function MLInsightsView() {
  const [ml, setMl] = useState<MLInsights | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    api
      .ml()
      .then((m) => mounted && setMl(m))
      .catch(() => mounted && setMl(null))
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner className="h-6 w-6 text-brand" />
      </div>
    );
  }

  if (!ml?.available) {
    return (
      <EmptyState
        icon={<BrainCircuit className="h-6 w-6" />}
        title="No ML evaluation available"
        message="Build the dataset and train models to see LOOCV evaluation here: python build_dataset.py && python train_models.py && python evaluate_models.py"
      />
    );
  }

  const rows = ml.comparison_table ?? [];
  const bestAcc = Math.max(0, ...rows.map((r) => r.accuracy ?? 0));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Stat label="Evaluation method" value="LOOCV" hint="Leave-one-video-out cross-validation" tone="violet" />
        <Stat label="Dataset size" value={ml.dataset_size ?? '—'} hint="per-video rows, never leaking across folds" tone="brand" />
        <Stat label="Shot classes" value={ml.n_classes ?? '—'} hint={SHOT_TYPES.map((s) => SHOT_TYPE_LABELS[s]).join(' · ')} tone="blue" />
        <Stat label="Best model" value={ml.best_model ?? '—'} hint={<span className="flex items-center gap-1"><Trophy className="h-3.5 w-3.5" /> {ml.accuracy ?? 'n/a'} accuracy</span>} tone="amber" />
      </div>

      <Card
        title="Model comparison"
        subtitle="Accuracy and macro-F1 for every classifier (tested with LOOCV, fixed seed 42)"
      >
        {rows.length === 0 ? (
          <EmptyState icon={<Sparkles className="h-5 w-5" />} title="Comparison table missing" message="Run evaluate_models.py to produce reports/model_comparison.csv" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] border-collapse text-left">
              <thead>
                <tr className="border-b border-line text-[11.5px] uppercase tracking-wide text-ink-3">
                  <th className="py-3 pr-4 font-semibold">Model</th>
                  <th className="py-3 pr-4 w-40 font-semibold">Accuracy</th>
                  <th className="py-3 pr-4 font-semibold">F1 (macro)</th>
                  <th className="py-3 pr-4 font-semibold">Precision</th>
                  <th className="py-3 font-semibold">Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {rows.map((r) => {
                  const acc = r.accuracy ?? 0;
                  const best = acc >= bestAcc && bestAcc > 0;
                  return (
                    <tr key={r.model} className={best ? 'bg-brand-50/60' : ''}>
                      <td className="py-3.5 pr-4">
                        <div className="flex items-center gap-2">
                          <span className="text-[13.5px] font-semibold text-ink">{r.model}</span>
                          {best && <Badge tone="brand"><Trophy className="h-3 w-3" /> best</Badge>}
                        </div>
                      </td>
                      <td className="py-3.5 pr-4">
                        <div className="flex items-center gap-2">
                          <div className="w-24">
                            <ProgressBar value={acc * 100} />
                          </div>
                          <span className="font-display text-[13.5px] font-semibold text-ink">{r.accuracy != null ? `${(acc * 100).toFixed(1)}%` : '—'}</span>
                        </div>
                      </td>
                      <td className="py-3.5 pr-4 text-[13.5px] text-ink-2">{r.f1_macro != null ? r.f1_macro.toFixed(3) : '—'}</td>
                      <td className="py-3.5 pr-4 text-[13.5px] text-ink-2">{r.precision_macro != null ? r.precision_macro.toFixed(3) : '—'}</td>
                      <td className="py-3.5 text-[13.5px] text-ink-2">{r.recall_macro != null ? r.recall_macro.toFixed(3) : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="How the models are trained"
        subtitle="Accuracy-critical design decisions baked into the pipeline"
      >
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {[
            { icon: Users, title: 'Leave-one-out CV', desc: 'With a small dataset a stratified split leaves almost no test data. LOOCV evaluates each held-out clip honestly.' },
            { icon: Sparkles, title: 'In-fold feature selection', desc: 'SelectKBest is fit on the training fold only, so the held-out clip never leaks into feature selection.' },
            { icon: ArrowDown, title: 'Torso-normalised features', desc: 'Distances are scaled by torso length and velocities use wall-clock seconds, so body size and frame rate cannot bias features.' },
          ].map((m) => {
            const Icon = m.icon;
            return (
              <div key={m.title} className="rounded-xl border border-line bg-surface-2 p-4">
                <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand">
                  <Icon className="h-4.5 w-4.5" />
                </div>
                <p className="text-[13.5px] font-semibold text-ink">{m.title}</p>
                <p className="mt-1 text-[12.5px] leading-relaxed text-ink-3">{m.desc}</p>
              </div>
            );
          })}
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-50 p-4 text-[12.5px] text-ink-3">
          <p>
            ML reports are predictions on a tiny dataset — not biological ground truth. See{' '}
            <code className="rounded bg-slate-200 px-1.5 py-0.5 font-mono">reports/scientific_validity.txt</code> for the full disclaimer.
          </p>
          <div className="flex gap-2">
            <a href="/reports/model_comparison.csv">
              <Button variant="secondary">Comparison CSV</Button>
            </a>
            <a href="/reports/model_results_summary.txt">
              <Button variant="secondary">Summary</Button>
            </a>
          </div>
        </div>
      </Card>
    </div>
  );
}