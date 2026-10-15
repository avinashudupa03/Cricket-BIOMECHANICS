import React, { useEffect, useState } from 'react';
import {
  Film,
  UploadCloud,
  BrainCircuit,
  Database,
  ArrowRight,
  Star,
  Activity,
} from 'lucide-react';
import { DatasetStats, NavRoute, shotLabel, SHOT_TYPES, SHOT_TYPE_LABELS, VideoMeta } from '../types';
import { api } from '../utils/api';
import { Card, Stat, Badge, Button, EmptyState, Spinner, valueOrDash } from '../components/ui';

export function DashboardView({
  onNavigate,
  onOpenVideo,
}: {
  onNavigate: (r: NavRoute) => void;
  onOpenVideo: (v: VideoMeta) => void;
}) {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    Promise.all([api.stats(), api.videos()])
      .then(([s, v]) => {
        if (!mounted) return;
        setStats(s);
        setVideos(v.videos);
      })
      .catch(() => {
        if (!mounted) return;
        setStats(null);
      })
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

  const ml = stats?.ml;
  const distribution = stats?.distribution ?? {};
  const maxCount = Math.max(1, ...Object.values(distribution));
  const rated = videos.filter((v) => v.rating !== null).length;
  const avgRating =
    videos.length > 0
      ? videos.reduce((a, v) => a + (v.rating ?? 0), 0) / Math.max(1, videos.length)
      : 0;

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-2xl bg-ink p-7 text-white">
        <div className="absolute -right-16 -top-16 h-56 w-56 rounded-full bg-brand/20 blur-3xl" />
        <div className="absolute -bottom-20 right-40 h-48 w-48 rounded-full bg-emerald-400/10 blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-2">
            <Badge tone="brand">AI-powered</Badge>
            <Badge tone="slate">Pose estimation · MediaPipe</Badge>
          </div>
          <h2 className="mt-3 font-display text-[26px] font-bold leading-tight">
            Cricket Batting Biomechanics Analysis
          </h2>
          <p className="mt-2 max-w-xl text-[14px] leading-relaxed text-slate-300">
            Upload a batting clip to detect Stance, Backlift, Downswing, Impact and
            Follow-through phases, track joint angles, and get a 0–10 shot rating backed
            by machine-learning classification.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Button variant="primary" onClick={() => onNavigate('analyze')}>
              <UploadCloud className="h-4 w-4" /> Upload a video
            </Button>
            <button
              onClick={() => onNavigate('ml')}
              className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-4 py-2.5 text-[13.5px] font-semibold text-white transition-colors hover:bg-white/20"
            >
              <BrainCircuit className="h-4 w-4" /> ML insights
            </button>
          </div>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Stat label="Processed videos" value={stats?.total_videos ?? videos.length} hint="clips with full analysis" tone="brand" />
        <Stat label="Shot classes" value={stats?.n_classes ?? 0} hint={(stats?.class_names ?? []).map(shotLabel).join(' · ') || SHOT_TYPES.map((s) => SHOT_TYPE_LABELS[s]).join(' · ')} tone="violet" />
        <Stat label="ML accuracy (LOOCV)" value={ml?.accuracy ?? '—'} hint={`best model: ${ml?.best_model ?? 'n/a'}`} tone="amber" />
        <Stat label="Average rating" value={avgRating > 0 ? avgRating.toFixed(1) : '—'} hint={`${rated}/${videos.length} rated clips`} tone="neutral" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Distribution */}
        <Card
          title="Dataset distribution"
          subtitle="Videos per shot class"
          className="lg:col-span-1"
          action={<Badge tone="brand"><Database className="h-3 w-3" /> {stats?.dataset_exists ? 'built' : 'not built'}</Badge>}
        >
          {Object.keys(distribution).length === 0 ? (
            <EmptyState
              icon={<Database className="h-5 w-5" />}
              title="No dataset yet"
              message="Process videos then run build_dataset.py to assemble the ML dataset."
            />
          ) : (
            <div className="space-y-4">
              {Object.entries(distribution).map(([name, count]) => (
                <div key={name}>
                  <div className="mb-1.5 flex items-center justify-between text-[13px]">
                    <span className="font-medium text-ink-2">{shotLabel(name)}</span>
                    <span className="font-display font-semibold text-ink">{count}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-brand"
                      style={{ width: `${(count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* ML summary */}
        <Card
          title="Model evaluation"
          subtitle={`${ml?.evaluation ?? 'Leave-One-Out CV'}`}
          className="lg:col-span-1"
          action={ml?.available ? <Badge tone="brand">available</Badge> : <Badge tone="amber">no reports</Badge>}
        >
          {!ml?.available ? (
            <EmptyState
              icon={<BrainCircuit className="h-5 w-5" />}
              title="No trained model yet"
              message="Run train_models.py to generate LOOCV evaluation reports."
            />
          ) : (
            <div className="space-y-3">
              <div className="rounded-xl bg-surface-2 p-4">
                <p className="text-[12px] text-ink-3">Best model</p>
                <p className="font-display text-[20px] font-bold text-ink">{ml.best_model}</p>
              </div>
              <div className="rounded-xl bg-surface-2 p-4">
                <p className="text-[12px] text-ink-3">Dataset size</p>
                <p className="font-display text-[20px] font-bold text-ink">
                  {ml.dataset_size ?? '—'} <span className="text-[12px] font-normal text-ink-3">videos</span>
                </p>
              </div>
              {ml.comparison_table.length > 0 && (
                <div>
                  <p className="mb-1.5 text-[12px] font-medium text-ink-3">Model comparison</p>
                  {ml.comparison_table.slice(0, 3).map((row) => (
                    <div key={row.model} className="mb-1.5 flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2 text-[13px]">
                      <span className="font-medium text-ink-2">{row.model}</span>
                      <span className="font-display font-semibold text-ink">
                        {row.accuracy !== null && row.accuracy !== undefined ? `${(row.accuracy * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              <Button variant="secondary" className="w-full" onClick={() => onNavigate('ml')}>
                View full evaluation <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </Card>

        {/* Recent videos */}
        <Card
          title="Recent analysis"
          subtitle="Newest processed clips"
          className="lg:col-span-1"
          action={
            <button onClick={() => onNavigate('history')} className="text-[13px] font-semibold text-brand hover:underline">
              View all →
            </button>
          }
        >
          {videos.length === 0 ? (
            <EmptyState
              icon={<Film className="h-5 w-5" />}
              title="No videos analysed yet"
              message="Upload your first batting clip to get pose, angle and rating results."
            >
              <Button variant="primary" onClick={() => onNavigate('analyze')}>
                <UploadCloud className="h-4 w-4" /> Upload video
              </Button>
            </EmptyState>
          ) : (
            <div className="divide-y divide-line">
              {videos.slice(0, 5).map((v) => (
                <button
                  key={v.name}
                  onClick={() => onOpenVideo(v)}
                  className="group flex w-full items-center gap-3 py-3 text-left"
                >
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand">
                    <Activity className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13.5px] font-semibold text-ink group-hover:text-brand">
                      {v.name}
                    </p>
                    <p className="text-[12px] capitalize text-ink-3">
                      {shotLabel(v.shot_type)} · {valueOrDash(v.frames, 0)} frames · {v.processed_at}
                    </p>
                  </div>
                  {v.classification?.is_unknown && (
                    <Badge tone="amber">Unknown</Badge>
                  )}
                  {!v.classification?.is_unknown &&
                    v.classification?.confidence != null && (
                      <span className="shrink-0 text-[11.5px] text-ink-3">
                        {(v.classification.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  {v.rating !== null && (
                    <span className="flex items-center gap-1 rounded-lg bg-amber-50 px-2 py-1 font-display text-[13px] font-bold text-amber-600">
                      <Star className="h-3.5 w-3.5 fill-current" /> {valueOrDash(v.rating)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}