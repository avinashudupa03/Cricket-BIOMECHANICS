import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  PlayCircle,
  Film,
  AlertTriangle,
  Gauge,
  BarChart3,
  Activity,
  Star,
  CheckCircle2,
  HelpCircle,
} from 'lucide-react';
import { PHASE_COLORS, ResultsPayload, VideoMeta, shotLabel } from '../types';
import { api } from '../utils/api';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  MetricRow,
  ProgressBar,
  Ring,
  Spinner,
  valueOrDash,
} from '../components/ui';

export function ResultsView({
  video,
  onBack,
}: {
  video: VideoMeta | null;
  onBack: () => void;
}) {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!video?.name) {
      setError(true);
      setLoading(false);
      return;
    }
    let mounted = true;
    setLoading(true);
    setError(false);
    api
      .results(video.name)
      .then((res) => {
        if (!mounted) return;
        if (!res.found) {
          setError(true);
        } else {
          setData(res);
        }
      })
      .catch(() => mounted && setError(true))
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, [video?.name]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner className="h-6 w-6 text-brand" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <EmptyState
        icon={<Film className="h-6 w-6" />}
        title="No results found"
        message="This clip has not been analysed yet. Upload it from the Upload & Analyze page."
      >
        <Button variant="primary" onClick={onBack}>Back to history</Button>
      </EmptyState>
    );
  }

  const rating = data.rating;
  const injury = data.injury;
  const phase = data.phase_timeline;
  const cls = data.classification ?? null;
  const charts = Object.entries(data.charts ?? {}).map(([k, f]) => ({
    key: k,
    name: k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    url: `${data.output_base}${f}`,
  }));

  const impactAnglePairs = data.impact_angle_columns
    .map((col) => ({ col, label: data.impact_angle_labels?.[col] ?? col, value: data.impact_angles?.[col] ?? null }))
    .filter((a) => a.value !== null);

  const totalMs = phase ? phase.total_frames : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={onBack} className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface px-3 py-2 text-[13px] font-semibold text-ink-2 hover:bg-surface-2">
          <ArrowLeft className="h-4 w-4" /> Back
        </button>
        <div>
          <h2 className="font-display text-[20px] font-bold text-ink">{data.video_name}</h2>
          <p className="text-[12.5px] capitalize text-ink-3">
            {data.shot_type_label || data.shot_type || 'shot'} · {valueOrDash(data.frames, 0)} frames
          </p>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          {cls && (
            <Badge tone={cls.is_unknown ? 'amber' : 'brand'}>
              {cls.is_unknown ? (
                <>
                  <HelpCircle className="h-3 w-3" /> Unknown
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3" /> {shotLabel(cls.shot_type)}
                </>
              )}
            </Badge>
          )}
          {rating?.rating != null && <Badge tone="amber"><Star className="h-3 w-3 fill-current" /> Rated {valueOrDash(rating.rating)} / 10</Badge>}
          {injury.available && (
            <Badge tone={injury.summary?.risk_level?.toLowerCase().includes('low') ? 'brand' : 'red'}>
              <AlertTriangle className="h-3 w-3" /> Risk: {injury.summary?.risk_level ?? 'n/a'}
            </Badge>
          )}
        </div>
      </div>

      {/* Video + rating */}
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2 !p-0 overflow-hidden" title={undefined}>
          {data.has_analysis && data.analysis_url ? (
            <video
              key={data.analysis_url}
              src={data.analysis_url}
              controls
              className="aspect-video w-full bg-black object-contain"
            >
              <PlayCircle className="h-10 w-10 text-white" />
            </video>
          ) : (
            <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 bg-slate-100 text-slate-400">
              <Film className="h-8 w-8" />
              <p className="text-[13px]">Annotated analysis video not available</p>
            </div>
          )}
        </Card>

        <Card title="Shot rating" subtitle={rating?.classification ?? 'Analysis'}>
          {rating?.rating != null ? (
            <>
              <div className="flex items-center justify-center py-2">
                <Ring value={rating.rating} />
              </div>
              {rating.confidence != null && (
                <div className="mb-3">
                  <div className="mb-1 flex justify-between text-[12px] text-ink-3">
                    <span>Model confidence</span>
                    <span className="font-semibold text-ink">{Math.round(rating.confidence * 100)}%</span>
                  </div>
                  <ProgressBar value={rating.confidence * 100} />
                </div>
              )}
              {rating.explanation && (
                <p className="text-[13px] leading-relaxed text-ink-2">{rating.explanation}</p>
              )}
              {rating.strengths.length > 0 && (
                <div className="mt-3 space-y-1">
                  {rating.strengths.map((s) => (
                    <div key={s} className="flex items-start gap-2 text-[12.5px] text-emerald-700">
                      <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" /> {s}
                    </div>
                  ))}
                </div>
              )}
              {rating.weaknesses.length > 0 && (
                <div className="mt-2 space-y-1">
                  {rating.weaknesses.map((s) => (
                    <div key={s} className="flex items-start gap-2 text-[12.5px] text-red-600">
                      <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" /> {s}
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <EmptyState
              icon={<Gauge className="h-5 w-5" />}
              title="Rating unavailable"
              message="The deterministic rating requires biomechanics features that are missing for this clip."
            />
          )}
        </Card>
      </div>

      {/* Shot classification (model prediction, Unknown when not confident) */}
      <Card
        title="Shot classification"
        subtitle={
          cls
            ? `${cls.model_name ?? 'Classifier'} · ${cls.n_classes} classes · threshold ${(cls.threshold * 100).toFixed(0)}%`
            : 'Model-based shot identification'
        }
        action={
          cls ? (
            <Badge tone={cls.is_unknown ? 'amber' : 'brand'}>
              {cls.is_unknown ? 'Unknown' : shotLabel(cls.shot_type)}
            </Badge>
          ) : undefined
        }
      >
        {!cls ? (
          <EmptyState
            icon={<HelpCircle className="h-5 w-5" />}
            title="No classification yet"
            message="This clip was analysed before shot classification was enabled. Re-run the analysis to get a model prediction."
          />
        ) : (
          <div className="space-y-4">
            <div
              className={`flex flex-wrap items-center gap-3 rounded-xl p-4 ${
                cls.is_unknown ? 'bg-amber-50' : 'bg-brand-50'
              }`}
            >
              <div
                className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-white ${
                  cls.is_unknown ? 'bg-amber-500' : 'bg-brand'
                }`}
              >
                {cls.is_unknown ? (
                  <HelpCircle className="h-6 w-6" />
                ) : (
                  <CheckCircle2 className="h-6 w-6" />
                )}
              </div>
              <div className="min-w-0">
                <p
                  className={`font-display text-[18px] font-bold ${
                    cls.is_unknown ? 'text-amber-800' : 'text-brand-600'
                  }`}
                >
                  {cls.is_unknown ? 'Unknown' : shotLabel(cls.shot_type)}
                </p>
                <p className="text-[12.5px] text-ink-3">
                  {cls.reason_label}
                  {cls.is_unknown &&
                    cls.predicted &&
                    cls.predicted !== 'unknown' && (
                      <> · closest match was {shotLabel(cls.predicted)}</>
                    )}
                </p>
              </div>
              <div className="ml-auto text-right">
                <p className="text-[11.5px] uppercase tracking-wide text-ink-3">
                  Confidence
                </p>
                <p className="font-display text-[20px] font-bold text-ink">
                  {cls.confidence !== null
                    ? `${(cls.confidence * 100).toFixed(1)}%`
                    : '—'}
                </p>
              </div>
            </div>

            {cls.is_unknown && (
              <p className="text-[13px] leading-relaxed text-ink-2">
                The classifier was not confident enough to name a shot, so it is
                reported as <strong>Unknown</strong> rather than forcing an
                incorrect label. The full biomechanics analysis below — pose,
                joint angles, phases, movement and rating — is unaffected.
              </p>
            )}

            {Object.keys(cls.probabilities).length > 0 && (
              <div>
                <p className="mb-2 text-[12px] font-medium uppercase tracking-wide text-ink-3">
                  Class probabilities
                </p>
                <div className="space-y-1.5">
                  {Object.entries(cls.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, p]) => {
                      const isTop = name === cls.predicted;
                      return (
                        <div key={name} className="flex items-center gap-3">
                          <span
                            className={`w-24 shrink-0 truncate text-[12.5px] ${
                              isTop
                                ? 'font-semibold text-ink'
                                : 'text-ink-3'
                            }`}
                          >
                            {shotLabel(name)}
                          </span>
                          <div className="w-40">
                            <ProgressBar
                              value={p * 100}
                              color={
                                isTop
                                  ? cls.is_unknown
                                    ? 'bg-amber-500'
                                    : 'bg-brand'
                                  : 'bg-slate-300'
                              }
                            />
                          </div>
                          <span className="text-[12.5px] text-ink-2">
                            {(p * 100).toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Key metrics */}
      <Card title="Key biomechanics metrics" subtitle="Measured from pose landmarks across the swing">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <MetricRow label="Impact frame" value={valueOrDash(data.metrics?.impact_frame, 0)} />
          <MetricRow label="Impact time" value={valueOrDash(data.metrics?.impact_time_ms, 0)} unit="ms" />
          <MetricRow label="Downswing" value={valueOrDash(data.metrics?.downswing_duration_ms, 0)} unit="ms" />
          <MetricRow label="Follow-through" value={valueOrDash(data.metrics?.followthrough_duration_ms, 0)} unit="ms" />
          <MetricRow label="Max movement" value={valueOrDash(data.metrics?.maximum_movement_score)} />
          <MetricRow label="Avg movement" value={valueOrDash(data.metrics?.average_movement_score)} />
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Impact angles */}
        <Card
          title="Joint angles at impact"
          subtitle="Degrees at the moment of ball contact"
          className="lg:col-span-1"
        >
          {impactAnglePairs.length === 0 ? (
            <EmptyState title="No impact angles" message="Impact angle columns are missing from the features file." />
          ) : (
            <div className="grid grid-cols-2 gap-2.5">
              {impactAnglePairs.map((a) => (
                <div key={a.col} className="rounded-xl bg-surface-2 px-3 py-2.5">
                  <p className="text-[11.5px] text-ink-3">{a.label}</p>
                  <p className="font-display text-[17px] font-bold text-ink">{valueOrDash(a.value)}°</p>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Phase timeline */}
        <Card
          title="Batting phases"
          subtitle={`${phase?.segments.length ?? 0} phase segments over ${totalMs} frames`}
          className="lg:col-span-2"
        >
          {!phase || phase.segments.length === 0 ? (
            <EmptyState title="No phase data" message="Run phase detection to see batting phases." />
          ) : (
            <>
              {/* timeline bar */}
              <div className="flex h-9 w-full overflow-hidden rounded-xl">
                {phase.segments.map((seg, i) => {
                  const span = (seg.end ?? phase.total_frames) - seg.start;
                  const width = phase.total_frames > 0 ? (span / phase.total_frames) * 100 : 0;
                  return (
                    <div
                      key={i}
                      className="flex items-center justify-center gap-1 border-r border-white/40 text-[11px] font-semibold text-white"
                      style={{
                        width: `${width}%`,
                        backgroundColor: PHASE_COLORS[seg.phase] ?? '#64748b',
                      }}
                    >
                      <span className="truncate px-1">{seg.phase}</span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {phase.segments.map((seg, i) => (
                  <span key={i} className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-2.5 py-1 text-[12px] font-medium text-ink-2">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: PHASE_COLORS[seg.phase] ?? '#64748b' }} />
                    {seg.phase} · {seg.start}–{seg.end ?? phase.total_frames}
                  </span>
                ))}
                {phase.impact_frame != null && (
                  <Badge tone="red">Impact @ frame {phase.impact_frame}</Badge>
                )}
              </div>
            </>
          )}
        </Card>
      </div>

      {/* Charts */}
      <Card
        title="Visual analytics"
        subtitle="Generated plots from per-frame angles and phases"
        action={data.plot ? (
          <a href={`${data.output_base}${data.plot}`} target="_blank" rel="noreferrer">
            <Badge tone="blue"><BarChart3 className="h-3 w-3" /> Full angle plot</Badge>
          </a>
        ) : undefined}
      >
        {charts.length === 0 ? (
          <EmptyState
            icon={<BarChart3 className="h-5 w-5" />}
            title="Charts not generated yet"
            message="Charts are created lazily when the results page is first opened."
          />
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {charts.map((c) => (
              <div key={c.key} className="overflow-hidden rounded-xl border border-line">
                <p className="border-b border-line bg-surface-2 px-4 py-2 text-[12.5px] font-semibold text-ink-2">
                  {c.name}
                </p>
                <img src={c.url} alt={c.name} className="w-full" loading="lazy" />
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Injury + rating factors */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Injury risk & batsman safety"
          subtitle={injury.available && injury.summary ? `Safety score ${valueOrDash(injury.summary.safety_score)} / 10` : 'No screening data'}
          action={injury.available ? (
            <Badge tone={injury.summary?.risk_level?.toLowerCase().includes('low') ? 'brand' : 'red'}>
              {injury.summary?.risk_level ?? 'unknown'}
            </Badge>
          ) : undefined}
        >
          {!injury.available || !injury.summary ? (
            <EmptyState icon={<AlertTriangle className="h-5 w-5" />} title="No injury screening" message="Run injury_risk.py to screen this clip." />
          ) : (
            <>
              {injury.summary.top_concern && (
                <div className="mb-3 rounded-xl bg-amber-50 p-3 text-[13px] text-amber-800">
                  <strong>Top concern:</strong> {injury.summary.top_concern}
                </div>
              )}
              {injury.summary.summary && (
                <p className="mb-4 text-[13px] leading-relaxed text-ink-2">{injury.summary.summary}</p>
              )}
              {injury.flags.length > 0 && (
                <div className="space-y-2">
                  {injury.flags.map((f, i) => (
                    <div key={i} className="rounded-xl border border-line p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-[13px] font-semibold capitalize text-ink">
                          {f.label} <span className="text-ink-3">({f.side})</span>
                        </span>
                        <Badge tone={f.severity === 'mild' ? 'amber' : 'red'}>{f.severity}</Badge>
                      </div>
                      <p className="mt-1 text-[12px] text-ink-3">
                        Peak {valueOrDash(f.peak)} {f.units} · {f.pct.toFixed(1)}% of frames · worst in {f.worst_phase}
                      </p>
                      {f.description && <p className="mt-1 text-[12.5px] text-ink-2">{f.description}</p>}
                      {f.cue && <p className="mt-1 text-[12px] font-medium text-brand-600">Cue: {f.cue}</p>}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </Card>

        <Card title="Rating factor breakdown" subtitle="Weighted components of the 0–10 rating">
          {rating && rating.factors && rating.factors.length > 0 ? (
            <div className="space-y-3">
              {rating.factors.map((f) => (
                <div key={f.code}>
                  <div className="mb-1 flex items-center justify-between text-[13px]">
                    <span className="font-medium text-ink-2">{f.label}</span>
                    <span className="font-display font-semibold text-ink">
                      {f.score != null ? valueOrDash(f.score) : '—'} <span className="text-[11px] font-normal text-ink-3">× {f.weight}</span>
                    </span>
                  </div>
                  <ProgressBar value={f.score != null ? f.score * 10 : 0} color={f.score != null && f.score < 5 ? 'bg-red-500' : 'bg-brand'} />
                  {f.comment && <p className="mt-1 text-[12px] text-ink-3">{f.comment}</p>}
                </div>
              ))}
              {rating.unavailable_metrics.length > 0 && (
                <p className="mt-3 flex items-center gap-1.5 text-[12px] text-ink-3">
                  <Activity className="h-3.5 w-3.5" />
                  Missing metrics: {rating.unavailable_metrics.join(', ')}
                </p>
              )}
            </div>
          ) : (
            <EmptyState icon={<Gauge className="h-5 w-5" />} title="No factor data" />
          )}
        </Card>
      </div>
    </div>
  );
}