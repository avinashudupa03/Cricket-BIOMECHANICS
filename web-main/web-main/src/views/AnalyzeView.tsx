import React, { useEffect, useRef, useState } from 'react';
import {
  UploadCloud,
  FileVideo,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowRight,
  Film,
} from 'lucide-react';
import { SHOT_TYPES, SHOT_TYPE_LABELS, VideoMeta } from '../types';
import { api } from '../utils/api';
import { Badge, Button, Card, Spinner } from '../components/ui';

type Stage =
  | { kind: 'idle' }
  | { kind: 'uploading' }
  | { kind: 'running'; jobId: string; steps: string[]; current: string }
  | { kind: 'done'; videoName: string; shotType: string }
  | { kind: 'existing'; videoName: string }
  | { kind: 'error'; message: string };

export function AnalyzeView({ onOpenVideo }: { onOpenVideo: (v: VideoMeta) => void }) {
  const [shotType, setShotType] = useState<string>('drive');
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [stage, setStage] = useState<Stage>({ kind: 'idle' });
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const reset = () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = null;
    setStage({ kind: 'idle' });
  };

  const selectFile = (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    setStage({ kind: 'idle' });
  };

  const handleUpload = async () => {
    if (!file) return;
    setStage({ kind: 'uploading' });
    try {
      const res = await api.upload(file, shotType);
      if (res.status === 'existing' && res.video_name) {
        setStage({ kind: 'existing', videoName: res.video_name });
        return;
      }
      if (res.status === 'queued' && res.job_id) {
        setStage({
          kind: 'running',
          jobId: res.job_id,
          steps: [],
          current: 'Queued for analysis…',
        });
        pollRef.current = window.setInterval(async () => {
          try {
            const p = await api.progress(res.job_id!);
            setStage((prev) =>
              prev.kind === 'running' && prev.jobId === res.job_id
                ? { ...prev, steps: p.steps, current: p.current }
                : prev
            );
            if (p.state === 'done') {
              if (pollRef.current) window.clearInterval(pollRef.current);
              pollRef.current = null;
              if (p.ok) {
                setStage({
                  kind: 'done',
                  videoName: p.video_name ?? res.video_name ?? 'video',
                  shotType: res.shot_type ?? shotType,
                });
              } else {
                setStage({
                  kind: 'error',
                  message: p.error ?? 'Processing failed on the server — see the console log.',
                });
              }
            }
          } catch {
            /* transient poll errors ignored */
          }
        }, 1500);
      } else if (res.status === 'error' || res.status === 'duplicate' || res.status === 'inflight') {
        setStage({ kind: 'error', message: res.message ?? 'Upload failed.' });
      }
    } catch (e) {
      setStage({
        kind: 'error',
        message: e instanceof Error ? e.message : 'Upload failed.',
      });
    }
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  // Mirrors PIPELINE_STEP_LABELS in app.py: the labels the job status endpoint
  // reports, in the order the pipeline runs them.
  const ACTIVE_STEPS = [
    'Detecting player pose & landmarks',
    'Calculating joint angles',
    'Detecting batting phases',
    'Extracting biomechanics features',
    'Rating the shot (0-10)',
    'Screening injury & batsman safety',
    'Rendering annotated analysis video',
  ];

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      {/* Upload card */}
      <Card
        className="lg:col-span-2"
        title="Upload a batting clip"
        subtitle="MP4, AVI, MOV or MKV · up to 512 MB"
      >
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            selectFile(e.dataTransfer.files?.[0]);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all ${
            dragOver ? 'border-brand bg-brand-50' : 'border-slate-300 bg-surface-2 hover:border-brand/60 hover:bg-brand-50/40'
          }`}
        >
          {file ? (
            <>
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-brand text-white">
                <FileVideo className="h-6 w-6" />
              </div>
              <p className="max-w-full truncate text-[14px] font-semibold text-ink">{file.name}</p>
              <p className="mt-1 text-[12.5px] text-ink-3">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
              <span className="mt-3 text-[13px] font-semibold text-brand">Click to choose a different file</span>
            </>
          ) : (
            <>
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50 text-brand">
                <UploadCloud className="h-6 w-6" />
              </div>
              <p className="text-[14px] font-semibold text-ink">Drag & drop a video here</p>
              <p className="mt-1 text-[12.5px] text-ink-3">or click to browse your files</p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv"
            className="hidden"
            onChange={(e) => selectFile(e.target.files?.[0])}
          />
        </label>

        <div className="mt-5">
          <p className="mb-2 text-[13px] font-medium text-ink-2">Shot type</p>
          <div className="grid grid-cols-3 gap-2">
            {SHOT_TYPES.map((st) => (
              <button
                key={st}
                onClick={() => setShotType(st)}
                className={`rounded-xl border px-3 py-2.5 text-[13px] font-semibold capitalize transition-all ${
                  shotType === st
                    ? 'border-brand bg-brand text-white shadow-sm'
                    : 'border-line bg-surface-2 text-ink-2 hover:border-brand/50'
                }`}
              >
                {SHOT_TYPE_LABELS[st]}
              </button>
            ))}
          </div>
        </div>

        <Button
          variant="primary"
          className="mt-5 w-full"
          disabled={!file || stage.kind === 'uploading' || stage.kind === 'running'}
          onClick={handleUpload}
        >
          {stage.kind === 'uploading' ? (
            <>
              <Spinner /> Uploading…
            </>
          ) : stage.kind === 'running' ? (
            <>
              <Spinner /> Processing…
            </>
          ) : (
            <>
              <UploadCloud className="h-4 w-4" /> Start analysis
            </>
          )}
        </Button>

        {stage.kind === 'error' && (
          <div className="mt-4 flex items-start gap-2 rounded-xl bg-red-50 p-3 text-[13px] text-red-700">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {stage.message}
          </div>
        )}

        {stage.kind === 'existing' && (
          <div className="mt-4 space-y-3">
            <div className="flex items-start gap-2 rounded-xl bg-amber-50 p-3 text-[13px] text-amber-700">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              This video was already analysed.
            </div>
            <Button variant="secondary" className="w-full" onClick={() => onOpenVideo({ name: stage.videoName, shot_type: '', rating: null, frames: null, processed_at: '', analysis_exists: false } as VideoMeta)}>
              View existing results <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {stage.kind === 'done' && (
          <div className="mt-4 space-y-3">
            <div className="flex items-start gap-2 rounded-xl bg-emerald-50 p-3 text-[13px] text-emerald-700">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                <strong>{SHOT_TYPE_LABELS[stage.shotType] ?? stage.shotType}</strong> — {stage.videoName} finished processing.
              </span>
            </div>
            <Button variant="primary" className="w-full" onClick={() => onOpenVideo({ name: stage.videoName, shot_type: stage.shotType, rating: null, frames: null, processed_at: '', analysis_exists: false } as VideoMeta)}>
              Open results <ArrowRight className="h-4 w-4" />
            </Button>
            <Button variant="ghost" className="w-full" onClick={reset}>
              Upload another clip
            </Button>
          </div>
        )}
      </Card>

      {/* Pipeline status + how it works */}
      <div className="space-y-6 lg:col-span-3">
        <Card
          title="Analysis pipeline"
          subtitle={stage.kind === 'running' ? `Job ${stage.jobId}` : 'Runs the per-video biomechanics pipeline'}
        >
          {stage.kind === 'running' ? (
            <div className="space-y-3">
              <div className="flex items-center gap-3 rounded-xl bg-brand-50 p-4">
                <Loader2 className="h-5 w-5 animate-spin text-brand" />
                <div>
                  <p className="text-[13.5px] font-semibold text-brand-600">
                    {stage.current}
                  </p>
                  <p className="text-[12px] text-ink-3">
                    {stage.steps.length}/{ACTIVE_STEPS.length} stages completed
                  </p>
                </div>
              </div>
              <div className="space-y-1.5">
                {ACTIVE_STEPS.map((s) => {
                  const idx = stage.steps.indexOf(s);
                  const done = idx >= 0;
                  const running = !done && stage.steps.length <= ACTIVE_STEPS.indexOf(s) + 1 && stage.steps.length === ACTIVE_STEPS.indexOf(s);
                  return (
                    <div
                      key={s}
                      className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] ${
                        done ? 'text-emerald-600' : 'text-ink-3'
                      }`}
                    >
                      {done ? (
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
                      ) : (
                        <div className="h-4 w-4 shrink-0 rounded-full border-2 border-slate-300" />
                      )}
                      {s}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {[
                { n: '01', label: 'Pose estimation', desc: 'MediaPipe detects 33 landmarks per frame with identity-locked tracking.' },
                { n: '02', label: 'Joint angles', desc: 'Elbow, knee, shoulder and hip angles computed per frame.' },
                { n: '03', label: 'Phase detection', desc: 'Stance → Backlift → Downswing → Impact → Follow-through.' },
                { n: '04', label: 'Shot rating', desc: 'Deterministic 0–10 rating with confidence and coaching cues.' },
                { n: '05', label: 'Injury screening', desc: 'Flags risky elbow, knee, shoulder and hip loading with coaching cues.' },
                { n: '06', label: 'Features & ML', desc: 'Torso-normalised metrics feed LOOCV shot classification.' },
                { n: '07', label: 'Analysis video', desc: 'Frame-by-frame annotated clip, H.264, browser playable.' },
              ].map((s) => (
                <div key={s.n} className="rounded-xl border border-line bg-surface-2 p-4">
                  <div className="flex items-center gap-2">
                    <span className="font-display text-[13px] font-bold text-brand">{s.n}</span>
                    <span className="text-[13px] font-semibold text-ink-2">{s.label}</span>
                  </div>
                  <p className="mt-1.5 text-[12px] leading-relaxed text-ink-3">{s.desc}</p>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card title="Supported shot types" subtitle="Classification classes used by the ML models">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            {SHOT_TYPES.map((st) => (
              <div key={st} className="rounded-xl border border-line bg-surface-2 p-4 text-center">
                <Film className="mx-auto mb-1.5 h-5 w-5 text-brand" />
                <p className="text-[13.5px] font-semibold text-ink">{SHOT_TYPE_LABELS[st]}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 flex flex-wrap gap-2 text-[12px] text-ink-3">
            <Badge tone="blue">{SHOT_TYPES.length} classes</Badge>
            <Badge tone="slate">Leave-one-out CV</Badge>
            <Badge tone="slate">Fixed seed 42</Badge>
          </p>
        </Card>
      </div>
    </div>
  );
}