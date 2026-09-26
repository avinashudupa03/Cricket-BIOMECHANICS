import React, { useEffect, useState } from 'react';
import {
  Film,
  Star,
  Search,
  Download,
  Activity,
  ArrowRight,
} from 'lucide-react';
import { shotLabel, VideoMeta } from '../types';
import { api } from '../utils/api';
import { Badge, Button, Card, EmptyState, Spinner, valueOrDash } from '../components/ui';

export function HistoryView({ onOpenVideo }: { onOpenVideo: (v: VideoMeta) => void }) {
  const [videos, setVideos] = useState<VideoMeta[] | null>(null);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<'all' | string>('all');
  const [modelFilter, setModelFilter] = useState<'all' | string>('all');

  useEffect(() => {
    let mounted = true;
    api
      .history()
      .then((h) => mounted && setVideos(h.videos))
      .catch(() => mounted && setVideos([]));
    return () => {
      mounted = false;
    };
  }, []);

  if (videos === null) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner className="h-6 w-6 text-brand" />
      </div>
    );
  }

  const shotTypes = Array.from(new Set(videos.map((v) => v.shot_type).filter(Boolean)));
  // Classes the model actually predicted, including Unknown, so a clip the
  // classifier was unsure about can be found here.
  const modelClasses = Array.from(
    new Set(videos.map((v) => v.classification?.shot_type).filter(Boolean) as string[])
  ).sort();
  const filtered = videos.filter((v) => {
    const matchQ = (v.name || '').toLowerCase().includes(query.toLowerCase());
    const matchF = filter === 'all' || v.shot_type === filter;
    const matchM =
      modelFilter === 'all' ||
      (v.classification?.shot_type ?? 'none') === modelFilter;
    return matchQ && matchF && matchM;
  });

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search clips…"
            className="w-64 rounded-xl border border-line bg-surface py-2.5 pl-9 pr-3 text-[13.5px] text-ink outline-none focus:border-brand focus:ring-2 focus:ring-emerald-100"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`rounded-xl px-3.5 py-2 text-[13px] font-semibold transition-all ${
              filter === 'all' ? 'bg-ink text-white' : 'border border-line bg-surface text-ink-2 hover:bg-surface-2'
            }`}
          >
            All
          </button>
          {shotTypes.map((st) => (
            <button
              key={st}
              onClick={() => setFilter(st)}
              className={`rounded-xl px-3.5 py-2 text-[13px] font-semibold transition-all ${
                filter === st ? 'bg-brand text-white' : 'border border-line bg-surface text-ink-2 hover:bg-surface-2'
              }`}
            >
              {shotLabel(st)}
            </button>
          ))}
        </div>
        {modelClasses.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[12px] font-medium uppercase tracking-wide text-ink-3">
              Model
            </span>
            <button
              onClick={() => setModelFilter('all')}
              className={`rounded-xl px-3.5 py-2 text-[13px] font-semibold transition-all ${
                modelFilter === 'all' ? 'bg-ink text-white' : 'border border-line bg-surface text-ink-2 hover:bg-surface-2'
              }`}
            >
              All
            </button>
            {modelClasses.map((mc) => (
              <button
                key={mc}
                onClick={() => setModelFilter(mc)}
                className={`rounded-xl px-3.5 py-2 text-[13px] font-semibold transition-all ${
                  modelFilter === mc ? 'bg-ink text-white' : 'border border-line bg-surface text-ink-2 hover:bg-surface-2'
                }`}
              >
                {shotLabel(mc)}
              </button>
            ))}
          </div>
        )}
        <div className="ml-auto">
          <a href="/api/dataset.csv">
            <Button variant="secondary">
              <Download className="h-4 w-4" /> Export dataset
            </Button>
          </a>
        </div>
      </div>

      <Card title={`Analysis history`} subtitle={`${filtered.length} clips shown`}>
        {filtered.length === 0 ? (
          <EmptyState
            icon={<Film className="h-6 w-6" />}
            title="No analysis found"
            message="Upload a batting clip on the Upload & Analyze page to get started."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] border-collapse text-left">
              <thead>
                <tr className="border-b border-line text-[11.5px] uppercase tracking-wide text-ink-3">
                  <th className="py-3 pr-4 font-semibold">Video</th>
                  <th className="py-3 pr-4 font-semibold">Filed as</th>
                  <th className="py-3 pr-4 font-semibold">Model prediction</th>
                  <th className="py-3 pr-4 font-semibold">Frames</th>
                  <th className="py-3 pr-4 font-semibold">Rating</th>
                  <th className="py-3 pr-4 font-semibold">Processed</th>
                  <th className="py-3 font-semibold" />
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {filtered.map((v) => (
                  <tr key={v.name} className="group hover:bg-surface-2/60">
                    <td className="py-3.5 pr-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand">
                          <Film className="h-4 w-4" />
                        </div>
                        <span className="max-w-[240px] truncate text-[13.5px] font-semibold text-ink group-hover:text-brand">
                          {v.name}
                        </span>
                      </div>
                    </td>
                    <td className="py-3.5 pr-4">
                      <Badge tone={v.shot_type ? 'blue' : 'slate'}>{shotLabel(v.shot_type)}</Badge>
                    </td>
                    <td className="py-3.5 pr-4">
                      {v.classification ? (
                        <div className="flex flex-col gap-1">
                          <Badge tone={v.classification.is_unknown ? 'amber' : 'brand'}>
                            {v.classification.is_unknown ? 'Unknown' : shotLabel(v.classification.shot_type)}
                          </Badge>
                          {v.classification.confidence !== null && (
                            <span className="text-[11.5px] text-ink-3">
                              {(v.classification.confidence * 100).toFixed(0)}% confidence
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-[13px] text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-3.5 pr-4 text-[13px] text-ink-2">{valueOrDash(v.frames, 0)}</td>
                    <td className="py-3.5 pr-4">
                      {v.rating !== null ? (
                        <span className="inline-flex items-center gap-1 rounded-lg bg-amber-50 px-2 py-1 font-display text-[13px] font-bold text-amber-600">
                          <Star className="h-3.5 w-3.5 fill-current" /> {valueOrDash(v.rating)}
                        </span>
                      ) : (
                        <span className="text-[13px] text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-3.5 pr-4 text-[13px] text-ink-3">{v.processed_at}</td>
                    <td className="py-3.5 text-right">
                      <button
                        onClick={() => onOpenVideo(v)}
                        className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-[13px] font-semibold text-brand hover:bg-brand-50"
                      >
                        Open <ArrowRight className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {videos.length > 0 && (
        <p className="flex items-center gap-1.5 text-[12.5px] text-ink-3">
          <Activity className="h-3.5 w-3.5" />
          {videos.length} processed clips · {videos.filter((v) => v.rating !== null).length} rated
        </p>
      )}
    </div>
  );
}