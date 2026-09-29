// GENERATED FILE - DO NOT EDIT MANUALLY
// Run python generate_types.py to regenerate
// Source: app.py SHOT_TYPES / SHOT_TYPE_LABELS

export const SHOT_TYPES = [
  'cut',
  'defence',
  'drive',
  'flick',
  'pull shot',
  'unknown'
] as const;

export type ShotType = (typeof SHOT_TYPES)[number];

export const SHOT_TYPE_LABELS: Record<string, string> = {
  cut: 'Cut',
    defence: 'Defence',
    drive: 'Drive',
    flick: 'Flick',
    'pull shot': 'Pull Shot',
    unknown: 'Unknown'
};

/** Human-readable label for a shot_type value from the API. */
export function shotLabel(shotType: string | null | undefined): string {
  if (!shotType) return 'Unknown';
  return SHOT_TYPE_LABELS[shotType.toLowerCase()] ?? shotType;
}
