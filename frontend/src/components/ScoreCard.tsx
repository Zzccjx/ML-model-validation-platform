import React from 'react';

interface ScoreCardProps {
  label: string;
  score: number | null;
  icon?: React.ReactNode;
  color?: string;
  statusLabel?: string;
}

function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-600';
  if (score >= 60) return 'text-yellow-600';
  if (score >= 40) return 'text-orange-500';
  return 'text-red-600';
}

function getScoreBg(score: number): string {
  if (score >= 80) return 'bg-green-50 border-green-200';
  if (score >= 60) return 'bg-yellow-50 border-yellow-200';
  if (score >= 40) return 'bg-orange-50 border-orange-200';
  return 'bg-red-50 border-red-200';
}

export default function ScoreCard({ label, score, icon, statusLabel }: ScoreCardProps) {
  const displayScore = score !== null ? Math.round(score) : (statusLabel ? statusLabel.replace('_', ' ') : '—');
  const colorClass = score !== null ? getScoreColor(score) : 'text-slate-500';
  const bgClass = score !== null ? getScoreBg(score) : 'bg-slate-50 border-slate-200';

  return (
    <div className={`rounded-lg border p-4 ${bgClass}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold ${colorClass}`}>
        {score !== null ? (
          <>
            {Math.round(score)}
            <span className="text-xs font-normal text-gray-400"> / 100</span>
          </>
        ) : (
          <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded bg-slate-200/80 text-slate-600 tracking-wider">
            {statusLabel ? statusLabel.replace('_', ' ') : 'Not Evaluated'}
          </span>
        )}
      </div>
    </div>
  );
}
