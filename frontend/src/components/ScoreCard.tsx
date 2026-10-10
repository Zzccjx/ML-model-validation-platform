import React from 'react';

interface ScoreCardProps {
  label: string;
  score: number | null;
  icon?: React.ReactNode;
  color?: string;
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

export default function ScoreCard({ label, score, icon }: ScoreCardProps) {
  const displayScore = score !== null ? Math.round(score) : '—';
  const colorClass = score !== null ? getScoreColor(score) : 'text-gray-400';
  const bgClass = score !== null ? getScoreBg(score) : 'bg-gray-50 border-gray-200';

  return (
    <div className={`rounded-lg border p-4 ${bgClass}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
      </div>
      <div className={`text-3xl font-bold ${colorClass}`}>
        {displayScore}
        {score !== null && <span className="text-sm font-normal text-gray-400"> / 100</span>}
      </div>
    </div>
  );
}
