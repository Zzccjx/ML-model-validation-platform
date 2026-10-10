import React from 'react';

const severityConfig: Record<string, { bg: string; text: string; label: string }> = {
  critical: { bg: 'bg-red-100', text: 'text-red-800', label: 'CRITICAL' },
  high: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'HIGH' },
  medium: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'MEDIUM' },
  low: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'LOW' },
  info: { bg: 'bg-gray-100', text: 'text-gray-800', label: 'INFO' },
};

export default function SeverityBadge({ severity }: { severity: string }) {
  const config = severityConfig[severity] || severityConfig.info;
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
      {config.label}
    </span>
  );
}
