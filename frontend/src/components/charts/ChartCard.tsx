import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { ChartDefinition } from '../../types';
import { Download, AlertCircle, Info } from 'lucide-react';
import api from '../../services/api';

interface ChartCardProps {
  chart: ChartDefinition;
  runId: string;
}

export default function ChartCard({ chart, runId }: ChartCardProps) {
  const handleExportCsv = async () => {
    try {
      const res = await api.get(`/validation/runs/${runId}/charts/${chart.chart_id}/export-csv`, {
        responseType: 'blob',
      });
      const blob = new Blob([res.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${chart.chart_id}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export CSV', err);
      // Fallback client-side CSV generator if backend endpoint is unavailable
      if (chart.data && chart.data.length > 0) {
        const headers = Object.keys(chart.data[0]);
        const csvRows = [headers.join(',')];
        for (const row of chart.data) {
          csvRows.push(headers.map((h) => JSON.stringify(row[h] ?? '')).join(','));
        }
        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `${chart.chart_id}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }
    }
  };

  const isUnevaluated = chart.status === 'NOT_EVALUATED' || chart.status === 'UNSUPPORTED';

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-xs p-5 flex flex-col justify-between hover:border-gray-300 transition-colors">
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-2">
          <div>
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-gray-900">{chart.title}</h4>
              <span
                className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${
                  isUnevaluated
                    ? 'bg-amber-100 text-amber-800'
                    : 'bg-blue-50 text-blue-700 border border-blue-200'
                }`}
              >
                {chart.status}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-0.5">{chart.description}</p>
          </div>
          {!isUnevaluated && chart.data && chart.data.length > 0 && (
            <button
              onClick={handleExportCsv}
              title="Export Tabular Data as CSV"
              className="inline-flex items-center gap-1 px-2 py-1 text-xs text-gray-600 hover:text-gray-900 bg-gray-50 hover:bg-gray-100 rounded border border-gray-200 transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-gray-500" />
              <span>CSV</span>
            </button>
          )}
        </div>

        {/* Content Body */}
        {isUnevaluated ? (
          <div className="my-6 p-4 rounded-md bg-amber-50 border border-amber-200 flex items-start gap-3">
            <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-xs font-semibold text-amber-900 uppercase tracking-wide">
                Module Not Evaluated
              </div>
              <p className="text-xs text-amber-800 mt-1 leading-relaxed">
                {chart.interpretation || chart.provenance.note || 'Required inputs or model prerequisites were not provided.'}
              </p>
            </div>
          </div>
        ) : chart.chart_type === 'heatmap' && chart.matrix ? (
          <div className="my-4 overflow-x-auto">
            <table className="min-w-full text-xs text-center border-collapse">
              <thead>
                <tr>
                  <th className="p-2 border border-gray-200 bg-gray-50 text-gray-600 font-medium">
                    Actual \ Predicted
                  </th>
                  {chart.matrix.labels.map((lbl) => (
                    <th key={lbl} className="p-2 border border-gray-200 bg-gray-50 font-semibold text-gray-700">
                      {lbl}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chart.matrix.labels.map((actLabel, rowIdx) => (
                  <tr key={actLabel}>
                    <td className="p-2 border border-gray-200 bg-gray-50 font-semibold text-gray-700 text-left">
                      {actLabel}
                    </td>
                    {chart.matrix!.labels.map((_, colIdx) => {
                      const count = chart.matrix!.raw_matrix[rowIdx]?.[colIdx] ?? 0;
                      const norm = chart.matrix!.normalized_matrix?.[rowIdx]?.[colIdx] ?? 0;
                      const isDiagonal = rowIdx === colIdx;
                      return (
                        <td
                          key={colIdx}
                          className={`p-2 border border-gray-200 font-mono transition-colors ${
                            isDiagonal
                              ? 'bg-emerald-50 text-emerald-900 font-bold'
                              : count > 0
                              ? 'bg-red-50 text-red-900 font-medium'
                              : 'text-gray-400'
                          }`}
                        >
                          <div>{(norm * 100).toFixed(1)}%</div>
                          <div className="text-[10px] text-gray-400 font-normal">({count})</div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : chart.chart_type === 'scatter' ? (
          <div className="h-64 my-3">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey={chart.x_axis.key}
                  name={chart.x_axis.label}
                  type={chart.x_axis.type === 'number' ? 'number' : undefined}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis
                  dataKey={chart.y_axis.key}
                  name={chart.y_axis.label}
                  type={chart.y_axis.type === 'number' ? 'number' : undefined}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Scatter name="Actual vs Predicted" data={chart.data} fill="#3b82f6" />
                {chart.data.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="ideal"
                    stroke="#94a3b8"
                    strokeDasharray="4 4"
                    dot={false}
                    name="Ideal (y = x)"
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : chart.chart_type === 'line' ? (
          <div className="h-64 my-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chart.data} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey={chart.x_axis.key}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                {chart.thresholds?.map((t, idx) => (
                  <ReferenceLine
                    key={idx}
                    y={t.value}
                    stroke={t.color || '#ef4444'}
                    strokeDasharray="3 3"
                    label={{
                      value: t.label,
                      fill: t.color || '#ef4444',
                      fontSize: 10,
                      position: 'top',
                    }}
                  />
                ))}
                {chart.series.map((s) => (
                  <Line
                    key={s.key}
                    type="monotone"
                    dataKey={s.key}
                    name={s.label}
                    stroke={s.color || '#3b82f6'}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-64 my-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart.data} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey={chart.x_axis.key}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                {chart.thresholds?.map((t, idx) => (
                  <ReferenceLine
                    key={idx}
                    y={t.value}
                    stroke={t.color || '#ef4444'}
                    strokeDasharray="3 3"
                    label={{
                      value: t.label,
                      fill: t.color || '#ef4444',
                      fontSize: 10,
                      position: 'top',
                    }}
                  />
                ))}
                {chart.series.map((s) => (
                  <Bar
                    key={s.key}
                    dataKey={s.key}
                    name={s.label}
                    fill={s.color || '#3b82f6'}
                    radius={[3, 3, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Footer Interpretation & Provenance */}
      <div className="mt-4 pt-3 border-t border-gray-100 flex flex-col gap-1 text-[11px] text-gray-500">
        {chart.interpretation && (
          <p className="text-gray-700 leading-snug">
            <span className="font-semibold text-gray-800">Insight:</span> {chart.interpretation}
          </p>
        )}
        <div className="flex items-center justify-between text-[10px] text-gray-400 mt-1">
          <span>
            Source: <span className="font-medium text-gray-600">{chart.provenance.source_validator}</span>
            {chart.provenance.sample_count ? ` (${chart.provenance.sample_count} samples)` : ''}
          </span>
          {chart.provenance.note && (
            <span className="italic max-w-xs truncate" title={chart.provenance.note}>
              {chart.provenance.note}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
