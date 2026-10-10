import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';
import { ValidationRun, Finding } from '../types';
import StatusBadge from '../components/StatusBadge';
import ScoreCard from '../components/ScoreCard';
import SeverityBadge from '../components/SeverityBadge';
import { ArrowLeft, CheckCircle, AlertTriangle, Activity, BarChart2, Database, ShieldAlert, Scale, Users, Wind, ShieldCheck, Sparkles, Download, FileText, Gauge } from 'lucide-react';

export default function ValidationRunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<ValidationRun | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [downloadingJson, setDownloadingJson] = useState(false);

  useEffect(() => {
    if (!id) return;
    Promise.all([
      api.get<ValidationRun>(`/validation/runs/${id}`),
      api.get<{ findings: Finding[]; total: number }>(`/validation/runs/${id}/findings`),
    ])
      .then(([runRes, findingsRes]) => {
        setRun(runRes.data);
        setFindings(findingsRes.data.findings || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="p-12 text-center text-gray-500">Loading validation run details...</div>;
  if (!run) return <div className="p-12 text-center text-red-500">Validation run not found.</div>;

  const categoryScores = run.category_scores_json || {};
  const perfMetrics = run.config_json?.metrics?.performance || {};
  const dqMetrics = run.config_json?.metrics?.data_quality || {};
  const fairnessMetrics = run.config_json?.metrics?.fairness || null;
  const driftMetrics = run.config_json?.metrics?.drift || null;
  const robustnessMetrics = run.config_json?.metrics?.robustness || null;
  const explainabilityMetrics = run.config_json?.metrics?.explainability || null;
  const reliabilityMetrics = run.config_json?.metrics?.reliability || null;

  const isClassification = perfMetrics.confusion_matrix !== undefined;
  const isRegression = perfMetrics.r2 !== undefined;
  const hasDataQuality = dqMetrics.row_count !== undefined;
  const hasFairness = Boolean(fairnessMetrics && fairnessMetrics.attributes && Object.keys(fairnessMetrics.attributes).length > 0);
  const hasDrift = Boolean(driftMetrics && driftMetrics.feature_drift && Object.keys(driftMetrics.feature_drift).length > 0);
  const hasRobustness = Boolean(robustnessMetrics && robustnessMetrics.perturbation_curve && robustnessMetrics.perturbation_curve.length > 0);
  const hasExplainability = Boolean(explainabilityMetrics && explainabilityMetrics.features && explainabilityMetrics.features.length > 0);
  const hasReliability = Boolean(reliabilityMetrics && (reliabilityMetrics.expected_calibration_error !== undefined || reliabilityMetrics.dispersion_ratio !== undefined));

  const handleDownloadPdf = async () => {
    if (!run) return;
    try {
      setDownloadingPdf(true);
      const res = await api.get(`/reports/${run.id}/pdf`, { responseType: 'blob' });
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `validation_report_${run.run_number}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download PDF report', err);
      alert('Error downloading PDF report. Please try again.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleDownloadJson = async () => {
    if (!run) return;
    try {
      setDownloadingJson(true);
      const res = await api.get(`/reports/${run.id}/json`);
      const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `validation_report_${run.run_number}.json`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download JSON report', err);
      alert('Error downloading JSON report. Please try again.');
    } finally {
      setDownloadingJson(false);
    }
  };

  const filteredFindings = selectedSeverity === 'all'
    ? findings
    : findings.filter(f => f.severity.toLowerCase() === selectedSeverity.toLowerCase());

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Navigation & Header */}
      <div>
        <Link to="/validation-runs" className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 mb-3">
          <ArrowLeft className="w-4 h-4" /> Back to Validation Runs
        </Link>
        <div className="flex flex-wrap justify-between items-center gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-gray-900">{run.run_number}</h1>
              <StatusBadge status={run.status} />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Started: {run.started_at ? new Date(run.started_at).toLocaleString() : '-'} | Duration: {run.duration_seconds ? `${run.duration_seconds.toFixed(2)}s` : '-'}
            </p>
          </div>

          {/* Export Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadPdf}
              disabled={downloadingPdf}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-900 rounded-md shadow-xs transition-colors disabled:opacity-50"
            >
              <Download className="w-4 h-4" />
              {downloadingPdf ? 'Exporting PDF...' : 'Download PDF Report'}
            </button>
            <button
              onClick={handleDownloadJson}
              disabled={downloadingJson}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded-md shadow-xs transition-colors disabled:opacity-50"
            >
              <FileText className="w-4 h-4" />
              {downloadingJson ? 'Exporting JSON...' : 'Download JSON Report'}
            </button>
          </div>
        </div>
      </div>

      {run.error_message && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-md">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-sm text-red-700 font-medium">{run.error_message}</p>
          </div>
        </div>
      )}

      {/* Platform Model Health Score & Category Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-1">
          <ScoreCard label="Model Health Score" score={run.overall_score} />
        </div>
        <div className="md:col-span-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <ScoreCard label="Performance" score={categoryScores.performance ?? null} />
          <ScoreCard label="Data Quality" score={categoryScores.data_quality ?? null} />
          <ScoreCard label="Fairness" score={categoryScores.fairness ?? null} />
          <ScoreCard label="Drift" score={categoryScores.drift ?? null} />
          <ScoreCard label="Robustness" score={categoryScores.robustness ?? null} />
          <ScoreCard label="Explainability" score={categoryScores.explainability ?? null} />
          <ScoreCard label="Reliability" score={categoryScores.reliability ?? null} />
        </div>
      </div>

      {/* Module 1: Classification Performance Diagnostics */}
      {isClassification && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 1 — Classification Performance Metrics</h3>
            </div>
            <span className="text-xs font-medium px-2.5 py-0.5 rounded bg-blue-100 text-blue-800">
              {perfMetrics.num_classes} Classes Detected
            </span>
          </div>

          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Accuracy</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{(perfMetrics.accuracy * 100).toFixed(1)}%</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Balanced Acc</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{(perfMetrics.balanced_accuracy * 100).toFixed(1)}%</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Macro F1</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.f1_macro?.toFixed(3)}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Weighted F1</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.f1_weighted?.toFixed(3)}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">ROC-AUC</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.roc_auc ? perfMetrics.roc_auc.toFixed(3) : 'N/A'}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">PR-AUC</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.pr_auc ? perfMetrics.pr_auc.toFixed(3) : 'N/A'}</div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">Confusion Matrix (Normalized / Raw)</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 border border-gray-200 bg-gray-50 text-gray-600">Actual \ Predicted</th>
                        {perfMetrics.confusion_matrix?.labels.map((lbl: string) => (
                          <th key={lbl} className="p-2 border border-gray-200 bg-gray-50 font-semibold text-gray-700">Class {lbl}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {perfMetrics.confusion_matrix?.matrix.map((row: number[], rowIdx: number) => {
                        const actualLabel = perfMetrics.confusion_matrix.labels[rowIdx];
                        return (
                          <tr key={rowIdx}>
                            <td className="p-2 border border-gray-200 bg-gray-50 font-semibold text-gray-700">Class {actualLabel}</td>
                            {row.map((cellVal: number, colIdx: number) => {
                              const normVal = perfMetrics.confusion_matrix.normalized?.[rowIdx]?.[colIdx] ?? 0;
                              const isDiagonal = rowIdx === colIdx;
                              return (
                                <td key={colIdx} className={`p-2 border border-gray-200 font-mono ${isDiagonal ? 'bg-green-50 text-green-900 font-bold' : 'text-gray-600'}`}>
                                  <div>{(normVal * 100).toFixed(1)}%</div>
                                  <div className="text-[10px] text-gray-400">({cellVal})</div>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">Per-Class Detailed Performance</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-600">Class</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Precision</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Recall</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">F1-Score</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {Object.entries(perfMetrics.per_class_metrics || {}).map(([cName, cMet]: [string, any]) => (
                        <tr key={cName}>
                          <td className="px-3 py-2 font-medium text-gray-800">Class {cName}</td>
                          <td className="px-3 py-2 text-right font-mono">{(cMet.precision * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right font-mono">
                            <span className={cMet.recall < 0.5 ? 'text-red-600 font-bold' : ''}>
                              {(cMet.recall * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right font-mono">{cMet.f1_score.toFixed(3)}</td>
                          <td className="px-3 py-2 text-right text-gray-500">{cMet.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Module 1: Regression Performance Diagnostics */}
      {isRegression && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 1 — Regression Performance & Residuals</h3>
            </div>
          </div>

          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">R² Score</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.r2?.toFixed(3)}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">MAE</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.mae?.toFixed(3)}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">RMSE</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.rmse?.toFixed(3)}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">MAPE</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.mape !== null ? `${perfMetrics.mape.toFixed(1)}%` : 'N/A'}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Max Error</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{perfMetrics.residuals?.max_error?.toFixed(3)}</div>
              </div>
            </div>

            {perfMetrics.residuals && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-800 mb-2">Residual Distribution & Bias</h4>
                  <ul className="text-xs space-y-2 text-gray-700">
                    <li className="flex justify-between border-b pb-1">
                      <span>Mean Residual (Systematic Bias):</span>
                      <span className="font-mono font-bold">{perfMetrics.residuals.mean_residual}</span>
                    </li>
                    <li className="flex justify-between border-b pb-1">
                      <span>Residual Standard Deviation:</span>
                      <span className="font-mono">{perfMetrics.residuals.std_residual}</span>
                    </li>
                    <li className="flex justify-between border-b pb-1">
                      <span>Residual Skewness:</span>
                      <span className="font-mono">{perfMetrics.residuals.skewness}</span>
                    </li>
                  </ul>
                </div>

                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-800 mb-2">Absolute Error Quantiles</h4>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">P25</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p25}</div>
                    </div>
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">Median (P50)</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p50}</div>
                    </div>
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">P75</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p75}</div>
                    </div>
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">P90</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p90}</div>
                    </div>
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">P95</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p95}</div>
                    </div>
                    <div className="p-2 bg-gray-50 rounded border">
                      <div className="text-gray-400">P99</div>
                      <div className="font-mono font-semibold">{perfMetrics.residuals.quantiles?.p99}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Module 2: Data Quality Diagnostics */}
      {hasDataQuality && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 2 — Data Quality Diagnostics</h3>
            </div>
            <span className={`text-xs font-semibold px-2.5 py-0.5 rounded ${
              dqMetrics.imbalance_level === 'EXTREME'
                ? 'bg-red-100 text-red-800'
                : dqMetrics.imbalance_level === 'HIGH'
                ? 'bg-amber-100 text-amber-800'
                : 'bg-green-100 text-green-800'
            }`}>
              Class Balance: {dqMetrics.imbalance_level}
            </span>
          </div>

          <div className="p-6 space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Overall Missing</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{dqMetrics.overall_missing_percentage}%</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Duplicate Rows</div>
                <div className="text-xl font-bold text-gray-800 mt-1">
                  {dqMetrics.duplicate_rows_percentage}%
                  <span className="text-[11px] font-normal text-gray-400 block">({dqMetrics.duplicate_rows_count})</span>
                </div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Constant Cols</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{dqMetrics.constant_columns?.length || 0}</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Outlier Density</div>
                <div className="text-xl font-bold text-gray-800 mt-1">{dqMetrics.outlier_percentage}%</div>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                <div className="text-xs text-gray-500">Leakage Suspects</div>
                <div className={`text-xl font-bold mt-1 ${dqMetrics.potential_leakage_columns?.length > 0 ? 'text-red-600' : 'text-gray-800'}`}>
                  {dqMetrics.potential_leakage_columns?.length || 0}
                </div>
              </div>
            </div>

            {/* Feature Health & Leakage Table */}
            {dqMetrics.feature_stats && (
              <div className="border border-gray-200 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">Feature Health, Outliers & Leakage Indicators</h4>
                <div className="overflow-x-auto max-h-72 overflow-y-auto">
                  <table className="min-w-full text-xs divide-y divide-gray-200">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-600">Feature</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-600">Type</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Missing %</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Variance</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Outlier Count</th>
                        <th className="px-3 py-2 text-right font-medium text-gray-600">Target Corr (|r|)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {Object.entries(dqMetrics.feature_stats).map(([fName, fStat]: [string, any]) => {
                        const mInfo = dqMetrics.missing_values?.[fName];
                        const mPct = mInfo?.percentage ?? 0;
                        const oCount = dqMetrics.outlier_counts?.[fName] ?? 0;
                        const corr = fStat.target_correlation;
                        const isLeakage = corr !== undefined && corr >= 0.95;

                        return (
                          <tr key={fName} className={isLeakage ? 'bg-red-50/50' : ''}>
                            <td className="px-3 py-2 font-medium text-gray-800 flex items-center gap-1.5">
                              {isLeakage && <ShieldAlert className="w-3.5 h-3.5 text-red-500" />}
                              {fName}
                            </td>
                            <td className="px-3 py-2 text-gray-500 font-mono">{fStat.dtype}</td>
                            <td className={`px-3 py-2 text-right font-mono ${mPct > 5 ? 'text-amber-600 font-semibold' : 'text-gray-600'}`}>
                              {mPct}%
                            </td>
                            <td className="px-3 py-2 text-right font-mono text-gray-600">
                              {fStat.variance !== null ? fStat.variance : '-'}
                            </td>
                            <td className="px-3 py-2 text-right font-mono text-gray-600">
                              {oCount > 0 ? oCount : '-'}
                            </td>
                            <td className={`px-3 py-2 text-right font-mono ${isLeakage ? 'text-red-600 font-bold' : 'text-gray-600'}`}>
                              {corr !== undefined ? corr : '-'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Module 3: Fairness & Bias Diagnostics */}
      {hasFairness && fairnessMetrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Scale className="w-5 h-5 text-indigo-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 3 — Model Fairness & Bias Validation</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium px-2.5 py-0.5 rounded bg-indigo-100 text-indigo-800">
                Protected Attributes: {fairnessMetrics.sensitive_attributes?.join(', ')}
              </span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-gray-200 text-gray-700">
                Target Positive Class: {fairnessMetrics.positive_label}
              </span>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* Responsible AI Disclaimer Banner */}
            <div className="bg-blue-50/60 border border-blue-200/80 rounded-md p-3 text-xs text-blue-800 flex items-center gap-2">
              <Users className="w-4 h-4 text-blue-600 flex-shrink-0" />
              <span>
                <strong>Responsible AI Note:</strong> Evaluations are scoped strictly within the tested sample distribution and defined demographic groups. Compliance indicates satisfaction of statistical thresholds (such as EEOC 80% Four-Fifths rule and equal opportunity parity) rather than an absolute guarantee of real-world fairness.
              </span>
            </div>

            {Object.entries(fairnessMetrics.attributes || {}).map(([attrName, attrData]: [string, any]) => {
              const di = attrData.disparate_impact_ratio;
              const dpDiff = attrData.demographic_parity_difference;
              const eoDiff = attrData.equal_opportunity_difference;
              const eqOddsDiff = attrData.equalized_odds_difference;
              const isCompliant = attrData.four_fifths_compliant;

              return (
                <div key={attrName} className="space-y-4 border border-gray-200 rounded-lg p-5 bg-white">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-100 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold uppercase text-gray-500">Sensitive Attribute:</span>
                      <span className="text-base font-bold text-gray-900">{attrName}</span>
                    </div>
                    <div>
                      {isCompliant ? (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200">
                          <CheckCircle className="w-3.5 h-3.5" /> 80% Rule Compliant (DI &ge; 0.80)
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
                          <AlertTriangle className="w-3.5 h-3.5" /> Four-Fifths Violation (DI &lt; 0.80)
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Summary Metric Cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-xs text-gray-500 font-medium">Disparate Impact Ratio</p>
                      <p className={`text-xl font-bold font-mono mt-1 ${isCompliant ? 'text-green-700' : 'text-red-600'}`}>
                        {di !== undefined ? di : '-'}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">Threshold &ge; 0.80</p>
                    </div>

                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-xs text-gray-500 font-medium">Demographic Parity Diff</p>
                      <p className={`text-xl font-bold font-mono mt-1 ${dpDiff > 0.15 ? 'text-amber-600' : 'text-gray-900'}`}>
                        {dpDiff !== undefined ? dpDiff : '-'}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">Max Selection Rate Gap</p>
                    </div>

                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-xs text-gray-500 font-medium">Equal Opportunity Diff (&Delta;TPR)</p>
                      <p className={`text-xl font-bold font-mono mt-1 ${eoDiff > 0.10 ? 'text-amber-600' : 'text-gray-900'}`}>
                        {eoDiff !== undefined ? eoDiff : '-'}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">Recall Gap (&le; 0.10)</p>
                    </div>

                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-xs text-gray-500 font-medium">Equalized Odds Diff</p>
                      <p className={`text-xl font-bold font-mono mt-1 ${eqOddsDiff > 0.10 ? 'text-amber-600' : 'text-gray-900'}`}>
                        {eqOddsDiff !== undefined ? eqOddsDiff : '-'}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">Max(TPR Gap, FPR Gap)</p>
                    </div>
                  </div>

                  {/* Subgroup Breakdown Table */}
                  {attrData.subgroups && (
                    <div className="mt-3">
                      <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                        Subgroup Metrics Breakdown
                      </h4>
                      <div className="overflow-x-auto border border-gray-200 rounded-md">
                        <table className="min-w-full text-xs divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-3 py-2 text-left font-medium text-gray-600">Subgroup Value</th>
                              <th className="px-3 py-2 text-right font-medium text-gray-600">Sample Count</th>
                              <th className="px-3 py-2 text-right font-medium text-gray-600">Selection Rate (P(ŷ=1))</th>
                              <th className="px-3 py-2 text-right font-medium text-gray-600">True Positive Rate (Recall)</th>
                              <th className="px-3 py-2 text-right font-medium text-gray-600">False Positive Rate</th>
                              <th className="px-3 py-2 text-right font-medium text-gray-600">Accuracy</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-100 bg-white">
                            {Object.entries(attrData.subgroups).map(([groupVal, gMetrics]: [string, any]) => {
                              const isLowSample = gMetrics.sample_count < 30;
                              return (
                                <tr key={groupVal} className={isLowSample ? 'bg-amber-50/40' : ''}>
                                  <td className="px-3 py-2 font-medium text-gray-900 flex items-center gap-1.5">
                                    {groupVal}
                                    {isLowSample && (
                                      <span className="text-[10px] px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded font-normal">
                                        N &lt; 30
                                      </span>
                                    )}
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-700">
                                    {gMetrics.sample_count}
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-900 font-semibold">
                                    {(gMetrics.selection_rate * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-700">
                                    {(gMetrics.true_positive_rate * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-700">
                                    {(gMetrics.false_positive_rate * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-700">
                                    {(gMetrics.accuracy * 100).toFixed(1)}%
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Module 4: Data & Concept Drift Diagnostics */}
      {hasDrift && driftMetrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Wind className="w-5 h-5 text-teal-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 4 — Data & Concept Drift Detection</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                driftMetrics.overall_drift_status === 'stable'
                  ? 'bg-green-100 text-green-800 border-green-200'
                  : driftMetrics.overall_drift_status === 'moderate'
                  ? 'bg-amber-100 text-amber-800 border-amber-200'
                  : 'bg-red-100 text-red-800 border-red-200'
              }`}>
                {driftMetrics.overall_drift_status?.toUpperCase()} DRIFT STATUS
              </span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                {driftMetrics.is_sequential_partition ? 'Sequential 50/50 Partition' : 'Reference Dataset Mode'}
              </span>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* KPI Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Drifted Features</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {driftMetrics.drifted_features_count} / {driftMetrics.total_features_evaluated}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">{driftMetrics.drifted_features_percentage}% of features</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Max Feature PSI</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  driftMetrics.max_psi >= 0.25 ? 'text-red-600' : driftMetrics.max_psi >= 0.10 ? 'text-amber-600' : 'text-green-700'
                }`}>
                  {driftMetrics.max_psi}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Threshold: &lt; 0.10 Stable</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Concept / Pred PSI</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  driftMetrics.concept_drift?.concept_drift_detected ? 'text-red-600' : 'text-green-700'
                }`}>
                  {driftMetrics.concept_drift?.prediction_psi !== undefined ? driftMetrics.concept_drift.prediction_psi : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Prediction Output Shift</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Sample Split (Ref / Curr)</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {driftMetrics.reference_row_count} / {driftMetrics.current_row_count}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Evaluation rows</p>
              </div>
            </div>

            {/* Feature Drift Breakdown Table */}
            <div>
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                Feature Distribution Drift & Stability Index
              </h4>
              <div className="overflow-x-auto border border-gray-200 rounded-md">
                <table className="min-w-full text-xs divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Feature</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Type</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">KS Statistic</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">KS p-value</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">PSI</th>
                      <th className="px-3 py-2 text-center font-medium text-gray-600">Drift Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 bg-white">
                    {Object.entries(driftMetrics.feature_drift || {}).map(([fCol, fData]: [string, any]) => {
                      const isDrifted = fData.drift_detected;
                      const statusColor = fData.drift_status === 'significant' ? 'text-red-700 bg-red-100 border-red-200' :
                        fData.drift_status === 'moderate' ? 'text-amber-800 bg-amber-100 border-amber-200' : 'text-green-800 bg-green-100 border-green-200';

                      return (
                        <tr key={fCol} className={isDrifted ? 'bg-amber-50/30' : ''}>
                          <td className="px-3 py-2 font-medium text-gray-900">{fCol}</td>
                          <td className="px-3 py-2 text-gray-500 font-mono">{fData.type}</td>
                          <td className="px-3 py-2 text-right font-mono text-gray-700">
                            {fData.ks_statistic !== null ? fData.ks_statistic : '-'}
                          </td>
                          <td className={`px-3 py-2 text-right font-mono ${fData.ks_pvalue !== null && fData.ks_pvalue < 0.05 ? 'text-amber-600 font-semibold' : 'text-gray-600'}`}>
                            {fData.ks_pvalue !== null ? fData.ks_pvalue : '-'}
                          </td>
                          <td className={`px-3 py-2 text-right font-mono font-bold ${fData.psi >= 0.25 ? 'text-red-600' : fData.psi >= 0.10 ? 'text-amber-600' : 'text-green-700'}`}>
                            {fData.psi}
                          </td>
                          <td className="px-3 py-2 text-center">
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase border ${statusColor}`}>
                              {fData.drift_status}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Module 5: Robustness & Adversarial Diagnostics */}
      {hasRobustness && robustnessMetrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-600" />
              <h3 className="text-base font-semibold text-gray-900">Module 5 — Model Robustness & Adversarial Testing</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                robustnessMetrics.robustness_rating === 'RESILIENT'
                  ? 'bg-green-100 text-green-800 border-green-200'
                  : robustnessMetrics.robustness_rating === 'MODERATE'
                  ? 'bg-amber-100 text-amber-800 border-amber-200'
                  : 'bg-red-100 text-red-800 border-red-200'
              }`}>
                {robustnessMetrics.robustness_rating} STABILITY
              </span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                {robustnessMetrics.numeric_features_tested} Features Perturbed
              </span>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* KPI Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">5% Noise Flip Rate</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  robustnessMetrics.flip_rate_5pct >= 0.15 ? 'text-red-600' : 'text-green-700'
                }`}>
                  {(robustnessMetrics.flip_rate_5pct * 100).toFixed(1)}%
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Threshold: &lt; 15%</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">10% Noise Flip Rate</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  robustnessMetrics.flip_rate_10pct >= 0.25 ? 'text-amber-600' : 'text-gray-900'
                }`}>
                  {(robustnessMetrics.flip_rate_10pct * 100).toFixed(1)}%
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Threshold: &lt; 20%</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Clean Baseline Score</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {robustnessMetrics.clean_metric_value !== null ? robustnessMetrics.clean_metric_value.toFixed(4) : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Unperturbed Input</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">20% Masking Flip Rate</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {robustnessMetrics.masking_sensitivity?.mask_20pct?.flip_rate !== undefined
                    ? `${(robustnessMetrics.masking_sensitivity.mask_20pct.flip_rate * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Missingness Tolerance</p>
              </div>
            </div>

            {/* Perturbation Curve Table */}
            <div>
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                Calibrated Gaussian Noise Perturbation Curve
              </h4>
              <div className="overflow-x-auto border border-gray-200 rounded-md">
                <table className="min-w-full text-xs divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Noise Intensity (&sigma; Ratio)</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Prediction Flip Rate</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Flipped Predictions Count</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Performance Under Noise</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Metric Degradation (&Delta;)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 bg-white">
                    {robustnessMetrics.perturbation_curve?.map((curveItem: any) => {
                      const isHighFlip = curveItem.prediction_flip_rate >= 0.15;
                      return (
                        <tr key={curveItem.noise_percentage} className={isHighFlip ? 'bg-red-50/20' : ''}>
                          <td className="px-3 py-2 font-medium text-gray-900">{curveItem.noise_percentage} std ({curveItem.noise_sigma})</td>
                          <td className={`px-3 py-2 text-right font-mono font-bold ${isHighFlip ? 'text-red-600' : 'text-gray-900'}`}>
                            {(curveItem.prediction_flip_rate * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-gray-700">{curveItem.flipped_count}</td>
                          <td className="px-3 py-2 text-right font-mono text-gray-800">
                            {curveItem.performance_score !== null ? curveItem.performance_score : '-'}
                          </td>
                          <td className={`px-3 py-2 text-right font-mono ${curveItem.metric_degradation !== null && curveItem.metric_degradation > 0.05 ? 'text-red-600 font-semibold' : 'text-gray-600'}`}>
                            {curveItem.metric_degradation !== null ? `-${curveItem.metric_degradation}` : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Module 6: Explainability & Feature Importance Diagnostics */}
      {hasExplainability && explainabilityMetrics && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-500" />
              <h3 className="text-base font-semibold text-gray-900">Module 6 — Model Explainability & Feature Importance</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                explainabilityMetrics.distribution_balance === 'BALANCED'
                  ? 'bg-green-100 text-green-800 border-green-200'
                  : explainabilityMetrics.distribution_balance === 'MODERATE'
                  ? 'bg-amber-100 text-amber-800 border-amber-200'
                  : 'bg-red-100 text-red-800 border-red-200'
              }`}>
                {explainabilityMetrics.distribution_balance} ATTRIBUTION
              </span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded font-mono">
                Method: {explainabilityMetrics.attribution_method}
              </span>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* KPI Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Top Influential Feature</p>
                <p className="text-xl font-bold font-mono text-gray-900 mt-1 truncate" title={explainabilityMetrics.top_feature_name}>
                  {explainabilityMetrics.top_feature_name}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Primary Decision Signal</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Top Feature Share</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  explainabilityMetrics.top_feature_share_percentage >= 60.0 ? 'text-red-600' : 'text-gray-900'
                }`}>
                  {explainabilityMetrics.top_feature_share_percentage}%
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Threshold: &lt; 60%</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Top 3 Features Share</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {explainabilityMetrics.top_3_share_percentage}%
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Cumulative Contribution</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Low-Impact Features</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {explainabilityMetrics.low_impact_features_count} / {explainabilityMetrics.total_features_analyzed}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">&lt; 1% Relative Share</p>
              </div>
            </div>

            {/* Feature Importance Attribution Breakdown Table with Progress Bars */}
            <div>
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                Global Feature Attribution & Importance Ranking
              </h4>
              <div className="overflow-x-auto border border-gray-200 rounded-md">
                <table className="min-w-full text-xs divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-center font-medium text-gray-600 w-12">Rank</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Feature</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Raw Score</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600 min-w-[200px]">Relative Share (%)</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Cumulative (%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 bg-white">
                    {explainabilityMetrics.features?.map((fEntry: any) => {
                      const isDominant = fEntry.relative_percentage >= 50.0;
                      return (
                        <tr key={fEntry.feature} className={isDominant ? 'bg-amber-50/20' : ''}>
                          <td className="px-3 py-2 text-center font-mono font-bold text-gray-500">#{fEntry.rank}</td>
                          <td className="px-3 py-2 font-medium text-gray-900">{fEntry.feature}</td>
                          <td className="px-3 py-2 text-right font-mono text-gray-600">{fEntry.importance_score}</td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    isDominant ? 'bg-amber-500' : 'bg-blue-600'
                                  }`}
                                  style={{ width: `${Math.min(100, Math.max(2, fEntry.relative_percentage))}%` }}
                                />
                              </div>
                              <span className="font-mono text-gray-800 font-semibold w-12 text-right">
                                {fEntry.relative_percentage}%
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-gray-600">{fEntry.cumulative_percentage}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Module 7: Model Reliability & Probability Calibration Diagnostics */}
      {hasReliability && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Gauge className="w-5 h-5 text-indigo-600" />
              <h3 className="text-base font-semibold text-gray-900">
                Module 7 — Model Reliability & Probability Calibration
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                reliabilityMetrics.expected_calibration_error <= 0.08
                  ? 'bg-green-100 text-green-800 border-green-200'
                  : reliabilityMetrics.expected_calibration_error <= 0.15
                  ? 'bg-amber-100 text-amber-800 border-amber-200'
                  : 'bg-red-100 text-red-800 border-red-200'
              }`}>
                {reliabilityMetrics.expected_calibration_error !== undefined
                  ? reliabilityMetrics.expected_calibration_error <= 0.08
                    ? 'WELL-CALIBRATED'
                    : reliabilityMetrics.expected_calibration_error <= 0.15
                    ? 'MODERATE CALIBRATION'
                    : 'MISCALIBRATED'
                  : 'REGRESSION DISPERSION'}
              </span>
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded font-mono">
                {reliabilityMetrics.classification_mode || reliabilityMetrics.prediction_type}
              </span>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* KPI Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Expected Calibration Error (ECE)</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  (reliabilityMetrics.expected_calibration_error ?? 0) > 0.15 ? 'text-red-600' : 'text-gray-900'
                }`}>
                  {reliabilityMetrics.expected_calibration_error !== undefined
                    ? reliabilityMetrics.expected_calibration_error
                    : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Benchmark: &le; 0.08</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Brier Score</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  (reliabilityMetrics.brier_score ?? 0) > 0.25 ? 'text-amber-600' : 'text-gray-900'
                }`}>
                  {reliabilityMetrics.brier_score !== undefined
                    ? reliabilityMetrics.brier_score
                    : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Mean Sq Error: &le; 0.25</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Mean Prediction Confidence</p>
                <p className="text-2xl font-bold font-mono text-gray-900 mt-1">
                  {reliabilityMetrics.mean_prediction_confidence !== undefined
                    ? `${(reliabilityMetrics.mean_prediction_confidence * 100).toFixed(1)}%`
                    : 'N/A'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">Average Winning Confidence</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                <p className="text-xs text-gray-500 font-medium">Overconfident Errors (&ge;90%)</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  (reliabilityMetrics.overconfident_errors_count ?? 0) > 0 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {reliabilityMetrics.overconfident_errors_count !== undefined
                    ? reliabilityMetrics.overconfident_errors_count
                    : '0'}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">
                  Rate: {reliabilityMetrics.overconfident_error_rate !== undefined ? `${(reliabilityMetrics.overconfident_error_rate * 100).toFixed(1)}%` : '0%'}
                </p>
              </div>
            </div>

            {/* Calibration Bins Breakdown Table */}
            {reliabilityMetrics.calibration_bins && reliabilityMetrics.calibration_bins.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                  Empirical Probability Calibration Bins (10 Quantile Intervals)
                </h4>
                <div className="overflow-x-auto border border-gray-200 rounded-md">
                  <table className="min-w-full text-xs divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left font-semibold text-gray-600">Bin Range</th>
                        <th className="px-3 py-2 text-right font-semibold text-gray-600">Sample Count</th>
                        <th className="px-3 py-2 text-right font-semibold text-gray-600">Mean Confidence</th>
                        <th className="px-3 py-2 text-right font-semibold text-gray-600">Observed Accuracy</th>
                        <th className="px-3 py-2 text-right font-semibold text-gray-600">Calibration Gap (|Acc - Conf|)</th>
                        <th className="px-3 py-2 text-center font-semibold text-gray-600">Alignment</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 bg-white">
                      {reliabilityMetrics.calibration_bins.map((bin: any) => {
                        const gap = bin.calibration_gap ?? 0;
                        const isSevere = gap > 0.15;
                        const isModerate = gap > 0.08;
                        return (
                          <tr key={bin.bin_index} className={isSevere ? 'bg-red-50/20' : ''}>
                            <td className="px-3 py-2 font-mono font-medium text-gray-900">{bin.bin_range}</td>
                            <td className="px-3 py-2 text-right font-mono text-gray-600">{bin.sample_count}</td>
                            <td className="px-3 py-2 text-right font-mono text-gray-700">
                              {bin.mean_confidence !== null ? bin.mean_confidence : '-'}
                            </td>
                            <td className="px-3 py-2 text-right font-mono text-gray-700">
                              {bin.observed_accuracy !== null ? bin.observed_accuracy : '-'}
                            </td>
                            <td className="px-3 py-2 text-right font-mono font-semibold">
                              <span className={isSevere ? 'text-red-600' : isModerate ? 'text-amber-600' : 'text-green-600'}>
                                {gap.toFixed(4)}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-center">
                              {bin.sample_count === 0 ? (
                                <span className="text-[11px] text-gray-400">Empty</span>
                              ) : isSevere ? (
                                <span className="text-[10px] font-semibold text-red-700 bg-red-100 px-2 py-0.5 rounded">High Gap</span>
                              ) : isModerate ? (
                                <span className="text-[10px] font-semibold text-amber-700 bg-amber-100 px-2 py-0.5 rounded">Moderate</span>
                              ) : (
                                <span className="text-[10px] font-semibold text-green-700 bg-green-100 px-2 py-0.5 rounded">Calibrated</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Findings Section with Filter Chips */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-gray-900">Validation Findings</h3>
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full font-medium">
              {filteredFindings.length}
            </span>
          </div>

          <div className="flex flex-wrap gap-1 text-xs">
            {['all', 'critical', 'high', 'medium', 'low', 'info'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSelectedSeverity(sev)}
                className={`px-2.5 py-1 rounded-md uppercase font-medium transition-colors ${
                  selectedSeverity === sev
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6">
          {filteredFindings.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-sm">
              <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2 opacity-60" />
              No findings matching the selected filter.
            </div>
          ) : (
            <div className="space-y-4">
              {filteredFindings.map((finding) => (
                <div
                  key={finding.id}
                  className="p-4 rounded-lg border border-gray-200 bg-white hover:border-gray-300 transition-shadow shadow-xs space-y-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <SeverityBadge severity={finding.severity} />
                        <span className="text-xs font-semibold uppercase text-gray-400 tracking-wider">
                          {finding.category}
                        </span>
                      </div>
                      <h4 className="text-sm font-semibold text-gray-900 mt-1">{finding.title}</h4>
                    </div>
                    {finding.validator_name && (
                      <span className="text-[11px] text-gray-400 bg-gray-50 px-2 py-0.5 rounded border border-gray-100">
                        {finding.validator_name} v{finding.validator_version}
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-gray-600 leading-relaxed">{finding.description}</p>

                  {(finding.metric_name || finding.threshold) && (
                    <div className="flex flex-wrap gap-4 text-xs bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                      {finding.metric_name && (
                        <div>
                          <span className="text-gray-500 font-sans">Observed Metric ({finding.metric_name}): </span>
                          <span className="font-bold text-gray-800">{finding.metric_value}</span>
                        </div>
                      )}
                      {finding.threshold && (
                        <div>
                          <span className="text-gray-500 font-sans">Acceptance Threshold: </span>
                          <span className="text-gray-700">{finding.threshold}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {finding.recommendation && (
                    <div className="text-xs text-blue-900 bg-blue-50/80 p-2.5 rounded border border-blue-100/70">
                      <span className="font-semibold text-blue-800">Recommendation: </span>
                      {finding.recommendation}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
