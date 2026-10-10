import React, { useState, useEffect } from 'react';
import api from '../services/api';
import DataTable from '../components/DataTable';
import SeverityBadge from '../components/SeverityBadge';
import { Finding } from '../types';

export default function FindingsPage() {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFindings();
  }, []);

  const fetchFindings = async () => {
    try {
      const { data } = await api.get<{ findings: Finding[], total: number }>('/validation/findings');
      setFindings(data.findings || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">All Findings</h1>
        <p className="text-gray-500">Cross-project validation findings and vulnerabilities</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {loading ? (
          <div className="p-12 text-center text-gray-500">Loading...</div>
        ) : (
          <DataTable
            columns={[
              { key: 'severity', header: 'Severity', render: (f) => <SeverityBadge severity={f.severity} /> },
              { key: 'category', header: 'Category' },
              { key: 'title', header: 'Title' },
              { key: 'description', header: 'Description', render: (f) => (
                <div className="truncate max-w-md" title={f.description}>
                  {f.description}
                </div>
              )},
              { key: 'created_at', header: 'Date', render: (f) => new Date(f.created_at).toLocaleDateString() }
            ]}
            data={findings}
            emptyMessage="No findings found."
          />
        )}
      </div>
    </div>
  );
}
