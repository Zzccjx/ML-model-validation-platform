import React, { useState, useEffect } from 'react';
import api from '../services/api';
import DataTable from '../components/DataTable';
import FileUpload from '../components/FileUpload';
import { Dataset, Project } from '../types';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedProject, setSelectedProject] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchDatasets();
    fetchProjects();
  }, []);

  const fetchDatasets = async () => {
    try {
      const { data } = await api.get<Dataset[]>('/datasets');
      setDatasets(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchProjects = async () => {
    try {
      const { data } = await api.get<Project[]>('/projects');
      setProjects(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!selectedProject || !targetColumn) {
      alert('Please select a project and provide a target column');
      return;
    }
    
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name.replace(/\.[^/.]+$/, ''));
    formData.append('project_id', selectedProject);
    formData.append('target_column', targetColumn);

    try {
      await api.post('/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setShowUpload(false);
      fetchDatasets();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-500">Manage uploaded datasets for validation</p>
        </div>
        <button
          onClick={() => setShowUpload(!showUpload)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700"
        >
          {showUpload ? 'Cancel' : 'Upload Dataset'}
        </button>
      </div>

      {showUpload && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
          <h3 className="text-lg font-medium mb-4">Upload New Dataset</h3>
          <div className="max-w-xl space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Project</label>
              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md border"
              >
                <option value="">Select a project...</option>
                {projects.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Target Column</label>
              <input
                type="text"
                placeholder="e.g. churn, price, label"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
            
            {selectedProject && targetColumn && (
              <div className="mt-4">
                {uploading ? (
                  <div className="text-center py-8">Uploading...</div>
                ) : (
                  <FileUpload 
                    accept=".csv,.parquet"
                    onFileSelect={handleFileUpload}
                    label="Drag and drop dataset file here"
                    description="Supports .csv, .parquet"
                  />
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {loading ? (
          <div className="p-12 text-center text-gray-500">Loading...</div>
        ) : (
          <DataTable
            columns={[
              { key: 'name', header: 'Name' },
              { key: 'row_count', header: 'Rows', render: (d) => d.row_count || '-' },
              { key: 'column_count', header: 'Columns', render: (d) => d.column_count || '-' },
              { key: 'target_column', header: 'Target' },
              { key: 'uploaded_at', header: 'Uploaded', render: (d) => new Date(d.uploaded_at).toLocaleDateString() }
            ]}
            data={datasets}
            emptyMessage="No datasets found. Upload one to get started."
          />
        )}
      </div>
    </div>
  );
}
