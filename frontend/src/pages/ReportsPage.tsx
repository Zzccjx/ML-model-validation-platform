import React from 'react';
import { FileText } from 'lucide-react';

export default function ReportsPage() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
      <div className="bg-blue-50 p-4 rounded-full">
        <FileText className="w-12 h-12 text-blue-500" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900">Reports</h2>
      <p className="text-gray-500 max-w-md">
        Report generation and automated exports will be available in a future release. 
        Stay tuned!
      </p>
    </div>
  );
}
