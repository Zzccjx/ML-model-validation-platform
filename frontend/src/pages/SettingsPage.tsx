import React from 'react';
import { Settings } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
      <div className="bg-gray-100 p-4 rounded-full">
        <Settings className="w-12 h-12 text-gray-500" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
      <p className="text-gray-500 max-w-md">
        Global platform settings and user preferences will go here.
      </p>
    </div>
  );
}
