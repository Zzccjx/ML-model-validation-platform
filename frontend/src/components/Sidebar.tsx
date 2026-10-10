import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, FolderKanban, Cpu, Database, Play,
  AlertTriangle, FileText, Settings, Shield
} from 'lucide-react';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/projects', label: 'Projects', icon: FolderKanban },
  { to: '/models', label: 'Models', icon: Cpu },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/validation-runs', label: 'Validation Runs', icon: Play },
  { to: '/findings', label: 'Findings', icon: AlertTriangle },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-900 text-gray-300 flex flex-col">
      <div className="p-5 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Shield className="w-7 h-7 text-blue-400" />
          <div>
            <h2 className="text-white font-bold text-sm">Model Validation</h2>
            <p className="text-xs text-gray-500">Assurance Platform</p>
          </div>
        </div>
      </div>
      <nav className="flex-1 py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white border-r-2 border-blue-400'
                  : 'hover:bg-gray-800 hover:text-white'
              }`
            }
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
        v0.1.0 — Phase 1
      </div>
    </aside>
  );
}
