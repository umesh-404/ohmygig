import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, Map, ShieldAlert, BadgeIndianRupee } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import RegionalMatrix from './pages/RegionalMatrix';
import FraudDetection from './pages/FraudDetection';
import PayoutLog from './pages/PayoutLog';

function Layout() {
  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="brand">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#paint0_linear)" />
            <path d="M2 17L12 22L22 17" stroke="url(#paint1_linear)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M2 12L12 17L22 12" stroke="url(#paint2_linear)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <defs>
              <linearGradient id="paint0_linear" x1="2" y1="7" x2="22" y2="7" gradientUnits="userSpaceOnUse">
                <stop stopColor="#8b5cf6" />
                <stop offset="1" stopColor="#c4b5fd" />
              </linearGradient>
              <linearGradient id="paint1_linear" x1="2" y1="19.5" x2="22" y2="19.5" gradientUnits="userSpaceOnUse">
                <stop stopColor="#8b5cf6" />
                <stop offset="1" stopColor="#10b981" />
              </linearGradient>
              <linearGradient id="paint2_linear" x1="2" y1="14.5" x2="22" y2="14.5" gradientUnits="userSpaceOnUse">
                <stop stopColor="#8b5cf6" />
                <stop offset="1" stopColor="#10b981" />
              </linearGradient>
            </defs>
          </svg>
          <h1>OhMyGig AI</h1>
        </div>
        
        <NavLink to="/" end className={({isActive}) => `nav-item ${isActive ? "active" : ""}`}>
          <LayoutDashboard size={20} /> Dashboard
        </NavLink>
        <NavLink to="/matrix" className={({isActive}) => `nav-item ${isActive ? "active" : ""}`}>
          <Map size={20} /> Regional Matrix
        </NavLink>
        <NavLink to="/fraud" className={({isActive}) => `nav-item ${isActive ? "active" : ""}`}>
          <ShieldAlert size={20} /> Fraud Detection
        </NavLink>
        <NavLink to="/payouts" className={({isActive}) => `nav-item ${isActive ? "active" : ""}`}>
          <BadgeIndianRupee size={20} /> Payout Log
        </NavLink>
      </nav>

      {/* Main Routing Outlet */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="matrix" element={<RegionalMatrix />} />
          <Route path="fraud" element={<FraudDetection />} />
          <Route path="payouts" element={<PayoutLog />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
