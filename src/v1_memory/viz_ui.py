"""Self-contained zero-build interactive 3D RAG Vector Explorer HTML/CSS/JS UI renderer."""

from __future__ import annotations

import json
from typing import Any


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v1 // 3D RAG Vector Explorer</title>
<link rel="icon" href="data:,">
<style>
  :root {
    --bg: #070b12;
    --bg-panel: rgba(10, 15, 26, 0.84);
    --bg-panel-solid: #0d1424;
    --bg-card: rgba(15, 23, 42, 0.72);
    --bg-card-hover: rgba(30, 41, 59, 0.84);
    --border: rgba(148, 163, 184, 0.18);
    --border-bright: rgba(34, 211, 238, 0.48);
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --text-muted: #64748b;
    --accent: #22d3ee;
    --accent-soft: rgba(34, 211, 238, 0.14);
    --gold: #fbbf24;
    --gold-soft: rgba(251, 191, 36, 0.16);
    --emerald: #10b981;
    --rose: #fb7185;
    --ease-spring: cubic-bezier(0.23, 1, 0.32, 1);
    --hud-offset: 72px;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }

  * {
    box-sizing: border-box;
  }

  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    -webkit-font-smoothing: antialiased;
    user-select: none;
  }

  #scene-canvas {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    display: block;
    touch-action: none;
    cursor: grab;
    background: radial-gradient(ellipse at 50% 48%, #0e1729 0%, #070b12 68%, #04070c 100%);
  }

  button:focus-visible,
  select:focus-visible,
  input:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  #scene-canvas.is-dragging {
    cursor: grabbing;
  }

  #scene-canvas.is-hovering {
    cursor: pointer;
  }

  .hud-panel {
    background: var(--bg-panel);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.56), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  }

  .mono {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
  }

  /* Top Query HUD Bar */
  #top-hud {
    position: fixed;
    top: 12px;
    left: 12px;
    right: 12px;
    z-index: 30;
    padding: 8px 12px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 10px;
  }

  .brand-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-right: 10px;
    border-right: 1px solid var(--border);
  }

  .brand-title {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.08em;
    color: var(--text);
    white-space: nowrap;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 7px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    border: 1px solid rgba(16, 185, 129, 0.35);
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    white-space: nowrap;
  }

  .status-pill.is-offline {
    border-color: rgba(245, 158, 11, 0.35);
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 8px currentColor;
  }

  #query-form {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1 1 320px;
    min-width: 240px;
    max-width: 600px;
  }

  .query-input-wrap {
    position: relative;
    flex: 1;
    display: flex;
    align-items: center;
  }

  #query-input {
    width: 100%;
    height: 32px;
    padding: 0 28px 0 11px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: rgba(7, 11, 18, 0.78);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }

  #query-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.16);
  }

  #query-input::placeholder {
    color: var(--text-muted);
  }

  #btn-clear-query {
    position: absolute;
    right: 6px;
    width: 20px;
    height: 20px;
    border: none;
    border-radius: 5px;
    background: rgba(148, 163, 184, 0.16);
    color: var(--text-dim);
    cursor: pointer;
    display: none;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    line-height: 1;
    padding: 0;
  }

  #btn-clear-query.is-visible {
    display: inline-flex;
  }

  #btn-clear-query:hover {
    background: rgba(251, 113, 133, 0.22);
    color: #fda4af;
  }

  .hud-btn {
    height: 32px;
    padding: 0 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.75);
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    transition: transform 0.14s var(--ease-spring), border-color 0.16s var(--ease-spring), background-color 0.16s var(--ease-spring), color 0.16s var(--ease-spring);
  }

  .hud-btn:active,
  .icon-btn:active,
  .tab-btn:active,
  .legend-pill:active {
    transform: scale(0.97);
  }

  .hud-btn:hover {
    border-color: rgba(148, 163, 184, 0.4);
    color: var(--text);
    background: rgba(30, 41, 59, 0.85);
  }

  .hud-btn.is-primary {
    border-color: rgba(34, 211, 238, 0.55);
    background: rgba(34, 211, 238, 0.16);
    color: #67e8f9;
    font-weight: 600;
  }

  .hud-btn.is-primary:hover {
    background: rgba(34, 211, 238, 0.25);
    color: #ecfeff;
  }

  .hud-btn.is-active {
    border-color: rgba(34, 211, 238, 0.48);
    background: rgba(34, 211, 238, 0.14);
    color: #cffafe;
  }

  .hud-btn.is-hybrid {
    border-color: rgba(167, 139, 250, 0.5);
    background: rgba(167, 139, 250, 0.16);
    color: #ddd6fe;
  }

  .hud-btn.is-sparse {
    border-color: rgba(245, 158, 11, 0.5);
    background: rgba(245, 158, 11, 0.14);
    color: #fde68a;
  }

  .hud-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .slider-box {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.65);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  #threshold-slider {
    width: 82px;
    accent-color: var(--accent);
    cursor: pointer;
  }

  #threshold-val {
    min-width: 34px;
    text-align: right;
    color: var(--accent);
    font-weight: 600;
  }

  #color-by-select {
    height: 32px;
    padding: 0 8px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.88);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    outline: none;
    cursor: pointer;
  }

  #query-elapsed {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    min-width: 48px;
    text-align: right;
  }

  /* Left Cluster Legend & Stats Panel */
  #left-panel {
    position: fixed;
    top: var(--hud-offset);
    left: 12px;
    width: 276px;
    max-width: calc(100vw - 24px);
    max-height: calc(100vh - var(--hud-offset) - 52px);
    z-index: 20;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: width 0.2s var(--ease-spring), top 0.16s var(--ease-spring);
  }

  #left-panel.is-collapsed {
    width: auto;
  }

  #left-panel.is-collapsed .panel-body {
    display: none;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }

  #left-panel.is-collapsed .panel-header {
    border-bottom: none;
    padding: 7px 10px;
  }

  .panel-title {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    text-transform: uppercase;
  }

  .icon-btn {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.7);
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 13px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
  }

  .icon-btn:hover {
    color: var(--text);
    border-color: rgba(148, 163, 184, 0.45);
  }

  .panel-body {
    padding: 10px 12px 12px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .panel-body::-webkit-scrollbar,
  .drawer-scroll::-webkit-scrollbar {
    width: 6px;
  }

  .panel-body::-webkit-scrollbar-thumb,
  .drawer-scroll::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.22);
    border-radius: 999px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }

  .stat-cell {
    padding: 7px 8px;
    border-radius: 8px;
    background: var(--bg-card);
    border: 1px solid rgba(148, 163, 184, 0.12);
  }

  .stat-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .stat-value {
    margin-top: 2px;
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
  }

  .section-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .pca-bars {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 8px 9px;
    border-radius: 8px;
    background: var(--bg-card);
    border: 1px solid rgba(148, 163, 184, 0.12);
  }

  .pca-row {
    display: grid;
    grid-template-columns: 34px 1fr 44px;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: 11px;
  }

  .pca-track {
    height: 6px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.12);
    overflow: hidden;
  }

  .pca-fill {
    height: 100%;
    border-radius: 999px;
  }

  .legend-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .legend-pill {
    width: 100%;
    padding: 6px 9px;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: var(--bg-card);
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    text-align: left;
    transition: all 0.14s ease;
  }

  .legend-pill:hover {
    background: var(--bg-card-hover);
    border-color: rgba(148, 163, 184, 0.34);
  }

  .legend-pill.is-selected {
    border-color: var(--accent);
    background: rgba(34, 211, 238, 0.14);
    box-shadow: inset 0 0 0 1px rgba(34, 211, 238, 0.25);
  }

  .legend-pill.is-dimmed {
    opacity: 0.46;
  }

  .legend-left {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
  }

  .swatch-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .legend-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .legend-count {
    color: var(--text-dim);
    font-size: 11px;
    flex-shrink: 0;
  }

  /* Right Inspector & Query Drawer */
  #right-drawer {
    position: fixed;
    top: var(--hud-offset);
    right: 12px;
    width: 396px;
    max-width: calc(100vw - 24px);
    max-height: calc(100vh - var(--hud-offset) - 14px);
    z-index: 25;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transform: translateX(calc(100% + 24px));
    opacity: 0;
    pointer-events: none;
    transition: transform 0.22s var(--ease-spring), opacity 0.18s var(--ease-spring), top 0.16s var(--ease-spring);
  }

  #right-drawer.is-open {
    transform: translateX(0);
    opacity: 1;
    pointer-events: auto;
  }

  .drawer-tabs {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .tab-btn {
    padding: 4px 9px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }

  .tab-btn:hover {
    color: var(--text);
  }

  .tab-btn.is-active {
    border-color: rgba(34, 211, 238, 0.42);
    background: rgba(34, 211, 238, 0.12);
    color: #cffafe;
  }

  .drawer-scroll {
    padding: 12px 14px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 13px;
    user-select: text;
  }

  .detail-title {
    margin: 0;
    font-size: 15px;
    font-weight: 650;
    line-height: 1.35;
    color: #f8fafc;
    word-break: break-word;
  }

  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.24);
    background: rgba(15, 23, 42, 0.85);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  .badge-hybrid {
    border-color: rgba(167, 139, 250, 0.45);
    background: rgba(167, 139, 250, 0.15);
    color: #ddd6fe;
  }

  .badge-sparse {
    border-color: rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.14);
    color: #fde68a;
  }

  .summary-card {
    padding: 9px 11px;
    border-radius: 8px;
    background: var(--bg-card);
    border: 1px solid rgba(148, 163, 184, 0.14);
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .pill-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }

  .token-pill {
    padding: 3px 7px;
    border-radius: 6px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    background: rgba(15, 23, 42, 0.78);
    color: #cbd5e1;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.12s ease;
  }

  .token-pill:hover {
    border-color: var(--accent);
    color: #ecfeff;
    background: rgba(34, 211, 238, 0.16);
  }

  .token-pill.is-concept {
    border-color: rgba(167, 139, 250, 0.42);
    background: rgba(167, 139, 250, 0.14);
    color: #e9d5ff;
  }

  .fingerprint-card {
    padding: 9px 10px 8px;
    border-radius: 8px;
    background: rgba(7, 11, 18, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.16);
  }

  .fingerprint-canvas {
    width: 100%;
    height: 76px;
    display: block;
  }

  .fingerprint-footer {
    margin-top: 4px;
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
  }

  .neighbor-list, .match-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .neighbor-item, .match-card {
    width: 100%;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: var(--bg-card);
    color: var(--text);
    text-align: left;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 5px;
    transition: all 0.14s ease;
  }

  .neighbor-item:hover, .match-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(34, 211, 238, 0.45);
  }

  .match-card.is-selected {
    border-color: var(--gold);
    background: rgba(251, 191, 36, 0.1);
  }

  .neighbor-top, .match-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .neighbor-title, .match-title {
    font-size: 12px;
    font-weight: 600;
    color: #f1f5f9;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .score-pill {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(34, 211, 238, 0.14);
    color: #67e8f9;
    flex-shrink: 0;
  }

  .score-pill.is-gold {
    background: rgba(251, 191, 36, 0.18);
    color: #fde68a;
  }

  .cos-track {
    height: 4px;
    width: 100%;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.14);
    overflow: hidden;
  }

  .cos-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22d3ee, #38bdf8);
  }

  .match-metrics {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  .cmd-box {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 9px;
    border-radius: 8px;
    background: rgba(7, 11, 18, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.22);
  }

  .cmd-text {
    font-family: var(--mono);
    font-size: 11px;
    color: #7dd3fc;
    overflow-x: auto;
    white-space: nowrap;
    flex: 1;
  }

  /* Tooltip & Footer Gimbal */
  #hover-tooltip {
    position: fixed;
    pointer-events: none;
    z-index: 50;
    max-width: 300px;
    padding: 8px 10px;
    border-radius: 9px;
    background: rgba(7, 11, 18, 0.94);
    border: 1px solid rgba(148, 163, 184, 0.3);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.65);
    display: none;
  }

  .tt-title {
    font-weight: 650;
    font-size: 12px;
    color: #f8fafc;
    margin-bottom: 5px;
    line-height: 1.3;
    word-break: break-word;
  }

  .tt-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
  }

  #bottom-hint {
    position: fixed;
    left: 12px;
    bottom: 12px;
    z-index: 15;
    padding: 6px 10px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 10px;
    pointer-events: none;
  }

  #empty-banner {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 18px 22px;
    text-align: center;
    max-width: 420px;
    z-index: 18;
    display: none;
  }

  #error-banner {
    position: fixed;
    top: var(--hud-offset);
    left: 50%;
    transform: translateX(-50%);
    max-width: calc(100vw - 24px);
    padding: 8px 14px;
    border-color: rgba(251, 113, 133, 0.5);
    background: rgba(69, 10, 24, 0.88);
    color: #fecdd3;
    font-family: var(--mono);
    font-size: 11px;
    z-index: 45;
    display: none;
  }

  @media (max-width: 1080px) {
    #left-panel {
      width: 244px;
    }
    #right-drawer {
      width: 356px;
    }
  }

  @media (max-width: 640px) {
    #top-hud {
      padding: 8px 10px;
      gap: 6px 8px;
    }
    .brand-box {
      width: 100%;
      border-right: none;
      padding-right: 0;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--border);
      justify-content: space-between;
    }
    #query-form {
      flex: 1 1 100%;
      min-width: 0;
      max-width: none;
    }
    .hud-btn,
    #color-by-select,
    .slider-box {
      height: 36px;
      font-size: 11px;
    }
    .icon-btn {
      width: 30px;
      height: 30px;
    }
    #left-panel {
      width: min(296px, calc(100vw - 24px));
      max-height: calc(100vh - var(--hud-offset) - 48px);
    }
    #right-drawer {
      width: calc(100vw - 24px);
      max-height: calc(100vh - var(--hud-offset) - 16px);
    }
    #bottom-hint {
      max-width: calc(100vw - 24px);
      font-size: 9px;
      padding: 5px 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      transition-duration: 0.01ms !important;
      animation-duration: 0.01ms !important;
    }
  }
</style>
</head>
<body>
  <canvas id="scene-canvas"></canvas>

  <header id="top-hud" class="hud-panel">
    <div class="brand-box">
      <span class="brand-title">v1 // 3D VECTOR SPACE</span>
      <span id="live-status-pill" class="status-pill"><span class="status-dot"></span><span id="live-status-text">LIVE</span></span>
    </div>

    <form id="query-form" autocomplete="off">
      <div class="query-input-wrap">
        <input id="query-input" type="search" placeholder="Probe 256D semantic space (press /)..." aria-label="Query RAG vector space">
        <button id="btn-clear-query" type="button" title="Clear query overlay">×</button>
      </div>
      <button id="btn-probe" type="submit" class="hud-btn is-primary">Probe</button>
      <span id="query-elapsed">-- ms</span>
    </form>

    <div class="hud-group">
      <button id="btn-rag-mode" type="button" class="hud-btn is-hybrid" aria-pressed="true" title="Toggle Hybrid-RAG (FTS5 + 256D cosine + RRF) vs Sparse-Only">Hybrid-RAG</button>
      <label class="slider-box" title="Match threshold %">
        <span>Thresh</span>
        <input id="threshold-slider" type="range" min="0" max="100" step="1" value="40">
        <span id="threshold-val">40%</span>
      </label>
      <select id="color-by-select" aria-label="Color nodes by">
        <option value="cluster">Color: Cluster</option>
        <option value="provider">Color: Provider</option>
      </select>
      <button id="btn-knn" type="button" class="hud-btn is-active" aria-pressed="true">k-NN Edges</button>
      <button id="btn-centroids" type="button" class="hud-btn is-active" aria-pressed="true">Centroids</button>
      <button id="btn-labels" type="button" class="hud-btn" aria-pressed="false">Labels</button>
      <button id="btn-orbit" type="button" class="hud-btn" aria-pressed="false">Auto-Orbit</button>
      <button id="btn-reset-cam" type="button" class="hud-btn" title="Reset camera (R or double-click space)">Reset Camera</button>
    </div>
  </header>

  <aside id="left-panel" class="hud-panel">
    <div class="panel-header">
      <span class="panel-title">Corpus Telemetry</span>
      <button id="btn-collapse-left" type="button" class="icon-btn" title="Collapse or expand panel">«</button>
    </div>
    <div class="panel-body">
      <div class="stats-grid">
        <div class="stat-cell">
          <div class="stat-label">Docs</div>
          <div id="stat-docs" class="stat-value">0</div>
        </div>
        <div class="stat-cell">
          <div class="stat-label">256D k-NN</div>
          <div id="stat-edges" class="stat-value">0</div>
        </div>
        <div class="stat-cell">
          <div class="stat-label">Clusters</div>
          <div id="stat-clusters" class="stat-value">0</div>
        </div>
      </div>

      <div>
        <div class="section-label">
          <span>PCA Variance Explained</span>
          <span id="pca-total" class="mono">0.0%</span>
        </div>
        <div id="pca-bars" class="pca-bars"></div>
      </div>

      <div>
        <div class="section-label">
          <span id="legend-heading">Cluster Filter</span>
          <button id="btn-clear-filter" type="button" class="tab-btn" style="padding:1px 6px;display:none;">Show All</button>
        </div>
        <div id="legend-list" class="legend-list"></div>
      </div>
    </div>
  </aside>

  <aside id="right-drawer" class="hud-panel" aria-live="polite">
    <div class="panel-header">
      <div class="drawer-tabs">
        <button id="tab-inspector" type="button" class="tab-btn is-active">Node Inspector</button>
        <button id="tab-query" type="button" class="tab-btn">Query Results (0)</button>
      </div>
      <button id="btn-close-drawer" type="button" class="icon-btn" title="Close drawer">×</button>
    </div>
    <div id="drawer-content" class="drawer-scroll"></div>
  </aside>

  <div id="hover-tooltip"></div>
  <div id="empty-banner" class="hud-panel">
    <div class="mono" style="font-size:13px;font-weight:700;margin-bottom:6px;">No Indexed Conversations Found</div>
    <div style="color:var(--text-dim);font-size:12px;line-height:1.5;">Run <code>v1-context --reindex</code> or check your provider archives, then reload the explorer.</div>
  </div>
  <div id="error-banner" class="hud-panel"></div>

  <div id="bottom-hint" class="hud-panel">
    <span>Left-Drag: Orbit</span>
    <span>Shift/Right-Drag: Pan</span>
    <span>Wheel: Zoom</span>
    <span>Click: Inspect</span>
    <span>DblClick / R: Reset</span>
  </div>

  <script id="initial-graph-data" type="application/json">__INITIAL_GRAPH_JSON__</script>
  <script>
  (function() {
    "use strict";

    const INITIAL_LIVE_API = __LIVE_API_BOOL__;

    const CLUSTER_COLORS = {
      rf_telemetry: "#22d3ee",
      mcu_embedded: "#f59e0b",
      uav_mavlink: "#10b981",
      rag_search: "#a78bfa",
      k8s_containers: "#38bdf8",
      auth_security: "#fb7185",
      database_storage: "#818cf8",
      cloud_backup_sync: "#2dd4bf",
      general: "#94a3b8"
    };

    const EXTRA_COLORS = ["#f472b6", "#a3e635", "#fb923c", "#60a5fa", "#e879f9", "#34d399"];

    const PROVIDER_COLORS = {
      antigravity: "#22d3ee",
      claude: "#f59e0b",
      codex: "#10b981",
      generic: "#a78bfa"
    };

    function hashString(str) {
      let h = 2166136261;
      for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return h >>> 0;
    }

    function stripConceptPrefix(name) {
      const s = String(name || "general").trim();
      return s.startsWith("concept:") ? s.slice(8) : s;
    }

    function clusterColor(name) {
      const key = stripConceptPrefix(name).toLowerCase();
      if (CLUSTER_COLORS[key]) return CLUSTER_COLORS[key];
      return EXTRA_COLORS[hashString(key) % EXTRA_COLORS.length];
    }

    function providerColor(name) {
      const key = String(name || "generic").toLowerCase();
      return PROVIDER_COLORS[key] || "#94a3b8";
    }

    function hexToRgb(hex) {
      const clean = String(hex || "#94a3b8").replace("#", "");
      if (clean.length !== 6) return [148, 163, 184];
      return [
        parseInt(clean.slice(0, 2), 16),
        parseInt(clean.slice(2, 4), 16),
        parseInt(clean.slice(4, 6), 16)
      ];
    }

    function rgba(hex, alpha) {
      const rgb = hexToRgb(hex);
      return "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + alpha + ")";
    }

    function safePos(pos) {
      if (!Array.isArray(pos) || pos.length < 3) return [0, 0, 0];
      const x = Number(pos[0]), y = Number(pos[1]), z = Number(pos[2]);
      return [
        Number.isFinite(x) ? x : 0,
        Number.isFinite(y) ? y : 0,
        Number.isFinite(z) ? z : 0
      ];
    }

    function parseGraph() {
      const el = document.getElementById("initial-graph-data");
      try {
        const parsed = JSON.parse((el && el.textContent) ? el.textContent : "{}");
        return parsed && typeof parsed === "object" ? parsed : {};
      } catch (err) {
        showError("Failed to parse embedded graph JSON: " + String(err));
        return {};
      }
    }

    function showError(msg) {
      const banner = document.getElementById("error-banner");
      if (!banner) return;
      if (!msg) {
        banner.style.display = "none";
        banner.textContent = "";
        return;
      }
      banner.textContent = msg;
      banner.style.display = "block";
    }

    const GRAPH = parseGraph();
    const IS_LIVE_API = Boolean(INITIAL_LIVE_API && GRAPH.live_api !== false);
    const RAW_NODES = Array.isArray(GRAPH.nodes) ? GRAPH.nodes : [];
    const RAW_CENTROIDS = Array.isArray(GRAPH.centroids) ? GRAPH.centroids : [];
    const NODES = RAW_NODES.map(function(n, idx) {
      const dom = stripConceptPrefix(n.dominant_concept || "general");
      const pos = safePos(n.pos);
      return {
        id: String(n.id || ("node-" + idx)),
        provider: String(n.provider || "generic").toLowerCase(),
        title: String(n.title || n.id || "Untitled Conversation"),
        last_modified_time: String(n.last_modified_time || ""),
        step_count: Math.max(1, Number(n.step_count) || 1),
        summary: String(n.summary || ""),
        dominant_concept: dom,
        concepts: n.concepts && typeof n.concepts === "object" ? n.concepts : {},
        top_tokens: Array.isArray(n.top_tokens) ? n.top_tokens.map(String) : [],
        vector_preview: Array.isArray(n.vector_preview) ? n.vector_preview.map(Number) : [],
        pos: pos,
        knn: Array.isArray(n.knn) ? n.knn : []
      };
    });

    const NODE_BY_ID = new Map();
    NODES.forEach(function(n) { NODE_BY_ID.set(n.id, n); });

    const CENTROIDS = RAW_CENTROIDS.map(function(c) {
      return {
        concept: stripConceptPrefix(c.concept || "general"),
        count: Number(c.count) || 0,
        pos: safePos(c.pos)
      };
    });

    const UNIQUE_EDGES = [];
    const edgeSeen = new Set();
    NODES.forEach(function(n) {
      n.knn.forEach(function(edge) {
        if (!edge || !edge.id || !NODE_BY_ID.has(edge.id) || edge.id === n.id) return;
        const a = n.id < edge.id ? n.id : edge.id;
        const b = n.id < edge.id ? edge.id : n.id;
        const key = a + "\u0000" + b;
        if (!edgeSeen.has(key)) {
          edgeSeen.add(key);
          UNIQUE_EDGES.push({ a: n.id, b: edge.id, cos: Number(edge.cos) || 0 });
        }
      });
    });

    const state = {
      colorBy: "cluster",
      showKnn: true,
      showCentroids: true,
      showLabels: false,
      autoOrbit: false,
      hybridRag: true,
      thresholdPct: Math.max(0, Math.min(100, Math.round(Number(GRAPH.default_threshold ?? 40)))),
      activeFilter: null,
      selectedNodeId: null,
      hoveredNodeId: null,
      drawerTab: "inspector",
      drawerOpen: false,
      queryResult: null,
      queryMatchById: new Map(),
      queryNearById: new Map(),
      isProbing: false,
      needsRender: true
    };

    const cam = {
      yaw: 0.68,
      pitch: 0.38,
      zoom: 1.05,
      panX: 0,
      panY: 0,
      targetX: 0,
      targetY: 0,
      targetZ: 0,
      goalTargetX: 0,
      goalTargetY: 0,
      goalTargetZ: 0,
      goalYaw: null,
      goalPitch: null,
      goalZoom: null,
      goalPanX: null,
      goalPanY: null
    };

    // Offline client-side fallback engine (tokenization, 8 concept clusters, CRC32 256D -> 64 preview)
    const OFFLINE_STOPWORDS = new Set([
      "a","about","above","after","again","against","agent","agents","all","almost","already","also",
      "always","am","an","and","another","any","anything","app","apps","are","around","as","ask","at",
      "auto","be","because","been","before","being","below","best","better","between","both","bug","bugs",
      "build","built","but","by","call","can","change","changes","chat","check","checked","code","coding",
      "come","config","context","control","conversation","conversations","could","create","created","current",
      "data","day","days","debug","debugging","default","dev","did","different","dir","do","doc","docs",
      "does","doing","done","down","during","each","edit","else","enough","error","etc","even","every",
      "example","explain","feature","file","files","find","fix","fixed","flag","flags","folder","for",
      "from","further","get","getting","give","given","go","going","good","got","great","had","has","have",
      "having","he","hello","help","her","here","hey","hi","him","his","how","i","idea","if","in",
      "include","info","input","into","is","issue","issues","it","its","job","jobs","just","keep","kind",
      "know","last","least","less","let","like","limit","line","lines","list","little","llm","look","lot",
      "made","make","making","manage","many","match","may","maybe","me","might","mode","model","models",
      "more","most","much","must","my","name","need","needs","never","new","next","no","not","nothing",
      "now","number","of","off","ok","okay","old","on","once","one","only","option","or","other","our",
      "out","output","over","own","part","past","path","permission","please","point","previous","probably",
      "project","prompt","prompts","put","python","question","quick","quite","rather","read","really",
      "recent","related","remember","repo","repos","run","running","said","same","say","script","scripts",
      "see","seems","seen","select","send","set","several","she","should","show","similar","simple","since",
      "skill","skills","so","some","someone","something","sort","start","started","step","steps","still",
      "subagent","such","sure","system","systems","take","talk","tell","terms","test","testing","tests",
      "text","than","thank","thanks","that","the","their","them","then","there","these","they","thing",
      "things","think","this","those","though","thought","three","through","time","times","to","today",
      "together","too","tool","tools","topic","try","trying","two","under","understand","until","up",
      "update","updated","us","use","used","useful","using","user","users","usually","very","want","was",
      "way","ways","we","week","well","were","what","whatever","when","where","whether","which","while",
      "who","whom","why","will","with","within","without","work","worked","working","works","would",
      "write","writing","written","wrote","yeah","yes","yesterday","yet","you","your","yours",
      "don","dont","won","wont","didn","isn","wasn","couldn","wouldn","shouldn","problem","problems",
      "expected","located","concern","concerns","demonstration","demo","verify","verified","verification",
      "ready","complete","completed","exact","directly","inside","across","instead","immediately",
      "automatically","implement","implemented","implementation","status","summary","overview","detail",
      "details","result","results","response","note","notes","via","vs","eg","ie","content","document"
    ]);

    const OFFLINE_CLUSTERS = {
      rf_telemetry: ["rf","radio","lora","lorawan","subghz","sub-ghz","915mhz","915","868mhz","433mhz","mhz","sx1262","sx1276","sx1278","sx1280","telemetry","rssi","snr","dbm","dropout","dropouts","preamble","transceiver","wireless","antenna","modulation","spreading","bandwidth","airtime","crc","fsk","sf7","sf9","sf12","esp-now","espnow","crsf","elrs","downlink","uplink"],
      mcu_embedded: ["mcu","embedded","stm32","stm32h7","stm32f4","stm32g4","esp32","esp32s3","fdcan","bxcan","canbus","prescaler","arbitration","bootloader","firmware","uart","usart","spi","i2c","dma","isr","interrupt","hal","gpio","pwm","watchdog","rtos","freertos","zephyr","microcontroller","peripheral","baud","baudrate","bitrate"],
      uav_mavlink: ["mavlink","uav","drone","px4","ardupilot","autopilot","sitl","gazebo","waypoint","waypoints","gimbal","offboard","attitude","imu","gps","ekf","quadcopter","flight","airspeed","altimeter"],
      rag_search: ["rag","hybrid_rag","rrf","bm25","fts","fts5","vector","embedding","embeddings","semantic","cosine","retrieval","rerank","reranker"],
      k8s_containers: ["kubernetes","k8s","helm","pod","pods","deployment","deployments","liveness","readiness","probe","manifest","manifests","container","containers","docker","dockerfile","compose","ingress","namespace","kubectl","autoscale","hpa"],
      auth_security: ["jwt","oauth","oauth2","oidc","sso","rbac","iam","apikey","api_key","authentication","authorization","auth","credential","credentials","login","secret","secrets","tls","ssl","certificate"],
      database_storage: ["database","db","sql","sqlite","sqlite3","wal","schema","migration","migrations","postgres","postgresql","bigquery","spanner","transaction","transactions","vacuum","duckdb","mysql","nosql","orm"],
      cloud_backup_sync: ["cloud","drive","gdrive","google-drive","rclone","backup","backups","restore","sync","syncing","synchronization","robocopy","storage","bucket","gcs","s3"]
    };

    const NO_STEM = new Set(["rtos","freertos","https","postgres","kubernetes","prometheus","macos","bios","cmos","ddos","nixos","centos","devops","mlops","gitops","keras","pandas"]);

    function stemToken(tok) {
      const t = String(tok || "").toLowerCase().replace(/^[._-]+|[._-]+$/g, "");
      if (t.length <= 3 || NO_STEM.has(t)) return t;
      if (t.endsWith("ies") && t.length > 4) return t.slice(0, -3) + "y";
      if (/(ches|shes|xes|zes|sses)$/.test(t) && t.length > 4) return t.slice(0, -2);
      if (t.endsWith("s") && !/(ss|us|is)$/.test(t)) return t.slice(0, -1);
      return t;
    }

    function tokenizeDomainOffline(text) {
      const cleaned = String(text || "").replace(/(?<=[a-z])['\u2019](?:re|ll|ve)\b/gi, "");
      const matches = cleaned.toLowerCase().match(/[a-z0-9][a-z0-9_.-]{1,}/g) || [];
      const seen = new Set();
      const out = [];
      function pushTok(tok) {
        const t = tok.replace(/^[._-]+|[._-]+$/g, "");
        if (t.length < 2) return;
        if (/^\d{1,2}$/.test(t)) return;
        if (OFFLINE_STOPWORDS.has(t)) return;
        if (!seen.has(t)) {
          seen.add(t);
          out.push(t);
        }
      }
      for (const raw of matches) {
        const c = raw.replace(/^[._-]+|[._-]+$/g, "");
        if (c.length < 2) continue;
        pushTok(c);
        if (c.includes("-") || c.includes(".")) {
          for (const part of c.split(/[.-]+/)) pushTok(part);
        }
        for (const sub of c.split(/[._-]+/)) {
          const re = /([a-z]+)(\d+)|(\d+)([a-z]+)/gi;
          let m;
          while ((m = re.exec(sub)) !== null) {
            const a = m[1] || m[3];
            const b = m[2] || m[4];
            if (a && b && a.length >= 3 && b.length >= 3) {
              pushTok(a);
              pushTok(b);
            }
          }
        }
      }
      return out;
    }

    const CRC_TABLE = (function() {
      const tbl = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        tbl[i] = c >>> 0;
      }
      return tbl;
    })();

    function crc32Utf8(str) {
      const bytes = new TextEncoder().encode(str);
      let crc = 0xFFFFFFFF;
      for (let i = 0; i < bytes.length; i++) {
        crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
      }
      return (crc ^ 0xFFFFFFFF) >>> 0;
    }

    const OFFLINE_TERM_INDEX = (function() {
      const termIndex = new Map();
      for (const cid of Object.keys(OFFLINE_CLUSTERS)) {
        for (const term of OFFLINE_CLUSTERS[cid]) {
          const add = function(k) {
            if (!termIndex.has(k)) termIndex.set(k, new Set());
            termIndex.get(k).add(cid);
          };
          add(term);
          add(stemToken(term));
        }
      }
      return termIndex;
    })();

    function offlineGroupConcepts(tokens) {
      const grouped = {};
      for (const t of tokens) {
        const s = stemToken(t);
        const cids = new Set();
        if (OFFLINE_TERM_INDEX.has(t)) OFFLINE_TERM_INDEX.get(t).forEach(function(c) { cids.add(c); });
        if (OFFLINE_TERM_INDEX.has(s)) OFFLINE_TERM_INDEX.get(s).forEach(function(c) { cids.add(c); });
        cids.forEach(function(cid) {
          const key = "concept:" + cid;
          if (!grouped[key]) grouped[key] = [];
          if (!grouped[key].includes(t)) grouped[key].push(t);
        });
      }
      return grouped;
    }

    function offlineVectorPreview(tokens) {
      const dim = 256;
      const previewLen = 64;
      if (!tokens.length) return new Array(previewLen).fill(0);
      const unique = Array.from(new Set(tokens)).sort().slice(0, 512);
      const conceptHits = offlineGroupConcepts(unique);
      const qualified = Object.keys(conceptHits).filter(function(cid) {
        return new Set(conceptHits[cid].map(stemToken)).size >= 2;
      }).sort();
      const gramSet = new Set();
      for (const tok of unique.slice(0, 256)) {
        const p = "^" + tok + "$";
        if (p.length < 4) gramSet.add(p);
        else {
          for (let i = 0; i <= p.length - 4; i++) gramSet.add(p.slice(i, i + 4));
        }
      }
      const ngrams = Array.from(gramSet).sort();
      const vec = new Float64Array(dim);
      function addHash(feat, w) {
        const h = crc32Utf8(feat);
        const idx = h % dim;
        const sign = (((h >>> 16) & 1) === 0) ? 1 : -1;
        vec[idx] += sign * w;
      }
      const tokScale = 1.0 / Math.sqrt(unique.length);
      for (const tok of unique) {
        addHash("tok:" + tok, tokScale);
        const st = stemToken(tok);
        if (st !== tok) addHash("tok:" + st, 0.5 * tokScale);
      }
      if (qualified.length) {
        const cScale = 1.6 / Math.sqrt(qualified.length * 4);
        for (const cid of qualified) {
          for (let p = 0; p < 4; p++) addHash(cid + "#" + p, cScale);
        }
      }
      if (ngrams.length) {
        const gScale = 0.5 / Math.sqrt(ngrams.length);
        for (const g of ngrams) addHash("g4:" + g, gScale);
      }
      let norm = 0;
      for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
      norm = Math.sqrt(norm);
      if (norm > 1e-12) for (let i = 0; i < dim; i++) vec[i] /= norm;

      const preview = new Array(previewLen).fill(0);
      const step = dim / previewLen;
      for (let b = 0; b < previewLen; b++) {
        let sum = 0, sq = 0;
        const start = Math.floor(b * step);
        const end = Math.floor((b + 1) * step);
        for (let i = start; i < end; i++) {
          sum += vec[i];
          sq += vec[i] * vec[i];
        }
        const rms = Math.sqrt(sq / Math.max(1, end - start));
        preview[b] = Number(((sum >= 0 ? 1 : -1) * rms).toFixed(4));
      }
      return preview;
    }

    function cosineSim(a, b) {
      if (!a || !b || !a.length || !b.length) return 0;
      const n = Math.min(a.length, b.length);
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < n; i++) {
        const va = Number(a[i]) || 0;
        const vb = Number(b[i]) || 0;
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
      }
      if (na <= 1e-12 || nb <= 1e-12) return 0;
      return Math.max(-1, Math.min(1, dot / (Math.sqrt(na) * Math.sqrt(nb))));
    }

    const OFFLINE_DF = (function() {
      const df = new Map();
      NODES.forEach(function(n) {
        const toks = new Set(
          n.top_tokens.concat(tokenizeDomainOffline(n.title + " " + n.summary)).map(stemToken)
        );
        toks.forEach(function(t) {
          df.set(t, (df.get(t) || 0) + 1);
        });
      });
      return df;
    })();

    function runOfflineQuery(queryText, thresholdPct, hybridRag) {
      const t0 = performance.now();
      const qTokens = tokenizeDomainOffline(queryText);
      const qConcepts = offlineGroupConcepts(qTokens);
      const qStems = qTokens.map(stemToken);
      const qPreview = offlineVectorPreview(qTokens);
      const nTotal = Math.max(1, NODES.length);

      function idf(stem) {
        const df = OFFLINE_DF.get(stem) || 0;
        return Math.log(1 + (nTotal - df + 0.5) / (df + 0.5));
      }

      const qIdfSum = qStems.reduce(function(acc, s) { return acc + idf(s); }, 0) || 1;
      const scored = [];

      for (const node of NODES) {
        const docToks = new Set(node.top_tokens.concat(tokenizeDomainOffline(node.title + " " + node.summary)));
        const docStems = new Set(Array.from(docToks).map(stemToken));
        const matched = [];
        let matchedW = 0;

        for (let i = 0; i < qTokens.length; i++) {
          const qt = qTokens[i];
          const qs = qStems[i];
          if (docToks.has(qt) || docStems.has(qs)) {
            matched.push(qt);
            matchedW += idf(qs);
          }
        }

        const exactPct = qTokens.length ? Math.min(100, (matchedW / qIdfSum) * 100) : 0;
        const previewCos = Math.max(0, cosineSim(qPreview, node.vector_preview));

        let conceptBoost = 0;
        const conceptTags = [];
        if (hybridRag) {
          const dConcepts = offlineGroupConcepts(Array.from(docToks));
          for (const cid of Object.keys(qConcepts)) {
            const qHit = qConcepts[cid] || [];
            const dHit = dConcepts[cid] || [];
            const nodeCidRaw = stripConceptPrefix(cid);
            const nodeHasCid = node.dominant_concept === nodeCidRaw || Boolean(node.concepts && (node.concepts[nodeCidRaw] || node.concepts[cid]));
            const qDistinct = new Set(qHit.map(stemToken)).size;
            const dDistinct = new Set(dHit.map(stemToken)).size;
            if ((qDistinct >= 2 && dDistinct >= 2) || (qDistinct >= 2 && nodeHasCid)) {
              conceptBoost = Math.max(conceptBoost, 58 + previewCos * 35);
              conceptTags.push(cid + "(" + (dHit.slice(0, 3).join("/") || nodeCidRaw) + ")");
            }
          }
        }

        const semScore = hybridRag ? Math.max(previewCos, exactPct / 100 * 0.72) : (exactPct / 100);
        const matchPct = hybridRag
          ? Math.min(100, Math.max(exactPct, conceptBoost, previewCos * 88))
          : exactPct;

        scored.push({
          node: node,
          cos: Number(semScore.toFixed(4)),
          exactPct: exactPct,
          matchPct: Number(matchPct.toFixed(1)),
          matchedTokens: matched.concat(conceptTags)
        });
      }

      const sparseRanked = scored.slice().sort(function(a, b) { return b.exactPct - a.exactPct; });
      const denseRanked = scored.slice().sort(function(a, b) { return b.cos - a.cos; });
      const rrfMap = new Map();
      sparseRanked.forEach(function(item, i) {
        rrfMap.set(item.node.id, (rrfMap.get(item.node.id) || 0) + 1 / (60 + i + 1));
      });
      if (hybridRag) {
        denseRanked.forEach(function(item, i) {
          rrfMap.set(item.node.id, (rrfMap.get(item.node.id) || 0) + 1 / (60 + i + 1));
        });
      }

      const matches = scored
        .filter(function(item) {
          if (!qTokens.length) return false;
          if (!hybridRag && item.matchedTokens.length === 0) return false;
          if (hybridRag && item.matchedTokens.length === 0 && item.cos < 0.38) return false;
          return item.matchPct >= thresholdPct;
        })
        .sort(function(a, b) {
          if (b.matchPct !== a.matchPct) return b.matchPct - a.matchPct;
          return (rrfMap.get(b.node.id) || 0) - (rrfMap.get(a.node.id) || 0);
        })
        .slice(0, 8)
        .map(function(item) {
          return {
            conversation_id: item.node.id,
            title: item.node.title,
            match_pct: item.matchPct,
            semantic_score: item.cos,
            rrf_score: Number((rrfMap.get(item.node.id) || 0).toFixed(6)),
            retrieval_mode: hybridRag ? "hybrid_rrf" : "sparse",
            matched_tokens: item.matchedTokens
          };
        });

      const nearestCos = denseRanked.slice(0, 5).map(function(item) {
        return {
          id: item.node.id,
          title: item.node.title,
          cos: item.cos
        };
      });

      // Weighted 3D position estimate from top positive neighbors
      let wx = 0, wy = 0, wz = 0, wsum = 0;
      const posSource = matches.length ? matches : nearestCos;
      for (const item of posSource.slice(0, 5)) {
        const nid = item.conversation_id || item.id;
        const nd = NODE_BY_ID.get(nid);
        if (!nd) continue;
        const rawW = item.match_pct ? Math.pow(item.match_pct / 100, 2) : Math.pow(Math.max(0.05, item.cos || 0), 2);
        wx += nd.pos[0] * rawW;
        wy += nd.pos[1] * rawW;
        wz += nd.pos[2] * rawW;
        wsum += rawW;
      }
      const qPos = wsum > 1e-6 ? [wx / wsum, wy / wsum, wz / wsum] : [0, 0, 0];

      return {
        query: queryText,
        query_tokens: qTokens,
        q_concepts: qConcepts,
        q_pos: qPos.map(function(v) { return Number(v.toFixed(4)); }),
        q_vector_preview: qPreview,
        hybrid_rag: hybridRag,
        threshold_pct: thresholdPct,
        elapsed_ms: Number((performance.now() - t0).toFixed(2)),
        matches: matches,
        nearest_cos: nearestCos
      };
    }

    // 3D Canvas Engine
    const canvas = document.getElementById("scene-canvas");
    const ctx = canvas.getContext("2d");
    const REDUCED_MOTION = Boolean(
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
    let viewW = 1280;
    let viewH = 720;
    let hudOffsetPx = 72;
    let dpr = 1;
    let projectedNodes = [];

    function updateHudLayout() {
      const topHud = document.getElementById("top-hud");
      if (!topHud) return;
      const rect = topHud.getBoundingClientRect();
      hudOffsetPx = Math.max(64, Math.ceil(rect.bottom + 10));
      document.documentElement.style.setProperty("--hud-offset", hudOffsetPx + "px");
      state.needsRender = true;
    }

    function setLeftPanelCollapsed(collapsed) {
      const leftPanel = document.getElementById("left-panel");
      const collapseBtn = document.getElementById("btn-collapse-left");
      if (!leftPanel || !collapseBtn) return;
      leftPanel.classList.toggle("is-collapsed", Boolean(collapsed));
      collapseBtn.textContent = collapsed ? "»" : "«";
      collapseBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }

    function resizeCanvas() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      viewW = window.innerWidth || 1280;
      viewH = window.innerHeight || 720;
      canvas.width = Math.max(1, Math.floor(viewW * dpr));
      canvas.height = Math.max(1, Math.floor(viewH * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      updateHudLayout();
      if (viewW < 860) {
        setLeftPanelCollapsed(true);
      }
      state.needsRender = true;
    }

    function nodeColor(node) {
      return state.colorBy === "provider"
        ? providerColor(node.provider)
        : clusterColor(node.dominant_concept);
    }

    function isNodeHighlightedByFilter(node) {
      if (!state.activeFilter) return true;
      if (state.colorBy === "provider") {
        return node.provider === state.activeFilter;
      }
      if (node.dominant_concept === state.activeFilter) return true;
      if (node.concepts && (node.concepts[state.activeFilter] || node.concepts["concept:" + state.activeFilter])) {
        return true;
      }
      return false;
    }

    function projectPoint(pos) {
      const dx = pos[0] - cam.targetX;
      const dy = pos[1] - cam.targetY;
      const dz = pos[2] - cam.targetZ;
      const cosY = Math.cos(cam.yaw), sinY = Math.sin(cam.yaw);
      const cosP = Math.cos(cam.pitch), sinP = Math.sin(cam.pitch);

      const x1 = dx * cosY - dz * sinY;
      const z1 = dx * sinY + dz * cosY;
      const y2 = dy * cosP - z1 * sinP;
      const z2 = dy * sinP + z1 * cosP;

      const camDist = 3.6;
      const vz = z2 + camDist;
      if (vz <= 0.15) return null;
      const persp = camDist / vz;
      const worldScale = Math.min(viewW, viewH) * 0.34 * cam.zoom;
      return {
        sx: viewW * 0.5 + cam.panX + x1 * worldScale * persp,
        sy: viewH * 0.5 + cam.panY - y2 * worldScale * persp,
        vz: vz,
        persp: persp
      };
    }

    function resetCamera() {
      cam.goalYaw = 0.68;
      cam.goalPitch = 0.38;
      cam.goalZoom = 1.05;
      cam.goalPanX = 0;
      cam.goalPanY = 0;
      cam.goalTargetX = 0;
      cam.goalTargetY = 0;
      cam.goalTargetZ = 0;
      state.needsRender = true;
    }

    function focusOnPoint(pos) {
      cam.goalTargetX = pos[0] * 0.55;
      cam.goalTargetY = pos[1] * 0.55;
      cam.goalTargetZ = pos[2] * 0.55;
      state.needsRender = true;
    }

    function measureRoundLabel(text, fontSpec) {
      ctx.font = fontSpec || '600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      const padX = 7;
      const metrics = ctx.measureText(text);
      return { w: Math.ceil(metrics.width + padX * 2), h: 20 };
    }

    function drawRoundLabel(text, x, y, strokeHex, bgAlpha, textColor, fontSpec) {
      const fontStr = fontSpec || '600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      const dim = measureRoundLabel(text, fontStr);
      const w = dim.w;
      const h = dim.h;
      const rx = Math.round(x - w * 0.5);
      const ry = Math.round(y - h * 0.5);

      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(rx, ry, w, h, 5);
      else ctx.rect(rx, ry, w, h);
      ctx.fillStyle = "rgba(7, 11, 18, " + (bgAlpha ?? 0.88) + ")";
      ctx.fill();
      if (strokeHex) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = strokeHex;
        ctx.stroke();
      }

      ctx.font = fontStr;
      ctx.fillStyle = textColor || "#f1f5f9";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x, y + 0.5);
    }

    function placeDeoverlappedBox(anchorX, anchorY, w, h, placedBoxes, offsets) {
      const marginX = Math.max(w * 0.5 + 8, 12);
      const maxX = Math.max(marginX, viewW - marginX);
      const minY = Math.min(viewH - 28, Math.max(hudOffsetPx + 14, 78));
      const maxY = Math.max(minY, viewH - 22);
      const candList = offsets || [
        [0, 0],
        [0, -24],
        [0, 24],
        [46, -16],
        [-46, -16],
        [64, 14],
        [-64, 14],
        [0, -48],
        [0, 48],
        [86, -34],
        [-86, -34],
        [0, -72],
        [0, 72]
      ];
      let bestX = Math.max(marginX, Math.min(maxX, anchorX));
      let bestY = Math.max(minY, Math.min(maxY, anchorY));
      let bestScore = Infinity;

      for (const off of candList) {
        const cx = Math.max(marginX, Math.min(maxX, anchorX + off[0]));
        const cy = Math.max(minY, Math.min(maxY, anchorY + off[1]));
        let hitArea = 0;
        for (const p of placedBoxes) {
          const ox = Math.max(0, Math.min(cx + w * 0.5, p.x + p.w * 0.5) - Math.max(cx - w * 0.5, p.x - p.w * 0.5) + 6);
          const oy = Math.max(0, Math.min(cy + h * 0.5, p.y + p.h * 0.5) - Math.max(cy - h * 0.5, p.y - p.h * 0.5) + 4);
          if (ox > 0 && oy > 0) hitArea += ox * oy;
        }
        const distPenalty = Math.hypot(cx - anchorX, cy - anchorY) * 0.35;
        const score = hitArea * 100 + distPenalty;
        if (score < bestScore) {
          bestScore = score;
          bestX = cx;
          bestY = cy;
        }
        if (hitArea === 0) break;
      }
      const box = { x: bestX, y: bestY, w: w, h: h };
      placedBoxes.push(box);
      return box;
    }

    function drawStar(x, y, outerR, innerR, points, angleOffset, fill, stroke) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < points * 2; i++) {
        const r = (i % 2 === 0) ? outerR : innerR;
        const a = angleOffset + (i * Math.PI) / points;
        const px = x + Math.cos(a) * r;
        const py = y + Math.sin(a) * r;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      if (stroke) {
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawScene(nowMs) {
      const t = nowMs * 0.001;
      ctx.clearRect(0, 0, viewW, viewH);

      // 1. Subtle 3D floor grid at y = -1.08
      const gridY = -1.08;
      const gridExt = 1.2;
      const gridStep = 0.3;
      ctx.save();
      ctx.lineWidth = 1;
      for (let g = -gridExt; g <= gridExt + 0.001; g += gridStep) {
        const isAxis = Math.abs(g) < 0.01;
        const p1 = projectPoint([g, gridY, -gridExt]);
        const p2 = projectPoint([g, gridY, gridExt]);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.sx, p1.sy);
          ctx.lineTo(p2.sx, p2.sy);
          ctx.strokeStyle = isAxis ? "rgba(34, 211, 238, 0.16)" : "rgba(148, 163, 184, 0.075)";
          ctx.stroke();
        }
        const q1 = projectPoint([-gridExt, gridY, g]);
        const q2 = projectPoint([gridExt, gridY, g]);
        if (q1 && q2) {
          ctx.beginPath();
          ctx.moveTo(q1.sx, q1.sy);
          ctx.lineTo(q2.sx, q2.sy);
          ctx.strokeStyle = isAxis ? "rgba(34, 211, 238, 0.16)" : "rgba(148, 163, 184, 0.075)";
          ctx.stroke();
        }
      }

      // Origin (PC1, PC2, PC3) triad
      const originP = projectPoint([0, 0, 0]);
      if (originP) {
        const axes = [
          { vec: [0.42, 0, 0], color: "rgba(34, 211, 238, 0.55)", label: "PC1" },
          { vec: [0, 0.42, 0], color: "rgba(16, 185, 129, 0.55)", label: "PC2" },
          { vec: [0, 0, 0.42], color: "rgba(167, 139, 250, 0.55)", label: "PC3" }
        ];
        for (const ax of axes) {
          const tip = projectPoint(ax.vec);
          if (!tip) continue;
          ctx.beginPath();
          ctx.moveTo(originP.sx, originP.sy);
          ctx.lineTo(tip.sx, tip.sy);
          ctx.strokeStyle = ax.color;
          ctx.lineWidth = 1.25;
          ctx.stroke();
          ctx.fillStyle = ax.color;
          ctx.font = '600 10px ui-monospace, monospace';
          ctx.fillText(ax.label, tip.sx + 4, tip.sy - 3);
        }
      }
      ctx.restore();

      // 2. Project all nodes
      const activeFocusId = state.hoveredNodeId || state.selectedNodeId;
      const focusNeighborIds = new Set();
      if (activeFocusId && NODE_BY_ID.has(activeFocusId)) {
        const fn = NODE_BY_ID.get(activeFocusId);
        fn.knn.forEach(function(e) {
          if (e && e.id) focusNeighborIds.add(e.id);
        });
      }

      const projMap = new Map();
      const renderItems = [];

      for (const node of NODES) {
        const p = projectPoint(node.pos);
        if (!p) continue;
        const baseR = (4.8 + Math.min(10.5, Math.sqrt(node.step_count) * 1.25)) * p.persp;
        const r = Math.max(3.2, Math.min(26, baseR));
        const item = {
          kind: "node",
          node: node,
          sx: p.sx,
          sy: p.sy,
          vz: p.vz,
          persp: p.persp,
          r: r
        };
        projMap.set(node.id, item);
        renderItems.push(item);
      }

      // 3. Centroid cluster halo rings (depth projected; labels deferred for global de-overlap)
      const centroidLabels = [];
      const activeQuery = Boolean(state.queryResult);
      if (state.showCentroids && state.colorBy === "cluster") {
        for (const c of CENTROIDS) {
          if (state.activeFilter && c.concept !== state.activeFilter) continue;
          const cp = projectPoint(c.pos);
          if (!cp) continue;
          const col = clusterColor(c.concept);
          const haloR = Math.max(18, Math.min(72, (16 + Math.sqrt(Math.max(1, c.count)) * 8.5) * cp.persp * cam.zoom));
          ctx.save();
          ctx.globalAlpha = activeQuery ? 0.52 : 1;
          ctx.beginPath();
          ctx.ellipse(cp.sx, cp.sy, haloR, haloR * 0.56, 0, 0, Math.PI * 2);
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = rgba(col, 0.34);
          ctx.stroke();

          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.arc(cp.sx, cp.sy, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = rgba(col, 0.72);
          ctx.fill();
          ctx.restore();

          if (c.concept) {
            centroidLabels.push({
              text: c.concept + " (" + c.count + ")",
              x: cp.sx,
              y: cp.sy - haloR * 0.56 - 9,
              color: rgba(col, activeQuery ? 0.68 : 0.9),
              count: c.count || 0
            });
          }
        }
      }

      // 4. Soft dashed 256D k-NN edges
      if (state.showKnn || activeFocusId) {
        ctx.save();
        for (const edge of UNIQUE_EDGES) {
          const pa = projMap.get(edge.a);
          const pb = projMap.get(edge.b);
          if (!pa || !pb) continue;
          const isFocusedEdge = activeFocusId && (edge.a === activeFocusId || edge.b === activeFocusId);
          if (!state.showKnn && !isFocusedEdge) continue;

          if (isFocusedEdge) {
            const focusNode = NODE_BY_ID.get(activeFocusId);
            const col = focusNode ? nodeColor(focusNode) : "#22d3ee";
            ctx.beginPath();
            ctx.setLineDash([5, 4]);
            ctx.lineDashOffset = -(t * 18);
            ctx.moveTo(pa.sx, pa.sy);
            ctx.lineTo(pb.sx, pb.sy);
            ctx.lineWidth = 1.75;
            ctx.strokeStyle = rgba(col, 0.85);
            ctx.stroke();

            const mx = (pa.sx + pb.sx) * 0.5;
            const my = (pa.sy + pb.sy) * 0.5;
            ctx.setLineDash([]);
            ctx.font = '600 10px ui-monospace, monospace';
            ctx.fillStyle = "#e2e8f0";
            ctx.textAlign = "center";
            ctx.fillText(edge.cos.toFixed(2), mx, my - 4);
          } else {
            const filterVis = isNodeHighlightedByFilter(pa.node) || isNodeHighlightedByFilter(pb.node);
            ctx.beginPath();
            ctx.setLineDash([3, 4]);
            ctx.lineDashOffset = 0;
            ctx.moveTo(pa.sx, pa.sy);
            ctx.lineTo(pb.sx, pb.sy);
            ctx.lineWidth = 0.9;
            ctx.strokeStyle = filterVis
              ? ("rgba(148, 163, 184, " + Math.max(0.08, Math.min(0.26, edge.cos * 0.30)) + ")")
              : "rgba(148, 163, 184, 0.03)";
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      // 5. Query Star & Animated Glowing Beams to matched docs
      let qProj = null;
      const beamBadges = [];
      if (state.queryResult && Array.isArray(state.queryResult.q_pos)) {
        const qp = projectPoint(safePos(state.queryResult.q_pos));
        if (qp) {
          qProj = {
            kind: "query",
            sx: qp.sx,
            sy: qp.sy,
            vz: qp.vz,
            persp: qp.persp,
            r: Math.max(10, 14 * qp.persp)
          };
          renderItems.push(qProj);

          const matches = Array.isArray(state.queryResult.matches) ? state.queryResult.matches : [];
          ctx.save();
          for (const m of matches) {
            const targetId = m.conversation_id || m.id;
            const targetProj = projMap.get(targetId);
            if (!targetProj) continue;
            const scoreFrac = Math.max(0.25, Math.min(1, (Number(m.match_pct) || 40) / 100));

            // Outer beam glow
            ctx.beginPath();
            ctx.setLineDash([]);
            ctx.moveTo(qp.sx, qp.sy);
            ctx.lineTo(targetProj.sx, targetProj.sy);
            ctx.lineWidth = 4.5 * scoreFrac;
            ctx.strokeStyle = "rgba(251, 191, 36, 0.20)";
            ctx.stroke();

            // Inner animated energy beam
            ctx.beginPath();
            ctx.setLineDash([9, 6]);
            ctx.lineDashOffset = -(t * 36);
            ctx.moveTo(qp.sx, qp.sy);
            ctx.lineTo(targetProj.sx, targetProj.sy);
            ctx.lineWidth = 1.75;
            ctx.strokeStyle = "#fde68a";
            ctx.stroke();

            beamBadges.push({
              text: Math.round(Number(m.match_pct) || 0) + "%",
              x: (qp.sx + targetProj.sx) * 0.5,
              y: (qp.sy + targetProj.sy) * 0.5
            });
          }
          ctx.restore();
        }
      }

      // 6. Depth-sorted painter's algorithm rendering
      renderItems.sort(function(a, b) { return b.vz - a.vz; });

      const queryActive = Boolean(state.queryResult);
      const visibleLabels = [];
      let queryStarLabel = null;
      projectedNodes = [];

      for (const item of renderItems) {
        if (item.kind === "query") {
          const pulse = REDUCED_MOTION ? 0.45 : (0.5 + 0.5 * Math.sin(t * 4.2));
          const haloR = item.r * (2.2 + pulse * 0.9);
          const grad = ctx.createRadialGradient(item.sx, item.sy, 1, item.sx, item.sy, haloR);
          grad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
          grad.addColorStop(0.28, "rgba(251, 191, 36, 0.58)");
          grad.addColorStop(0.65, "rgba(245, 158, 11, 0.16)");
          grad.addColorStop(1, "rgba(245, 158, 11, 0)");
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, haloR, 0, Math.PI * 2);
          ctx.fill();

          const starSpin = REDUCED_MOTION ? 0 : t * 0.4;
          drawStar(item.sx, item.sy, item.r * (0.95 + pulse * 0.12), item.r * 0.36, 4, starSpin, "#ffffff", "#fbbf24");
          queryStarLabel = { x: item.sx, y: item.sy - item.r - 15 };
          continue;
        }

        const n = item.node;
        projectedNodes.push(item);

        const col = nodeColor(n);
        const isSelected = n.id === state.selectedNodeId;
        const isHovered = n.id === state.hoveredNodeId;
        const isNeighbor = focusNeighborIds.has(n.id);
        const matchObj = state.queryMatchById.get(n.id);
        const nearObj = state.queryNearById.get(n.id);
        const filterOk = isNodeHighlightedByFilter(n);

        let alpha = 0.94;
        if (!filterOk) alpha = 0.14;
        if (queryActive && !matchObj && !nearObj && !isSelected && !isHovered) {
          alpha = Math.min(alpha, 0.24);
        }

        ctx.save();
        ctx.globalAlpha = alpha;

        // Soft outer glow for selected / hovered / matched nodes
        if (isSelected || isHovered || matchObj) {
          const glowR = item.r * (isSelected ? 2.5 : 2.1);
          const gCol = matchObj ? "#fbbf24" : col;
          const glow = ctx.createRadialGradient(item.sx, item.sy, item.r * 0.3, item.sx, item.sy, glowR);
          glow.addColorStop(0, rgba(gCol, 0.55));
          glow.addColorStop(1, rgba(gCol, 0));
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, glowR, 0, Math.PI * 2);
          ctx.fill();
        }

        // Radial specular glow sphere
        const specX = item.sx - item.r * 0.32;
        const specY = item.sy - item.r * 0.32;
        const sphereGrad = ctx.createRadialGradient(specX, specY, item.r * 0.08, item.sx, item.sy, item.r);
        sphereGrad.addColorStop(0, "#ffffff");
        sphereGrad.addColorStop(0.28, col);
        sphereGrad.addColorStop(1, rgba(col, 0.32));
        ctx.fillStyle = sphereGrad;
        ctx.beginPath();
        ctx.arc(item.sx, item.sy, item.r, 0, Math.PI * 2);
        ctx.fill();

        // Crisp rim ring
        if (isSelected || isHovered || matchObj || isNeighbor) {
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, item.r + (isSelected ? 2.5 : 1.5), 0, Math.PI * 2);
          ctx.lineWidth = isSelected ? 2.0 : 1.25;
          ctx.strokeStyle = isSelected ? "#ffffff" : (matchObj ? "#fde68a" : col);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, item.r, 0, Math.PI * 2);
          ctx.lineWidth = 0.75;
          ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
          ctx.stroke();
        }

        ctx.restore();

        if (isSelected || isHovered || matchObj || (state.showLabels && filterOk)) {
          visibleLabels.push({
            item: item,
            priority: isSelected ? 4 : (isHovered ? 3 : (matchObj ? 2 : 1)),
            label: n.title.length > 30 ? (n.title.slice(0, 29) + "…") : n.title,
            stroke: matchObj ? "rgba(251, 191, 36, 0.72)" : rgba(col, 0.6)
          });
        }
      }

      // Unified screen-space label placement (de-overlapped across query star, beam badges, node labels, centroids)
      const placedBoxes = [];

      if (queryStarLabel) {
        const qDim = measureRoundLabel("QUERY");
        const qBox = placeDeoverlappedBox(queryStarLabel.x, queryStarLabel.y, qDim.w, qDim.h, placedBoxes, [
          [0, 0], [0, -24], [38, -12], [-38, -12], [0, 28]
        ]);
        drawRoundLabel("QUERY", qBox.x, qBox.y, "rgba(251, 191, 36, 0.88)", 0.94, "#fef3c7");
      }

      // Draw crisp non-crowded node labels (highest priority first; de-overlap ALL priorities!)
      visibleLabels.sort(function(a, b) {
        if (b.priority !== a.priority) return b.priority - a.priority;
        return a.item.vz - b.item.vz;
      });
      let ambientDrawn = 0;
      for (const lbl of visibleLabels) {
        if (lbl.priority === 1 && ambientDrawn >= 24) continue;
        const anchorX = lbl.item.sx;
        const anchorY = lbl.item.sy - lbl.item.r - 15;
        const dim = measureRoundLabel(lbl.label);

        if (lbl.priority === 1) {
          // Ambient label skips if direct slot would collide rather than creating noisy leaders
          let hits = false;
          for (const p of placedBoxes) {
            if (Math.abs(p.x - anchorX) < (p.w + dim.w) * 0.5 + 6 &&
                Math.abs(p.y - anchorY) < (p.h + dim.h) * 0.5 + 4) {
              hits = true;
              break;
            }
          }
          if (hits) continue;
          ambientDrawn += 1;
          placedBoxes.push({ x: anchorX, y: anchorY, w: dim.w, h: dim.h });
          drawRoundLabel(lbl.label, anchorX, anchorY, lbl.stroke, 0.78);
          continue;
        }

        const box = placeDeoverlappedBox(anchorX, anchorY, dim.w, dim.h, placedBoxes, [
          [0, 0],
          [0, -26],
          [0, (lbl.item.r + 16) * 2],
          [56, -18],
          [-56, -18],
          [78, 12],
          [-78, 12],
          [0, -52],
          [0, 56],
          [110, -28],
          [-110, -28],
          [0, -78],
          [0, 82]
        ]);
        const shiftDist = Math.hypot(box.x - lbl.item.sx, box.y - lbl.item.sy);
        if (shiftDist > lbl.item.r + 22) {
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([2, 3]);
          ctx.moveTo(lbl.item.sx, lbl.item.sy);
          ctx.lineTo(box.x, box.y);
          ctx.lineWidth = 1;
          ctx.strokeStyle = lbl.stroke;
          ctx.stroke();
          ctx.restore();
        }
        drawRoundLabel(lbl.label, box.x, box.y, lbl.stroke, 0.92);
      }

      // Draw beam match_pct% badges clearly without stacking onto node labels
      for (const bb of beamBadges) {
        const bDim = measureRoundLabel(bb.text);
        const bBox = placeDeoverlappedBox(bb.x, bb.y, bDim.w, bDim.h, placedBoxes, [
          [0, 0], [0, -22], [0, 22], [30, 0], [-30, 0], [34, -22], [-34, 22]
        ]);
        drawRoundLabel(bb.text, bBox.x, bBox.y, "rgba(251, 191, 36, 0.75)", 0.92, "#fde68a");
      }

      // Draw centroid cluster badges with clean vertical separation
      centroidLabels.sort(function(a, b) { return b.count - a.count; });
      const centroidFont = '600 10px ui-monospace, monospace';
      for (const cl of centroidLabels) {
        const cDim = measureRoundLabel(cl.text, centroidFont);
        const cBox = placeDeoverlappedBox(cl.x, cl.y, cDim.w, cDim.h, placedBoxes, [
          [0, 0], [0, -22], [0, 22], [52, 0], [-52, 0], [0, -44], [0, 44]
        ]);
        drawRoundLabel(cl.text, cBox.x, cBox.y, null, 0.58, cl.color, centroidFont);
      }

      // 7. Bottom-right mini (PC1, PC2, PC3) orientation gimbal
      drawOrientationGimbal();
    }

    function drawOrientationGimbal() {
      if (state.drawerOpen && viewW <= 820) return;
      const drawerClear = (state.drawerOpen && viewW > 820) ? (viewW <= 1080 ? 372 : 412) : 0;
      const cx = viewW - 46 - drawerClear;
      const cy = viewH - 46;
      const len = 24;
      const cosY = Math.cos(cam.yaw), sinY = Math.sin(cam.yaw);
      const cosP = Math.cos(cam.pitch), sinP = Math.sin(cam.pitch);

      function rotAxis(v) {
        const x1 = v[0] * cosY - v[2] * sinY;
        const z1 = v[0] * sinY + v[2] * cosY;
        const y2 = v[1] * cosP - z1 * sinP;
        const z2 = v[1] * sinP + z1 * cosP;
        return { x: x1, y: -y2, z: z2 };
      }

      const axes = [
        { vec: [1, 0, 0], color: "#22d3ee", name: "PC1" },
        { vec: [0, 1, 0], color: "#10b981", name: "PC2" },
        { vec: [0, 0, 1], color: "#a78bfa", name: "PC3" }
      ].map(function(a) {
        return { p: rotAxis(a.vec), color: a.color, name: a.name };
      }).sort(function(a, b) { return a.p.z - b.p.z; });

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, 33, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(10, 15, 26, 0.72)";
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
      ctx.stroke();

      for (const a of axes) {
        const tx = cx + a.p.x * len;
        const ty = cy + a.p.y * len;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(tx, ty);
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = a.color;
        ctx.stroke();

        ctx.font = '700 9px ui-monospace, monospace';
        ctx.fillStyle = a.color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(a.name, cx + a.p.x * (len + 10), cy + a.p.y * (len + 10));
      }
      ctx.restore();
    }

    // 60fps animation loop
    let lastFrameTime = performance.now();
    function tick(now) {
      const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
      lastFrameTime = now;

      let animating = false;
      if (state.autoOrbit && !pointer.dragging) {
        cam.yaw += dt * 0.28;
        animating = true;
      }

      // Smooth camera target/reset easing
      const ease = 0.14;
      if (Math.abs(cam.goalTargetX - cam.targetX) > 0.001 ||
          Math.abs(cam.goalTargetY - cam.targetY) > 0.001 ||
          Math.abs(cam.goalTargetZ - cam.targetZ) > 0.001) {
        cam.targetX += (cam.goalTargetX - cam.targetX) * ease;
        cam.targetY += (cam.goalTargetY - cam.targetY) * ease;
        cam.targetZ += (cam.goalTargetZ - cam.targetZ) * ease;
        animating = true;
      }

      if (cam.goalYaw !== null) {
        cam.yaw += (cam.goalYaw - cam.yaw) * ease;
        cam.pitch += (cam.goalPitch - cam.pitch) * ease;
        cam.zoom += (cam.goalZoom - cam.zoom) * ease;
        cam.panX += (cam.goalPanX - cam.panX) * ease;
        cam.panY += (cam.goalPanY - cam.panY) * ease;
        animating = true;
        if (Math.abs(cam.goalYaw - cam.yaw) < 0.002 &&
            Math.abs(cam.goalPitch - cam.pitch) < 0.002 &&
            Math.abs(cam.goalZoom - cam.zoom) < 0.002 &&
            Math.abs(cam.goalPanX - cam.panX) < 0.4 &&
            Math.abs(cam.goalPanY - cam.panY) < 0.4) {
          cam.yaw = cam.goalYaw;
          cam.pitch = cam.goalPitch;
          cam.zoom = cam.goalZoom;
          cam.panX = cam.goalPanX;
          cam.panY = cam.goalPanY;
          cam.goalYaw = cam.goalPitch = cam.goalZoom = cam.goalPanX = cam.goalPanY = null;
        }
      }

      if (!REDUCED_MOTION && (state.queryResult || state.selectedNodeId || state.hoveredNodeId)) {
        animating = true;
      }

      if (state.needsRender || animating) {
        drawScene(now);
        state.needsRender = false;
      }

      requestAnimationFrame(tick);
    }

    // Pointer Interaction (Orbit, Pan, Wheel Zoom, Hover, Click, DblClick)
    const pointer = {
      dragging: false,
      mode: "orbit",
      startX: 0,
      startY: 0,
      lastX: 0,
      lastY: 0,
      moved: 0
    };

    function hitTest(clientX, clientY) {
      // projectedNodes is sorted far-to-near; scan reverse for front-most sphere
      for (let i = projectedNodes.length - 1; i >= 0; i--) {
        const p = projectedNodes[i];
        const dx = clientX - p.sx;
        const dy = clientY - p.sy;
        const hitR = p.r + 5;
        if (dx * dx + dy * dy <= hitR * hitR) {
          return p.node;
        }
      }
      return null;
    }

    function updateTooltip(node, clientX, clientY) {
      const tt = document.getElementById("hover-tooltip");
      if (!node) {
        tt.style.display = "none";
        return;
      }
      tt.replaceChildren();

      const titleDiv = document.createElement("div");
      titleDiv.className = "tt-title";
      titleDiv.textContent = node.title;
      tt.appendChild(titleDiv);

      const metaDiv = document.createElement("div");
      metaDiv.className = "tt-meta";

      const pBadge = document.createElement("span");
      pBadge.className = "badge";
      pBadge.textContent = node.provider;
      metaDiv.appendChild(pBadge);

      const cBadge = document.createElement("span");
      cBadge.className = "badge";
      const dot = document.createElement("span");
      dot.className = "swatch-dot";
      dot.style.background = clusterColor(node.dominant_concept);
      cBadge.appendChild(dot);
      const cText = document.createElement("span");
      cText.textContent = node.dominant_concept;
      cBadge.appendChild(cText);
      metaDiv.appendChild(cBadge);

      const sBadge = document.createElement("span");
      sBadge.className = "badge";
      sBadge.textContent = node.step_count + " steps";
      metaDiv.appendChild(sBadge);

      const m = state.queryMatchById.get(node.id);
      if (m) {
        const mBadge = document.createElement("span");
        mBadge.className = "badge";
        mBadge.style.borderColor = "rgba(251, 191, 36, 0.55)";
        mBadge.style.color = "#fde68a";
        mBadge.textContent = Number(m.match_pct).toFixed(1) + "% match";
        metaDiv.appendChild(mBadge);
      }

      tt.appendChild(metaDiv);
      tt.style.display = "block";

      const rect = tt.getBoundingClientRect();
      let left = clientX + 14;
      let top = clientY + 14;
      if (left + rect.width > viewW - 10) left = Math.max(10, clientX - rect.width - 14);
      if (top + rect.height > viewH - 10) top = Math.max(10, clientY - rect.height - 14);
      tt.style.left = left + "px";
      tt.style.top = top + "px";
    }

    canvas.addEventListener("contextmenu", function(e) { e.preventDefault(); });

    canvas.addEventListener("mousedown", function(e) {
      cam.goalYaw = cam.goalPitch = cam.goalZoom = cam.goalPanX = cam.goalPanY = null;
      pointer.dragging = true;
      pointer.mode = (e.button === 2 || e.button === 1 || e.shiftKey) ? "pan" : "orbit";
      pointer.startX = pointer.lastX = e.clientX;
      pointer.startY = pointer.lastY = e.clientY;
      pointer.moved = 0;
      canvas.classList.add("is-dragging");
    });

    window.addEventListener("mousemove", function(e) {
      if (pointer.dragging) {
        const dx = e.clientX - pointer.lastX;
        const dy = e.clientY - pointer.lastY;
        pointer.moved += Math.hypot(dx, dy);
        pointer.lastX = e.clientX;
        pointer.lastY = e.clientY;
        if (pointer.mode === "pan") {
          cam.panX += dx;
          cam.panY += dy;
        } else {
          cam.yaw += dx * 0.0065;
          cam.pitch = Math.max(-1.42, Math.min(1.42, cam.pitch + dy * 0.0065));
        }
        updateTooltip(null, 0, 0);
        state.needsRender = true;
        return;
      }

      if (e.target !== canvas) {
        if (state.hoveredNodeId !== null) {
          state.hoveredNodeId = null;
          canvas.classList.remove("is-hovering");
          updateTooltip(null, 0, 0);
          state.needsRender = true;
        }
        return;
      }

      const hit = hitTest(e.clientX, e.clientY);
      const nextId = hit ? hit.id : null;
      if (nextId !== state.hoveredNodeId) {
        state.hoveredNodeId = nextId;
        canvas.classList.toggle("is-hovering", Boolean(hit));
        state.needsRender = true;
      }
      updateTooltip(hit, e.clientX, e.clientY);
    });

    window.addEventListener("mouseup", function(e) {
      if (!pointer.dragging) return;
      const wasClick = pointer.moved < 5 && e.button === 0;
      pointer.dragging = false;
      canvas.classList.remove("is-dragging");
      if (wasClick && e.target === canvas) {
        const hit = hitTest(e.clientX, e.clientY);
        if (hit) {
          selectNode(hit.id, true);
        }
      }
    });

    canvas.addEventListener("dblclick", function(e) {
      const hit = hitTest(e.clientX, e.clientY);
      if (!hit) resetCamera();
    });

    canvas.addEventListener("wheel", function(e) {
      e.preventDefault();
      cam.goalZoom = null;
      const factor = Math.exp(-e.deltaY * 0.0011);
      cam.zoom = Math.max(0.35, Math.min(6.0, cam.zoom * factor));
      state.needsRender = true;
    }, { passive: false });

    window.addEventListener("keydown", function(e) {
      const tag = document.activeElement ? document.activeElement.tagName : "";
      const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (e.key === "Escape") {
        if (typing) document.activeElement.blur();
        else if (state.drawerOpen) closeDrawer();
        return;
      }
      if (typing) return;
      if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        resetCamera();
      } else if (e.key === "/") {
        e.preventDefault();
        const inp = document.getElementById("query-input");
        if (inp) inp.focus();
      }
    });

    // Safe DOM Helper Builders (Strict XSS-safe: createElement + textContent only)
    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }

    function formatDate(raw) {
      if (!raw) return "Unknown date";
      const d = new Date(raw);
      if (Number.isNaN(d.getTime())) return raw;
      return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
    }

    function drawFingerprintCanvas(cv, vec, accentHex) {
      const vals = Array.isArray(vec) ? vec : [];
      const w = 348;
      const h = 76;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = w * ratio;
      cv.height = h * ratio;
      const c = cv.getContext("2d");
      c.setTransform(ratio, 0, 0, ratio, 0, 0);
      c.clearRect(0, 0, w, h);

      const midY = h * 0.5;
      c.strokeStyle = "rgba(148, 163, 184, 0.25)";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(0, midY);
      c.lineTo(w, midY);
      c.stroke();

      if (!vals.length) {
        c.fillStyle = "#64748b";
        c.font = "11px ui-monospace, monospace";
        c.textAlign = "center";
        c.fillText("No vector preview available", w * 0.5, midY - 8);
        return;
      }

      let maxAbs = 0;
      for (let i = 0; i < vals.length; i++) {
        const a = Math.abs(Number(vals[i]) || 0);
        if (a > maxAbs) maxAbs = a;
      }
      if (maxAbs < 1e-9) {
        c.fillStyle = "#64748b";
        c.font = "11px ui-monospace, monospace";
        c.textAlign = "center";
        c.fillText("Zero-norm vector", w * 0.5, midY - 8);
        return;
      }

      const barW = w / vals.length;
      for (let i = 0; i < vals.length; i++) {
        const v = Number(vals[i]) || 0;
        const norm = v / maxAbs;
        const barH = Math.max(1, Math.abs(norm) * (midY - 5));
        const x = i * barW + 0.5;
        const bw = Math.max(1.2, barW - 1.2);
        c.fillStyle = v >= 0 ? (accentHex || "#22d3ee") : "#fb7185";
        if (v >= 0) c.fillRect(x, midY - barH, bw, barH);
        else c.fillRect(x, midY, bw, barH);
      }
    }

    function renderFingerprintSection(titleText, vec, accentHex) {
      const wrap = el("div");
      const lbl = el("div", "section-label");
      lbl.appendChild(el("span", null, titleText));
      const vals = Array.isArray(vec) ? vec : [];
      const peak = vals.reduce(function(m, v) { return Math.max(m, Math.abs(Number(v) || 0)); }, 0);
      lbl.appendChild(el("span", "mono", vals.length + " buckets · peak " + peak.toFixed(3)));
      wrap.appendChild(lbl);

      const card = el("div", "fingerprint-card");
      const cv = document.createElement("canvas");
      cv.className = "fingerprint-canvas";
      drawFingerprintCanvas(cv, vals, accentHex);
      card.appendChild(cv);

      const footer = el("div", "fingerprint-footer");
      footer.appendChild(el("span", null, "+dim (above axis)"));
      footer.appendChild(el("span", null, "-dim (below axis)"));
      card.appendChild(footer);
      wrap.appendChild(card);
      return wrap;
    }

    function copyTextToClipboard(text, btn) {
      function markCopied() {
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        btn.classList.add("is-active");
        setTimeout(function() {
          btn.textContent = orig;
          btn.classList.remove("is-active");
        }, 1400);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(markCopied).catch(function() {
          fallbackCopy(text, markCopied);
        });
      } else {
        fallbackCopy(text, markCopied);
      }
    }

    function fallbackCopy(text, onDone) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        onDone();
      } catch (_err) {
        // Ignore clipboard denial
      } finally {
        document.body.removeChild(ta);
      }
    }

    function renderNodeInspector() {
      const container = document.getElementById("drawer-content");
      container.replaceChildren();

      const node = NODE_BY_ID.get(state.selectedNodeId);
      if (!node) {
        container.appendChild(el("div", "summary-card", "Click any node in the 3D space or run a query to inspect its 256-dim semantic vector, k-NN neighbors, and recall command."));
        return;
      }

      if (state.queryResult) {
        const backBtn = el("button", "hud-btn", "← Back to Query Results (" + (state.queryResult.matches ? state.queryResult.matches.length : 0) + ")");
        backBtn.type = "button";
        backBtn.addEventListener("click", function() {
          state.drawerTab = "query";
          syncDrawerTabs();
        });
        container.appendChild(backBtn);
      }

      const title = el("h2", "detail-title", node.title);
      container.appendChild(title);

      const meta = el("div", "meta-row");
      const provBadge = el("span", "badge");
      const pDot = el("span", "swatch-dot");
      pDot.style.background = providerColor(node.provider);
      provBadge.appendChild(pDot);
      provBadge.appendChild(el("span", null, node.provider));
      meta.appendChild(provBadge);

      meta.appendChild(el("span", "badge", formatDate(node.last_modified_time)));
      meta.appendChild(el("span", "badge", node.step_count + " steps"));
      container.appendChild(meta);

      // Cluster tags
      const clusterWrap = el("div");
      clusterWrap.appendChild(el("div", "section-label", "Concept Clusters"));
      const cPills = el("div", "pill-wrap");

      const domBadge = el("span", "badge");
      domBadge.style.borderColor = rgba(clusterColor(node.dominant_concept), 0.65);
      const cDot = el("span", "swatch-dot");
      cDot.style.background = clusterColor(node.dominant_concept);
      domBadge.appendChild(cDot);
      domBadge.appendChild(el("span", null, node.dominant_concept + " (dominant)"));
      cPills.appendChild(domBadge);

      if (Array.isArray(node.concepts)) {
        node.concepts.forEach(function(cid) {
          const clean = stripConceptPrefix(cid);
          if (clean !== node.dominant_concept) {
            cPills.appendChild(el("span", "badge", clean));
          }
        });
      } else {
        Object.keys(node.concepts || {}).forEach(function(cid) {
          const clean = stripConceptPrefix(cid);
          const terms = Array.isArray(node.concepts[cid]) ? node.concepts[cid] : [];
          if (clean === node.dominant_concept && !terms.length) return;
          const txt = terms.length ? (clean + ": " + terms.slice(0, 4).join(", ")) : clean;
          cPills.appendChild(el("span", "badge", txt));
        });
      }
      clusterWrap.appendChild(cPills);
      container.appendChild(clusterWrap);

      // Summary
      if (node.summary) {
        const sumWrap = el("div");
        sumWrap.appendChild(el("div", "section-label", "Summary"));
        sumWrap.appendChild(el("div", "summary-card", node.summary));
        container.appendChild(sumWrap);
      }

      // Top Distinctive Tokens
      const tokWrap = el("div");
      const tokLbl = el("div", "section-label");
      tokLbl.appendChild(el("span", null, "Top Distinctive Tokens"));
      tokLbl.appendChild(el("span", null, "click to probe"));
      tokWrap.appendChild(tokLbl);
      const tokPills = el("div", "pill-wrap");
      if (node.top_tokens.length) {
        node.top_tokens.forEach(function(tok) {
          const pill = el("button", "token-pill", tok);
          pill.type = "button";
          pill.addEventListener("click", function() {
            const inp = document.getElementById("query-input");
            inp.value = tok;
            executeQuery(tok);
          });
          tokPills.appendChild(pill);
        });
      } else {
        tokPills.appendChild(el("span", "badge", "No extracted tokens"));
      }
      tokWrap.appendChild(tokPills);
      container.appendChild(tokWrap);

      // 256-dim Vector Fingerprint
      container.appendChild(
        renderFingerprintSection("256-dim Vector Fingerprint", node.vector_preview, nodeColor(node))
      );

      // Top True 256D Neighbors
      const knnWrap = el("div");
      knnWrap.appendChild(el("div", "section-label", "Top True 256D Neighbors"));
      const knnList = el("div", "neighbor-list");
      const neighbors = node.knn || [];
      if (!neighbors.length) {
        knnList.appendChild(el("div", "summary-card", "No k-NN edges above cosine floor for this node."));
      } else {
        neighbors.forEach(function(edge) {
          const target = NODE_BY_ID.get(edge.id);
          const cos = Number(edge.cos) || 0;
          const btn = el("button", "neighbor-item");
          btn.type = "button";
          if (!target) btn.disabled = true;

          const top = el("div", "neighbor-top");
          const left = el("div", "legend-left");
          const dot = el("span", "swatch-dot");
          dot.style.background = target ? nodeColor(target) : "#64748b";
          left.appendChild(dot);
          left.appendChild(el("span", "neighbor-title", target ? target.title : edge.id));
          top.appendChild(left);
          top.appendChild(el("span", "score-pill", "cos " + cos.toFixed(3)));
          btn.appendChild(top);

          const track = el("div", "cos-track");
          const fill = el("div", "cos-fill");
          fill.style.width = Math.max(4, Math.min(100, cos * 100)) + "%";
          track.appendChild(fill);
          btn.appendChild(track);

          if (target) {
            btn.addEventListener("click", function() {
              selectNode(target.id, true);
            });
          }
          knnList.appendChild(btn);
        });
      }
      knnWrap.appendChild(knnList);
      container.appendChild(knnWrap);

      // Copyable v1-recall command
      const cmdWrap = el("div");
      cmdWrap.appendChild(el("div", "section-label", "Recall Command"));
      const quotedId = /\s/.test(node.id) ? ('"' + node.id.replace(/"/g, '\\"') + '"') : node.id;
      const recallCmd = "v1-recall --show " + quotedId;
      const cmdBox = el("div", "cmd-box");
      cmdBox.appendChild(el("code", "cmd-text", recallCmd));
      const copyBtn = el("button", "hud-btn", "Copy");
      copyBtn.type = "button";
      copyBtn.style.height = "26px";
      copyBtn.addEventListener("click", function() { copyTextToClipboard(recallCmd, copyBtn); });
      cmdBox.appendChild(copyBtn);
      cmdWrap.appendChild(cmdBox);
      container.appendChild(cmdWrap);
    }

    function renderQueryResults() {
      const container = document.getElementById("drawer-content");
      container.replaceChildren();
      const qr = state.queryResult;
      if (!qr) {
        container.appendChild(el("div", "summary-card", "Enter a query in the top bar and click Probe to project the query vector into 3D space."));
        return;
      }

      const headerCard = el("div", "summary-card");
      const meta = el("div", "meta-row");
      meta.style.marginBottom = "6px";
      meta.appendChild(el("span", qr.hybrid_rag ? "badge badge-hybrid" : "badge badge-sparse", qr.hybrid_rag ? "Hybrid-RAG" : "Sparse-Only"));
      meta.appendChild(el("span", "badge", "thresh >= " + Math.round(Number(qr.threshold_pct) || 0) + "%"));
      meta.appendChild(el("span", "badge", Number(qr.elapsed_ms || 0).toFixed(1) + " ms"));
      if (!IS_LIVE_API) {
        meta.appendChild(el("span", "badge", "offline estimate"));
      }
      headerCard.appendChild(meta);
      headerCard.appendChild(el("div", "mono", 'Query: "' + String(qr.query || "") + '"'));
      container.appendChild(headerCard);

      // Query Tokens & Concepts
      const qToks = Array.isArray(qr.query_tokens) ? qr.query_tokens : [];
      const qConcepts = qr.q_concepts || {};
      const conceptKeys = Array.isArray(qConcepts) ? qConcepts : Object.keys(qConcepts);
      if (qToks.length || conceptKeys.length) {
        const qpWrap = el("div");
        qpWrap.appendChild(el("div", "section-label", "Extracted Query Tokens & Concepts"));
        const pills = el("div", "pill-wrap");
        conceptKeys.forEach(function(cid) {
          const clean = stripConceptPrefix(cid);
          const p = el("span", "token-pill is-concept", "concept:" + clean);
          p.style.cursor = "default";
          pills.appendChild(p);
        });
        qToks.forEach(function(tok) {
          const p = el("span", "token-pill", tok);
          p.style.cursor = "default";
          pills.appendChild(p);
        });
        qpWrap.appendChild(pills);
        container.appendChild(qpWrap);
      }

      if (Array.isArray(qr.q_vector_preview) && qr.q_vector_preview.length) {
        container.appendChild(
          renderFingerprintSection("Query 256-dim Vector Fingerprint", qr.q_vector_preview, "#fbbf24")
        );
      }

      // Top Matches
      const matches = Array.isArray(qr.matches) ? qr.matches : [];
      const matchWrap = el("div");
      matchWrap.appendChild(el("div", "section-label", "Top Matches (" + matches.length + ")"));
      const mList = el("div", "match-list");
      if (!matches.length) {
        mList.appendChild(el("div", "summary-card", "No conversations exceeded the " + Math.round(Number(qr.threshold_pct) || 0) + "% match threshold. Inspect Near-Miss Cosine neighbors below or lower the threshold slider."));
      } else {
        matches.forEach(function(m, idx) {
          const targetId = m.conversation_id || m.id;
          const card = el("button", "match-card" + (state.selectedNodeId === targetId ? " is-selected" : ""));
          card.type = "button";

          const top = el("div", "match-top");
          top.appendChild(el("span", "match-title", "#" + (idx + 1) + "  " + (m.title || targetId)));
          top.appendChild(el("span", "score-pill is-gold", Number(m.match_pct || 0).toFixed(1) + "%"));
          card.appendChild(top);

          const metrics = el("div", "match-metrics");
          const modeStr = String(m.retrieval_mode || "hybrid_rrf");
          metrics.appendChild(el("span", modeStr === "sparse" ? "badge badge-sparse" : "badge badge-hybrid", modeStr));
          metrics.appendChild(el("span", null, "sem " + Number(m.semantic_score || 0).toFixed(3)));
          metrics.appendChild(el("span", null, "rrf " + Number(m.rrf_score || 0).toFixed(4)));
          card.appendChild(metrics);

          const mt = Array.isArray(m.matched_tokens) ? m.matched_tokens : [];
          if (mt.length) {
            const mp = el("div", "pill-wrap");
            mt.slice(0, 10).forEach(function(tok) {
              const s = String(tok);
              const pill = el("span", "token-pill" + (s.startsWith("concept:") ? " is-concept" : ""), s);
              pill.style.cursor = "default";
              mp.appendChild(pill);
            });
            card.appendChild(mp);
          }

          card.addEventListener("click", function() {
            selectNode(targetId, true);
          });
          mList.appendChild(card);
        });
      }
      matchWrap.appendChild(mList);
      container.appendChild(matchWrap);

      // Near-Miss Cosine Neighbors
      const near = Array.isArray(qr.nearest_cos) ? qr.nearest_cos : [];
      const nearFiltered = near.filter(function(item) {
        return !state.queryMatchById.has(item.id);
      });
      const nearToRender = nearFiltered.length ? nearFiltered : near;
      if (nearToRender.length) {
        const nearWrap = el("div");
        nearWrap.appendChild(
          el("div", "section-label", nearFiltered.length ? "Near-Miss Cosine Neighbors" : "Top 256D Cosine Neighbors")
        );
        const nList = el("div", "neighbor-list");
        nearToRender.forEach(function(item) {
          const target = NODE_BY_ID.get(item.id);
          const cos = Number(item.cos) || 0;
          const btn = el("button", "neighbor-item");
          btn.type = "button";

          const top = el("div", "neighbor-top");
          const left = el("div", "legend-left");
          const dot = el("span", "swatch-dot");
          dot.style.background = target ? nodeColor(target) : "#94a3b8";
          left.appendChild(dot);
          left.appendChild(el("span", "neighbor-title", item.title || (target ? target.title : item.id)));
          top.appendChild(left);
          top.appendChild(el("span", "score-pill", "cos " + cos.toFixed(3)));
          btn.appendChild(top);

          const track = el("div", "cos-track");
          const fill = el("div", "cos-fill");
          fill.style.width = Math.max(4, Math.min(100, cos * 100)) + "%";
          track.appendChild(fill);
          btn.appendChild(track);

          btn.addEventListener("click", function() {
            if (NODE_BY_ID.has(item.id)) selectNode(item.id, true);
          });
          nList.appendChild(btn);
        });
        nearWrap.appendChild(nList);
        container.appendChild(nearWrap);
      }
    }

    function syncDrawerTabs() {
      const drawer = document.getElementById("right-drawer");
      const tabInsp = document.getElementById("tab-inspector");
      const tabQuery = document.getElementById("tab-query");
      const hint = document.getElementById("bottom-hint");
      const mCount = state.queryResult && Array.isArray(state.queryResult.matches)
        ? state.queryResult.matches.length
        : 0;
      tabQuery.textContent = "Query Results (" + mCount + ")";
      tabInsp.classList.toggle("is-active", state.drawerTab === "inspector");
      tabQuery.classList.toggle("is-active", state.drawerTab === "query");
      drawer.classList.toggle("is-open", state.drawerOpen);

      if (state.drawerOpen && viewW < 1040) {
        setLeftPanelCollapsed(true);
      }
      if (hint) {
        hint.style.display = (state.drawerOpen && viewW < 760) ? "none" : "flex";
      }
      state.needsRender = true;

      if (state.drawerOpen) {
        if (state.drawerTab === "query") renderQueryResults();
        else renderNodeInspector();
      }
    }

    function closeDrawer() {
      state.drawerOpen = false;
      syncDrawerTabs();
    }

    function selectNode(nodeId, openInspector) {
      if (!NODE_BY_ID.has(nodeId)) return;
      state.selectedNodeId = nodeId;
      const node = NODE_BY_ID.get(nodeId);
      focusOnPoint(node.pos);
      if (openInspector) {
        state.drawerTab = "inspector";
        state.drawerOpen = true;
      }
      syncDrawerTabs();
      state.needsRender = true;
    }

    function applyQueryResult(data) {
      state.queryResult = data;
      state.queryMatchById = new Map();
      state.queryNearById = new Map();
      (Array.isArray(data.matches) ? data.matches : []).forEach(function(m) {
        const nid = m.conversation_id || m.id;
        if (nid) state.queryMatchById.set(nid, m);
      });
      (Array.isArray(data.nearest_cos) ? data.nearest_cos : []).forEach(function(n) {
        if (n && n.id) state.queryNearById.set(n.id, n);
      });

      const elapsedEl = document.getElementById("query-elapsed");
      elapsedEl.textContent = Number(data.elapsed_ms || 0).toFixed(1) + " ms";
      document.getElementById("btn-clear-query").classList.add("is-visible");

      if (Array.isArray(data.q_pos)) {
        focusOnPoint(safePos(data.q_pos));
      }
      state.drawerTab = "query";
      state.drawerOpen = true;
      syncDrawerTabs();
      state.needsRender = true;
    }

    function clearQueryOverlay() {
      state.queryResult = null;
      state.queryMatchById.clear();
      state.queryNearById.clear();
      document.getElementById("query-input").value = "";
      document.getElementById("query-elapsed").textContent = "-- ms";
      document.getElementById("btn-clear-query").classList.remove("is-visible");
      if (state.drawerTab === "query") {
        if (state.selectedNodeId) state.drawerTab = "inspector";
        else state.drawerOpen = false;
      }
      syncDrawerTabs();
      state.needsRender = true;
    }

    async function executeQuery(queryText) {
      const q = String(queryText || "").trim();
      if (!q) {
        clearQueryOverlay();
        return;
      }
      showError("");
      state.isProbing = true;
      const probeBtn = document.getElementById("btn-probe");
      probeBtn.textContent = "Probing…";

      try {
        if (!IS_LIVE_API) {
          const offlineRes = runOfflineQuery(q, state.thresholdPct, state.hybridRag);
          applyQueryResult(offlineRes);
          return;
        }
        const resp = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: q,
            threshold_pct: state.thresholdPct,
            hybrid_rag: state.hybridRag
          })
        });
        const payload = await resp.json();
        if (!resp.ok) {
          throw new Error(payload && payload.error ? payload.error : ("HTTP " + resp.status));
        }
        applyQueryResult(payload);
      } catch (err) {
        showError("Query failed: " + String(err.message || err));
      } finally {
        state.isProbing = false;
        probeBtn.textContent = "Probe";
      }
    }

    function renderTelemetryAndLegend() {
      document.getElementById("stat-docs").textContent = String(NODES.length);
      document.getElementById("stat-edges").textContent = String(UNIQUE_EDGES.length);

      const clusterSet = new Set();
      NODES.forEach(function(n) { clusterSet.add(n.dominant_concept); });
      document.getElementById("stat-clusters").textContent = String(clusterSet.size);

      // PCA variance bars
      const varArr = Array.isArray(GRAPH.variance_explained) ? GRAPH.variance_explained : [0, 0, 0];
      const pcMeta = [
        { label: "PC1", val: Number(varArr[0]) || 0, color: "#22d3ee" },
        { label: "PC2", val: Number(varArr[1]) || 0, color: "#10b981" },
        { label: "PC3", val: Number(varArr[2]) || 0, color: "#a78bfa" }
      ];
      const totalVar = pcMeta.reduce(function(acc, item) { return acc + item.val; }, 0);
      document.getElementById("pca-total").textContent = totalVar.toFixed(1) + "% total";

      const pcaBars = document.getElementById("pca-bars");
      pcaBars.replaceChildren();
      pcMeta.forEach(function(item) {
        const row = el("div", "pca-row");
        row.appendChild(el("span", null, item.label));
        const track = el("div", "pca-track");
        const fill = el("div", "pca-fill");
        fill.style.width = Math.max(0, Math.min(100, item.val)) + "%";
        fill.style.background = item.color;
        track.appendChild(fill);
        row.appendChild(track);
        const valSpan = el("span", null, item.val.toFixed(1) + "%");
        valSpan.style.textAlign = "right";
        row.appendChild(valSpan);
        pcaBars.appendChild(row);
      });

      // Cluster or Provider Legend
      const legendHeading = document.getElementById("legend-heading");
      legendHeading.textContent = state.colorBy === "provider" ? "Provider Filter" : "Cluster Filter";

      const counts = new Map();
      NODES.forEach(function(n) {
        const k = state.colorBy === "provider" ? n.provider : n.dominant_concept;
        counts.set(k, (counts.get(k) || 0) + 1);
      });
      if (state.colorBy === "cluster" && Array.isArray(GRAPH.cluster_names)) {
        GRAPH.cluster_names.forEach(function(cn) {
          const clean = stripConceptPrefix(cn);
          if (!counts.has(clean)) counts.set(clean, 0);
        });
      }

      const sortedKeys = Array.from(counts.keys()).sort(function(a, b) {
        const diff = (counts.get(b) || 0) - (counts.get(a) || 0);
        return diff !== 0 ? diff : a.localeCompare(b);
      });

      const clearFilterBtn = document.getElementById("btn-clear-filter");
      clearFilterBtn.style.display = state.activeFilter ? "inline-flex" : "none";

      const legendList = document.getElementById("legend-list");
      legendList.replaceChildren();
      sortedKeys.forEach(function(key) {
        const count = counts.get(key) || 0;
        const isSel = state.activeFilter === key;
        const isDim = Boolean(state.activeFilter && !isSel);
        const btn = el("button", "legend-pill" + (isSel ? " is-selected" : "") + (isDim ? " is-dimmed" : ""));
        btn.type = "button";

        const left = el("div", "legend-left");
        const dot = el("span", "swatch-dot");
        dot.style.background = state.colorBy === "provider" ? providerColor(key) : clusterColor(key);
        left.appendChild(dot);
        left.appendChild(el("span", "legend-name", key));
        btn.appendChild(left);
        btn.appendChild(el("span", "legend-count", String(count)));

        btn.addEventListener("click", function() {
          state.activeFilter = (state.activeFilter === key) ? null : key;
          renderTelemetryAndLegend();
          state.needsRender = true;
        });
        legendList.appendChild(btn);
      });
    }

    // Wire Controls & Events
    function initControls() {
      const livePill = document.getElementById("live-status-pill");
      const liveText = document.getElementById("live-status-text");
      if (!IS_LIVE_API) {
        livePill.classList.add("is-offline");
        liveText.textContent = "OFFLINE";
      } else {
        liveText.textContent = "LIVE";
      }

      const slider = document.getElementById("threshold-slider");
      const sliderVal = document.getElementById("threshold-val");
      slider.value = String(state.thresholdPct);
      sliderVal.textContent = state.thresholdPct + "%";

      let sliderDebounce = null;
      slider.addEventListener("input", function() {
        state.thresholdPct = Math.max(0, Math.min(100, Number(slider.value) || 0));
        sliderVal.textContent = state.thresholdPct + "%";
        if (state.queryResult && state.queryResult.query) {
          clearTimeout(sliderDebounce);
          sliderDebounce = setTimeout(function() {
            executeQuery(state.queryResult.query);
          }, 160);
        }
      });

      const ragBtn = document.getElementById("btn-rag-mode");
      function syncRagBtn() {
        ragBtn.textContent = state.hybridRag ? "Hybrid-RAG" : "Sparse-Only";
        ragBtn.setAttribute("aria-pressed", state.hybridRag ? "true" : "false");
        ragBtn.classList.toggle("is-hybrid", state.hybridRag);
        ragBtn.classList.toggle("is-sparse", !state.hybridRag);
      }
      syncRagBtn();
      ragBtn.addEventListener("click", function() {
        state.hybridRag = !state.hybridRag;
        syncRagBtn();
        if (state.queryResult && state.queryResult.query) {
          executeQuery(state.queryResult.query);
        }
      });

      document.getElementById("color-by-select").addEventListener("change", function(e) {
        state.colorBy = e.target.value === "provider" ? "provider" : "cluster";
        state.activeFilter = null;
        renderTelemetryAndLegend();
        if (state.drawerOpen && state.drawerTab === "inspector") renderNodeInspector();
        state.needsRender = true;
      });

      function bindToggle(btnId, prop) {
        const btn = document.getElementById(btnId);
        function sync() {
          btn.classList.toggle("is-active", Boolean(state[prop]));
          btn.setAttribute("aria-pressed", state[prop] ? "true" : "false");
        }
        sync();
        btn.addEventListener("click", function() {
          state[prop] = !state[prop];
          sync();
          state.needsRender = true;
        });
      }

      bindToggle("btn-knn", "showKnn");
      bindToggle("btn-centroids", "showCentroids");
      bindToggle("btn-labels", "showLabels");
      bindToggle("btn-orbit", "autoOrbit");

      document.getElementById("btn-reset-cam").addEventListener("click", resetCamera);

      document.getElementById("query-form").addEventListener("submit", function(e) {
        e.preventDefault();
        executeQuery(document.getElementById("query-input").value);
      });

      const queryInput = document.getElementById("query-input");
      queryInput.addEventListener("input", function() {
        const hasVal = Boolean(queryInput.value.trim() || state.queryResult);
        document.getElementById("btn-clear-query").classList.toggle("is-visible", hasVal);
      });

      document.getElementById("btn-clear-query").addEventListener("click", clearQueryOverlay);

      const leftPanel = document.getElementById("left-panel");
      const collapseBtn = document.getElementById("btn-collapse-left");
      collapseBtn.addEventListener("click", function() {
        const nextCollapsed = !leftPanel.classList.contains("is-collapsed");
        setLeftPanelCollapsed(nextCollapsed);
        if (!nextCollapsed && viewW < 1040 && state.drawerOpen) {
          closeDrawer();
        }
      });

      const topHud = document.getElementById("top-hud");
      if (topHud && typeof ResizeObserver === "function") {
        new ResizeObserver(updateHudLayout).observe(topHud);
      }

      document.getElementById("btn-clear-filter").addEventListener("click", function() {
        state.activeFilter = null;
        renderTelemetryAndLegend();
        state.needsRender = true;
      });

      document.getElementById("tab-inspector").addEventListener("click", function() {
        state.drawerTab = "inspector";
        syncDrawerTabs();
      });
      document.getElementById("tab-query").addEventListener("click", function() {
        state.drawerTab = "query";
        syncDrawerTabs();
      });
      document.getElementById("btn-close-drawer").addEventListener("click", closeDrawer);

      if (NODES.length === 0) {
        document.getElementById("empty-banner").style.display = "block";
      }
    }

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    initControls();
    renderTelemetryAndLegend();
    requestAnimationFrame(tick);
  })();
  </script>
</body>
</html>
"""


def render_visualizer_html(initial_graph_json: str, live_api: bool = True) -> str:
    """Render a zero-build, 100% offline self-contained 3D RAG Vector Explorer HTML document.

    Escapes ``</`` -> ``<\\/`` and HTML comment openers before embedding ``initial_graph_json``
    inside ``<script id="initial-graph-data" type="application/json">`` so untrusted conversation
    metadata cannot break out of the script element or execute XSS payloads.
    """
    if isinstance(initial_graph_json, str):
        raw_json = initial_graph_json.strip() or "{}"
    else:
        raw_json = json.dumps(initial_graph_json, ensure_ascii=False)

    safe_json = raw_json.replace("</", "<\\/").replace("<!--", "\\u003c!--")
    live_flag = "true" if live_api else "false"
    return (
        _HTML_TEMPLATE.replace("__INITIAL_GRAPH_JSON__", safe_json, 1).replace(
            "__LIVE_API_BOOL__", live_flag, 1
        )
    )
