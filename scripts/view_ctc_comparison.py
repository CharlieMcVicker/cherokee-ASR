#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_ctc_comparison.py

Lightweight webview server and UI to inspect 3-Way Comparative Alignments:
1. Native Cherokee Syllabary
2. Greedy ASR Inference (Unconstrained acoustic emission)
3. Syllabary-Guided CTC Segmentation (Trellis phonotactics emission)

Supports playing verse and word-level audio snippets, diff highlighting, and anomaly filtering.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
from typing import Any, Dict, List, Optional
import urllib.parse
import webbrowser

BASE_DIR = Path(__file__).resolve().parent.parent

# Candidate default JSON files in order of priority
DEFAULT_CANDIDATE_JSONS = [
    BASE_DIR / "cherokee_new_testament" / "alignments" / "mark_alignment_records.json",
    BASE_DIR / "cherokee_new_testament" / "alignments" / "bible_alignment_records.json",
    BASE_DIR / "runs" / "evaluation" / "ctc_segmentation_100_verses_comparison.json",
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cherokee Alignment 3-Way Inspector (Syllabary vs Greedy vs Guided)</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --surface-hover: #1f242c;
      --surface-border: #30363d;
      --text-primary: #f0f6fc;
      --text-secondary: #8b949e;
      --text-muted: #6e7681;
      --accent: #58a6ff;
      --accent-glow: rgba(88, 166, 255, 0.15);
      --greedy-color: #79c0ff;
      --greedy-bg: rgba(56, 139, 253, 0.12);
      --guided-color: #d2a8ff;
      --guided-bg: rgba(187, 128, 247, 0.12);
      --danger: #f85149;
      --danger-bg: rgba(248, 81, 73, 0.15);
      --danger-border: rgba(248, 81, 73, 0.4);
      --warning: #d29922;
      --warning-bg: rgba(210, 153, 34, 0.15);
      --warning-border: rgba(210, 153, 34, 0.4);
      --success: #3fb950;
      --success-bg: rgba(63, 185, 80, 0.15);
      --success-border: rgba(63, 185, 80, 0.4);
      --tag-bg: #21262d;
      --card-radius: 12px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg);
      color: var(--text-primary);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding-bottom: 90px;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(22, 27, 34, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--surface-border);
      padding: 16px 24px;
    }

    .header-content {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }

    .title-row h1 {
      font-size: 1.25rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .stats-bar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .stat-badge {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.8rem;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .stat-badge:hover {
      border-color: var(--accent);
      background: var(--surface-hover);
    }

    .stat-badge.active {
      border-color: var(--accent);
      background: var(--accent-glow);
      color: var(--text-primary);
    }

    .stat-badge.danger {
      border-color: var(--danger-border);
    }

    .stat-badge.danger.active {
      box-shadow: 0 0 12px rgba(248, 81, 73, 0.35);
      border-color: var(--danger);
    }

    .controls-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }

    .search-box {
      flex: 1;
      min-width: 260px;
      position: relative;
    }

    .search-box input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 8px 12px 8px 34px;
      color: var(--text-primary);
      font-size: 0.88rem;
      outline: none;
      transition: border-color 0.15s ease;
    }

    .search-box input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }

    .search-icon {
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .filter-tabs {
      display: flex;
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 3px;
      gap: 2px;
    }

    .tab-btn {
      background: none;
      border: none;
      color: var(--text-secondary);
      padding: 6px 14px;
      font-size: 0.82rem;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.15s ease;
    }

    .tab-btn:hover {
      color: var(--text-primary);
    }

    .tab-btn.active {
      background: var(--surface);
      color: var(--text-primary);
      box-shadow: 0 1px 3px rgba(0,0,0,0.4);
      font-weight: 600;
    }

    .tab-btn.active.anomalies {
      color: #ff7b72;
    }

    .sort-select {
      background: var(--bg);
      border: 1px solid var(--surface-border);
      color: var(--text-primary);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      outline: none;
      cursor: pointer;
    }

    main {
      max-width: 1400px;
      margin: 20px auto;
      padding: 0 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .verse-card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--card-radius);
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      transition: border-color 0.2s ease;
    }

    .verse-card:hover {
      border-color: #444c56;
    }

    .verse-card.has-anomaly {
      border-left: 4px solid var(--danger);
      background: linear-gradient(90deg, rgba(248, 81, 73, 0.05) 0%, var(--surface) 12%);
    }

    .verse-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid var(--surface-border);
      padding-bottom: 12px;
    }

    .verse-title-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .verse-id-badge {
      background: var(--tag-bg);
      border: 1px solid var(--surface-border);
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-primary);
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .verse-meta {
      font-size: 0.84rem;
      color: var(--text-muted);
    }

    .anomaly-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.76rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .anomaly-badge.danger {
      background: var(--danger-bg);
      border: 1px solid var(--danger-border);
      color: #ff7b72;
    }

    .anomaly-badge.success {
      background: var(--success-bg);
      border: 1px solid var(--success-border);
      color: #7ee787;
    }

    .player-container {
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .play-btn {
      background: var(--accent);
      color: #000;
      border: none;
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 0.95rem;
      font-weight: bold;
      transition: transform 0.1s ease, background 0.15s ease;
      flex-shrink: 0;
    }

    .play-btn:hover {
      background: #79b8ff;
      transform: scale(1.05);
    }

    .audio-timeline-wrap {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .audio-progress {
      flex: 1;
      height: 8px;
      background: var(--surface-border);
      border-radius: 4px;
      position: relative;
      cursor: pointer;
    }

    .audio-progress-bar {
      height: 100%;
      background: var(--accent);
      border-radius: 4px;
      width: 0%;
      position: relative;
      transition: width 0.05s linear;
    }

    .time-display {
      font-size: 0.78rem;
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
      min-width: 85px;
      text-align: right;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .speed-select {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      color: var(--text-secondary);
      font-size: 0.78rem;
      border-radius: 5px;
      padding: 4px 8px;
      cursor: pointer;
      outline: none;
    }

    /* 3-Way Comparative Tiers */
    .comparison-tiers {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .tier-card {
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .tier-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .tier-label {
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .tier-badge {
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .tier-badge.syllabary {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
    }

    .tier-badge.greedy {
      background: var(--greedy-bg);
      color: var(--greedy-color);
      border: 1px solid rgba(88, 166, 255, 0.3);
    }

    .tier-badge.guided {
      background: var(--guided-bg);
      color: var(--guided-color);
      border: 1px solid rgba(187, 128, 247, 0.3);
    }

    .syllabary-content {
      font-size: 1.45rem;
      font-weight: 500;
      color: #ffffff;
      line-height: 1.8;
      letter-spacing: 0.9px;
      font-family: "Plantagenet Cherokee", "Noto Sans Cherokee", "Apple Symbols", sans-serif;
    }

    .phonetic-content {
      font-size: 0.96rem;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      line-height: 1.6;
      word-break: break-word;
    }

    .phonetic-content.greedy {
      color: #a5d6ff;
    }

    .phonetic-content.guided {
      color: #d2a8ff;
    }

    .diff-highlight-box {
      background: rgba(0, 0, 0, 0.3);
      border: 1px dashed var(--surface-border);
      border-radius: 6px;
      padding: 8px 12px;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      font-size: 0.86rem;
      line-height: 1.5;
    }

    .diff-ins {
      background: rgba(63, 185, 80, 0.25);
      color: #7ee787;
      padding: 1px 3px;
      border-radius: 3px;
      font-weight: bold;
    }

    .diff-del {
      background: rgba(248, 81, 73, 0.25);
      color: #ff7b72;
      text-decoration: line-through;
      padding: 1px 3px;
      border-radius: 3px;
    }

    .diff-sub {
      background: rgba(210, 153, 34, 0.25);
      color: #e3b341;
      padding: 1px 3px;
      border-radius: 3px;
    }

    .flagged-callout {
      background: rgba(248, 81, 73, 0.08);
      border: 1px solid var(--danger-border);
      border-radius: 8px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .flagged-callout-header {
      font-weight: 700;
      color: #ff7b72;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .flagged-items-grid {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .flagged-item-row {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(248, 81, 73, 0.2);
      padding: 8px 12px;
      border-radius: 6px;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      font-size: 0.84rem;
    }

    .words-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .section-subtitle {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .words-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .word-chip {
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 5px;
      cursor: pointer;
      transition: all 0.15s ease;
      min-width: 120px;
      position: relative;
    }

    .word-chip:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 4px 10px rgba(0,0,0,0.35);
    }

    .word-chip.active-playback {
      border-color: var(--accent);
      background: var(--accent-glow);
      box-shadow: 0 0 14px rgba(88, 166, 255, 0.45);
    }

    .word-chip.flagged {
      border-color: var(--danger-border);
      background: var(--danger-bg);
    }

    .word-chip.flagged:hover {
      border-color: var(--danger);
      box-shadow: 0 0 12px rgba(248, 81, 73, 0.4);
    }

    .word-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }

    .ref-word {
      font-weight: 700;
      font-size: 0.94rem;
      color: var(--text-primary);
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .play-snippet-icon {
      font-size: 0.72rem;
      color: var(--accent);
      opacity: 0.8;
    }

    .emitted-word {
      font-size: 0.82rem;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      color: var(--guided-color);
    }

    .word-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      font-size: 0.74rem;
    }

    .conf-badge {
      padding: 1px 5px;
      border-radius: 3px;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .conf-high {
      background: var(--success-bg);
      color: #7ee787;
    }

    .conf-mid {
      background: var(--warning-bg);
      color: #e3b341;
    }

    .conf-low {
      background: var(--danger-bg);
      color: #ff7b72;
      font-weight: bold;
    }

    .time-badge {
      color: var(--text-muted);
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .empty-state {
      text-align: center;
      padding: 80px 20px;
      color: var(--text-muted);
      font-size: 1.1rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="header-content">
      <div class="title-row">
        <h1>
          <span>📖</span> Cherokee Alignment 3-Way Inspector
        </h1>
        <div class="stats-bar">
          <div class="stat-badge" id="stat-total" onclick="setFilter('all')">
            Total Verses: <strong id="val-total">0</strong>
          </div>
          <div class="stat-badge danger active" id="stat-anomalies" onclick="setFilter('anomalies')">
            ⚠️ Anomalous Verses: <strong id="val-anomalies">0</strong>
          </div>
          <div class="stat-badge" id="stat-flagged" onclick="setFilter('anomalies')">
            🚩 Flagged Words: <strong id="val-flagged">0</strong>
          </div>
          <div class="stat-badge" id="stat-divergent" onclick="setFilter('divergent')">
            ⚡ Greedy ≠ Guided: <strong id="val-divergent">0</strong>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <div class="search-box">
          <span class="search-icon">🔍</span>
          <input type="text" id="search-input" placeholder="Search verses (e.g. 020101, Mark 1:1, words, syllabary, diffs)..." oninput="render()" />
        </div>

        <div class="filter-tabs">
          <button class="tab-btn" id="tab-all" onclick="setFilter('all')">All Verses</button>
          <button class="tab-btn active anomalies" id="tab-anomalies" onclick="setFilter('anomalies')">⚠️ Anomalies Only</button>
          <button class="tab-btn" id="tab-divergent" onclick="setFilter('divergent')">⚡ Divergent Only</button>
          <button class="tab-btn" id="tab-clean" onclick="setFilter('clean')">✅ Clean Verses</button>
        </div>

        <select class="sort-select" id="sort-select" onchange="render()">
          <option value="anomalies-desc" selected>Sort: Flagged Words (Most First)</option>
          <option value="divergent-first">Sort: Greedy ≠ Guided Divergence First</option>
          <option value="id-asc">Sort: Verse ID (Asc)</option>
          <option value="id-desc">Sort: Verse ID (Desc)</option>
          <option value="confidence-asc">Sort: Lowest Word Confidence</option>
          <option value="duration-desc">Sort: Longest Audio</option>
        </select>
      </div>
    </div>
  </header>

  <main id="verse-container">
    <div class="empty-state">Loading alignment records...</div>
  </main>

  <script>
    let globalData = null;
    let currentFilter = 'anomalies'; // 'all', 'anomalies', 'clean', 'divergent'
    const audioPlayers = new Map(); // verse_id -> Audio instance
    let activeSnippetTimeout = null;

    // Simple LCS character diff between greedy and guided
    function computeCharDiffHtml(s1, s2) {
      if (!s1 || !s2) return '';
      if (s1 === s2) return `<span style="color: var(--text-muted);">(Greedy & Guided match exactly)</span>`;

      // Word level comparison
      const w1 = s1.split(/\s+/);
      const w2 = s2.split(/\s+/);
      
      let html = '<div style="display: flex; flex-direction: column; gap: 4px;">';
      html += '<div><strong>Greedy:</strong> ';
      html += w1.map(word => {
        if (!w2.includes(word)) {
          return `<span class="diff-del">${word}</span>`;
        }
        return `<span>${word}</span>`;
      }).join(' ');
      html += '</div>';

      html += '<div><strong>Guided:</strong> ';
      html += w2.map(word => {
        if (!w1.includes(word)) {
          return `<span class="diff-ins">${word}</span>`;
        }
        return `<span>${word}</span>`;
      }).join(' ');
      html += '</div></div>';

      return html;
    }

    async function init() {
      try {
        if (!globalData) {
          const res = await fetch('/api/data');
          globalData = await res.json();
        }

        // Normalize if list
        if (Array.isArray(globalData)) {
          const total = globalData.length;
          const anomalies = globalData.filter(r => r.has_anomalies);
          const totalFlagged = globalData.reduce((acc, r) => {
            return acc + (r.words ? r.words.filter(w => w.flagged).length : 0);
          }, 0);
          const divergent = globalData.filter(r => {
            const gr = (r.greedy_hypothesis || '').trim();
            const gd = (r.guided_hypothesis || r.reconciled_phonetics || '').trim();
            return gr && gd && gr !== gd;
          }).length;

          globalData = {
            total_verses: total,
            verses_with_anomalies: anomalies.length,
            total_flagged_words: totalFlagged,
            divergent_verses: divergent,
            results: globalData,
          };
        }
        
        document.getElementById('val-total').textContent = globalData.total_verses || (globalData.results ? globalData.results.length : 0);
        document.getElementById('val-anomalies').textContent = globalData.verses_with_anomalies || (globalData.anomalous_verses ? globalData.anomalous_verses.length : 0);
        document.getElementById('val-flagged').textContent = globalData.total_flagged_words || 0;
        document.getElementById('val-divergent').textContent = globalData.divergent_verses || (globalData.results ? globalData.results.filter(r => (r.greedy_hypothesis||'').trim() !== (r.guided_hypothesis||r.reconciled_phonetics||'').trim()).length : 0);

        render();
      } catch (err) {
        console.error("Failed to load data:", err);
        document.getElementById('verse-container').innerHTML = `<div class="empty-state" style="color: var(--danger);">Failed to load comparison data: ${err.message}</div>`;
      }
    }

    function setFilter(filter) {
      currentFilter = filter;
      document.getElementById('tab-all').classList.toggle('active', filter === 'all');
      document.getElementById('tab-anomalies').classList.toggle('active', filter === 'anomalies');
      document.getElementById('tab-divergent').classList.toggle('active', filter === 'divergent');
      document.getElementById('tab-clean').classList.toggle('active', filter === 'clean');

      document.getElementById('stat-total').classList.toggle('active', filter === 'all');
      document.getElementById('stat-anomalies').classList.toggle('active', filter === 'anomalies');
      document.getElementById('stat-divergent').classList.toggle('active', filter === 'divergent');
      render();
    }

    function formatVerseTitle(verse) {
      const vid = verse.verse_id;
      if (vid && vid.length === 6) {
        const bookCode = vid.slice(0, 2);
        const bookName = bookCode === '02' ? 'Mark' : (bookCode === '01' ? 'Matthew' : `Book ${bookCode}`);
        const chapter = parseInt(vid.slice(2, 4), 10);
        const verseNum = parseInt(vid.slice(4, 6), 10);
        return `${bookName} ${chapter}:${verseNum} <span style="opacity: 0.6; font-size: 0.8em; font-weight: normal;">(#${vid})</span>`;
      }
      return `Verse ${vid || 'Unknown'}`;
    }

    function getConfidenceClass(conf) {
      if (conf === undefined || conf === null) return '';
      if (conf < 0.1) return 'conf-low';
      if (conf < 0.8) return 'conf-mid';
      return 'conf-high';
    }

    function stopAllAudio() {
      if (activeSnippetTimeout) {
        clearTimeout(activeSnippetTimeout);
        activeSnippetTimeout = null;
      }
      audioPlayers.forEach((player) => {
        player.pause();
      });
      document.querySelectorAll('.play-btn').forEach(btn => btn.textContent = '▶');
      document.querySelectorAll('.word-chip').forEach(chip => chip.classList.remove('active-playback'));
    }

    function getAudioPlayer(verseId, audioPath) {
      if (!audioPlayers.has(verseId)) {
        const audio = new Audio('/audio/' + encodeURI(audioPath));
        audioPlayers.set(verseId, audio);

        audio.addEventListener('timeupdate', () => {
          const bar = document.getElementById(`bar-${verseId}`);
          const timeEl = document.getElementById(`time-${verseId}`);
          if (bar && audio.duration) {
            const pct = (audio.currentTime / audio.duration) * 100;
            bar.style.width = pct + '%';
          }
          if (timeEl && audio.duration) {
            timeEl.textContent = `${audio.currentTime.toFixed(2)}s / ${audio.duration.toFixed(2)}s`;
          }

          // Highlight active word chips
          const chips = document.querySelectorAll(`[data-verse="${verseId}"][data-start]`);
          chips.forEach(chip => {
            const start = parseFloat(chip.dataset.start);
            const end = parseFloat(chip.dataset.end);
            if (audio.currentTime >= start && audio.currentTime <= end) {
              chip.classList.add('active-playback');
            } else {
              chip.classList.remove('active-playback');
            }
          });
        });

        audio.addEventListener('ended', () => {
          const btn = document.getElementById(`play-btn-${verseId}`);
          if (btn) btn.textContent = '▶';
          const chips = document.querySelectorAll(`[data-verse="${verseId}"]`);
          chips.forEach(c => c.classList.remove('active-playback'));
        });
      }
      return audioPlayers.get(verseId);
    }

    function toggleVerseAudio(verseId, audioPath) {
      const audio = getAudioPlayer(verseId, audioPath);
      const btn = document.getElementById(`play-btn-${verseId}`);

      if (!audio.paused) {
        audio.pause();
        if (btn) btn.textContent = '▶';
      } else {
        stopAllAudio();
        const speedSelect = document.getElementById(`speed-${verseId}`);
        if (speedSelect) {
          audio.playbackRate = parseFloat(speedSelect.value);
        }
        audio.play();
        if (btn) btn.textContent = '⏸';
      }
    }

    function playWordSnippet(verseId, audioPath, startSec, endSec) {
      stopAllAudio();
      const audio = getAudioPlayer(verseId, audioPath);
      const speedSelect = document.getElementById(`speed-${verseId}`);
      if (speedSelect) {
        audio.playbackRate = parseFloat(speedSelect.value);
      }

      audio.currentTime = Math.max(0, startSec);
      audio.play();

      const btn = document.getElementById(`play-btn-${verseId}`);
      if (btn) btn.textContent = '⏸';

      const durationMs = ((endSec - startSec) / audio.playbackRate) * 1000;
      activeSnippetTimeout = setTimeout(() => {
        audio.pause();
        if (btn) btn.textContent = '▶';
        document.querySelectorAll(`[data-verse="${verseId}"]`).forEach(c => c.classList.remove('active-playback'));
      }, Math.max(80, durationMs));
    }

    function seekVerse(event, verseId, audioPath) {
      const audio = getAudioPlayer(verseId, audioPath);
      const progressWrap = event.currentTarget;
      const rect = progressWrap.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      if (audio.duration) {
        audio.currentTime = pct * audio.duration;
      }
    }

    function setPlaybackSpeed(verseId, audioPath, speed) {
      const audio = getAudioPlayer(verseId, audioPath);
      audio.playbackRate = parseFloat(speed);
    }

    function render() {
      if (!globalData || !globalData.results) return;

      const query = document.getElementById('search-input').value.toLowerCase().trim();
      const sortMode = document.getElementById('sort-select').value;
      const container = document.getElementById('verse-container');

      let filtered = globalData.results.filter(v => {
        const isDivergent = (v.greedy_hypothesis || '').trim() !== (v.guided_hypothesis || v.reconciled_phonetics || '').trim();
        if (currentFilter === 'anomalies' && !v.has_anomalies) return false;
        if (currentFilter === 'divergent' && !isDivergent) return false;
        if (currentFilter === 'clean' && v.has_anomalies) return false;

        if (query) {
          const matchId = (v.verse_id || '').toLowerCase().includes(query);
          const matchSyllabary = (v.cherokee_syllabary || v.reference_syllabary || '').toLowerCase().includes(query);
          const matchGreedy = (v.greedy_hypothesis || '').toLowerCase().includes(query);
          const matchGuided = (v.guided_hypothesis || v.reconciled_phonetics || '').toLowerCase().includes(query);
          const matchWords = (v.words || v.words_ctc_aligned || []).some(w => 
            (w.word || '').toLowerCase().includes(query) || ((w.emitted_word || '').toLowerCase().includes(query))
          );
          return matchId || matchSyllabary || matchGreedy || matchGuided || matchWords;
        }
        return true;
      });

      filtered.sort((a, b) => {
        const aWords = a.words || a.words_ctc_aligned || [];
        const bWords = b.words || b.words_ctc_aligned || [];
        const aFlagged = aWords.filter(w => w.flagged).length;
        const bFlagged = bWords.filter(w => w.flagged).length;

        if (sortMode === 'anomalies-desc') {
          if (bFlagged !== aFlagged) return bFlagged - aFlagged;
          return (a.verse_id || '').localeCompare(b.verse_id || '');
        }
        if (sortMode === 'divergent-first') {
          const aDiv = (a.greedy_hypothesis || '').trim() !== (a.guided_hypothesis || a.reconciled_phonetics || '').trim();
          const bDiv = (b.greedy_hypothesis || '').trim() !== (b.guided_hypothesis || b.reconciled_phonetics || '').trim();
          if (aDiv !== bDiv) return aDiv ? -1 : 1;
          return (a.verse_id || '').localeCompare(b.verse_id || '');
        }
        if (sortMode === 'id-asc') return (a.verse_id || '').localeCompare(b.verse_id || '');
        if (sortMode === 'id-desc') return (b.verse_id || '').localeCompare(a.verse_id || '');
        if (sortMode === 'confidence-asc') {
          const getMinConf = (words) => {
            if (!words || words.length === 0) return 1.0;
            return Math.min(...words.map(w => w.confidence ?? 1.0));
          };
          return getMinConf(aWords) - getMinConf(bWords);
        }
        if (sortMode === 'duration-desc') return (b.duration_sec || 0) - (a.duration_sec || 0);
        return 0;
      });

      if (filtered.length === 0) {
        container.innerHTML = `<div class="empty-state">No matching verses found. Try adjusting your search or filter.</div>`;
        return;
      }

      container.innerHTML = filtered.map(v => {
        const hasAnomaly = v.has_anomalies;
        const wordsList = v.words || v.words_ctc_aligned || [];
        const flaggedWords = wordsList.filter(w => w.flagged);
        const flaggedCount = flaggedWords.length;
        const syllabary = v.cherokee_syllabary || v.reference_syllabary || '';
        const greedyHyp = v.greedy_hypothesis || v.asr_hypothesis || '';
        const guidedHyp = v.guided_hypothesis || v.reconciled_phonetics || v.new_reconciled_phonetics || '';
        const diffHtml = computeCharDiffHtml(greedyHyp, guidedHyp);

        return `
          <div class="verse-card ${hasAnomaly ? 'has-anomaly' : ''}" id="card-${v.verse_id}">
            <div class="verse-header">
              <div class="verse-title-group">
                <div class="verse-id-badge">${formatVerseTitle(v)}</div>
                <div class="verse-meta">Duration: <strong>${(v.duration_sec || 0).toFixed(2)}s</strong></div>
              </div>
              <div>
                ${hasAnomaly 
                  ? `<div class="anomaly-badge danger">⚠️ ${flaggedCount} Flagged ${flaggedCount === 1 ? 'Word' : 'Words'}</div>`
                  : `<div class="anomaly-badge success">✅ Clean (0 Flagged)</div>`
                }
              </div>
            </div>

            <div class="player-container">
              <button class="play-btn" id="play-btn-${v.verse_id}" onclick="toggleVerseAudio('${v.verse_id}', '${v.audio_path}')" title="Play / Pause Full Verse">▶</button>
              <div class="audio-timeline-wrap">
                <div class="audio-progress" onclick="seekVerse(event, '${v.verse_id}', '${v.audio_path}')">
                  <div class="audio-progress-bar" id="bar-${v.verse_id}"></div>
                </div>
                <div class="time-display" id="time-${v.verse_id}">0.00s / ${(v.duration_sec || 0).toFixed(2)}s</div>
              </div>
              <select class="speed-select" id="speed-${v.verse_id}" onchange="setPlaybackSpeed('${v.verse_id}', '${v.audio_path}', this.value)">
                <option value="0.5">0.5x</option>
                <option value="0.75">0.75x</option>
                <option value="1" selected>1.0x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
              </select>
            </div>

            <!-- 3-Way Comparative Tiers -->
            <div class="comparison-tiers">
              <!-- Tier 1: Syllabary -->
              <div class="tier-card">
                <div class="tier-header">
                  <span class="tier-label" style="color: #ffffff;"><span>📜</span> 1. Native Cherokee Syllabary</span>
                  <span class="tier-badge syllabary">Ground Truth Syllabary</span>
                </div>
                <div class="syllabary-content">${syllabary}</div>
              </div>

              <!-- Tier 2: Greedy Inference -->
              <div class="tier-card" style="border-left: 3px solid var(--greedy-color);">
                <div class="tier-header">
                  <span class="tier-label" style="color: var(--greedy-color);"><span>🎙️</span> 2. Greedy ASR Inference</span>
                  <span class="tier-badge greedy">Unconstrained Acoustic Emission ${v.greedy_confidence !== undefined ? `(${v.greedy_confidence})` : ''}</span>
                </div>
                <div class="phonetic-content greedy">${greedyHyp || '<span style="color: var(--text-muted);">(No greedy hypothesis recorded)</span>'}</div>
              </div>

              <!-- Tier 3: Guided Inference -->
              <div class="tier-card" style="border-left: 3px solid var(--guided-color);">
                <div class="tier-header">
                  <span class="tier-label" style="color: var(--guided-color);"><span>✨</span> 3. Syllabary-Guided CTC Segmentation</span>
                  <span class="tier-badge guided">Trellis Reconciled Phonotactics ${v.cost !== undefined ? `(Cost: ${v.cost})` : ''}</span>
                </div>
                <div class="phonetic-content guided">${guidedHyp || '<span style="color: var(--text-muted);">(No guided hypothesis recorded)</span>'}</div>
              </div>

              <!-- Comparative Diff Highlight -->
              <div class="diff-highlight-box">
                <div style="font-size: 0.74rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; font-weight: 600;">
                  🔍 Greedy vs Guided Word Diff Comparison:
                </div>
                ${diffHtml}
              </div>
            </div>

            ${hasAnomaly && flaggedCount > 0 ? `
              <div class="flagged-callout">
                <div class="flagged-callout-header">
                  <span>🚨</span> Flagged Low-Confidence Words / Anomalies (${flaggedCount})
                </div>
                <div class="flagged-items-grid">
                  ${flaggedWords.map(f => `
                    <div class="flagged-item-row">
                      <span>Target: <strong style="color: var(--text-primary);">${f.word}</strong></span>
                      <span>→ Emitted: <strong style="color: #ff7b72;">${f.emitted_word || f.word}</strong></span>
                      <span class="conf-badge conf-low">Confidence: ${(f.confidence || 0).toFixed(6)}</span>
                      <span style="font-size: 0.76rem; color: var(--text-muted);">${(f.start_sec || 0).toFixed(2)}s – ${(f.end_sec || 0).toFixed(2)}s</span>
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : ''}

            <div class="words-section">
              <div class="section-subtitle">
                <span>⚡</span> Word-Level Alignment Chips (${wordsList.length} words, click any chip to play audio)
              </div>
              <div class="words-grid">
                ${wordsList.map((w) => {
                  const isFlagged = w.flagged || (w.confidence !== undefined && w.confidence < 0.1);
                  const confCls = getConfidenceClass(w.confidence);
                  return `
                    <div 
                      class="word-chip ${isFlagged ? 'flagged' : ''}" 
                      data-verse="${v.verse_id}" 
                      data-start="${w.start_sec}" 
                      data-end="${w.end_sec}"
                      onclick="playWordSnippet('${v.verse_id}', '${v.audio_path}', ${w.start_sec}, ${w.end_sec})"
                      title="Click to play ${w.start_sec.toFixed(2)}s - ${w.end_sec.toFixed(2)}s"
                    >
                      <div class="word-header">
                        <span class="ref-word">${w.word}</span>
                        <span class="play-snippet-icon">▶</span>
                      </div>
                      ${w.emitted_word && w.emitted_word !== w.word ? `
                        <div class="emitted-word">${w.emitted_word}</div>
                      ` : `
                        <div class="emitted-word" style="opacity: 0.5;">${w.word}</div>
                      `}
                      <div class="word-footer">
                        <span class="conf-badge ${confCls}">
                          ${w.confidence !== undefined ? (w.confidence >= 0.999 ? '1.000' : w.confidence.toFixed(3)) : '—'}
                        </span>
                        <span class="time-badge">${w.start_sec.toFixed(2)}s</span>
                      </div>
                    </div>
                  `;
                }).join('')}
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Request Handler with support for Range requests (essential for audio seeking/buffering),
    serving the embedded interactive HTML UI, and serving the comparison JSON.
    """

    def __init__(self, *args, json_data=None, **kwargs):
        self.json_data = json_data
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            content = HTML_TEMPLATE.encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            content = json.dumps(self.json_data).encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path.startswith("/audio/"):
            audio_rel_path = urllib.parse.unquote(path[len("/audio/") :])
            audio_full_path = (BASE_DIR / audio_rel_path).resolve()

            try:
                audio_full_path.relative_to(BASE_DIR)
            except ValueError:
                self.send_error(403, "Forbidden")
                return

            if not audio_full_path.exists() or not audio_full_path.is_file():
                self.send_error(404, f"Audio file not found: {audio_rel_path}")
                return

            self.serve_audio_file(audio_full_path, head_only=False)
            return

        super().do_GET()

    def serve_audio_file(self, file_path: Path, head_only: bool = False):
        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range")

        content_type = "audio/wav"
        if file_path.suffix.lower() == ".mp3":
            content_type = "audio/mpeg"
        elif file_path.suffix.lower() == ".ogg":
            content_type = "audio/ogg"

        if range_header:
            try:
                ranges = range_header.replace("bytes=", "").split("-")
                start = int(ranges[0]) if ranges[0] else 0
                end = int(ranges[1]) if len(ranges) > 1 and ranges[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                if not head_only:
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                return
            except Exception as e:
                self.send_error(416, f"Requested Range Not Satisfiable: {e}")
                return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        if not head_only:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)


def find_free_port(start_port: int = 8080, max_attempts: int = 50) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def resolve_default_json_path() -> Path:
    for cand in DEFAULT_CANDIDATE_JSONS:
        if cand.exists():
            return cand
    return DEFAULT_CANDIDATE_JSONS[0]


def main():
    default_json = resolve_default_json_path()
    parser = argparse.ArgumentParser(
        description="Launch Cherokee Alignment 3-Way Comparative Inspector"
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=default_json,
        help=f"Path to alignment/comparison JSON file (default: {default_json})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve webview on (default: auto-select starting from 8080)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        default=None,
        help="Export self-contained static HTML file and exit without starting server",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: JSON file not found at {args.json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading comparison data from: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if args.export_html:
        static_html = HTML_TEMPLATE.replace(
            "const res = await fetch('/api/data');\n          globalData = await res.json();",
            f"globalData = {json.dumps(json_data)};",
        )
        args.export_html.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_html, "w", encoding="utf-8") as f:
            f.write(static_html)
        print(f"Exported static HTML to: {args.export_html}")
        return

    port = args.port if args.port is not None else find_free_port(8080)
    handler = lambda *h_args, **h_kwargs: RangeRequestHandler(  # noqa: E731
        *h_args, json_data=json_data, directory=str(BASE_DIR), **h_kwargs
    )

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}"
        total_count = (
            len(json_data)
            if isinstance(json_data, list)
            else json_data.get("total_verses", len(json_data.get("results", [])))
        )
        print("=" * 70)
        print(" Cherokee Alignment 3-Way Comparative Inspector")
        print("=" * 70)
        print(f" Total Verses:          {total_count}")
        print(f" Dataset Path:          {args.json_path}")
        print(f"\n Server running at:     {url}")
        print(" Press Ctrl+C to stop server.")
        print("=" * 70)

        if not args.no_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    main()
