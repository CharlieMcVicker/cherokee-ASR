#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_ctc_comparison.py

Lightweight webview server and UI to inspect CTC segmentation benchmark comparison data,
anomalous verses, flagged low-confidence words, and listen to verse and word-level audio snippets.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
import urllib.parse
import webbrowser

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JSON = (
    BASE_DIR / "runs" / "evaluation" / "ctc_segmentation_100_verses_comparison.json"
)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CTC Segmentation Anomaly & Alignment Viewer</title>
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
      font-size: 1.3rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-primary);
    }

    .stats-bar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .stat-badge {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.84rem;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      color: var(--text-secondary);
      user-select: none;
    }

    .stat-badge strong {
      color: var(--text-primary);
    }

    .stat-badge:hover {
      border-color: var(--accent);
      background: var(--surface-hover);
    }

    .stat-badge.active {
      border-color: var(--accent);
      background: var(--accent-glow);
      color: var(--accent);
    }

    .stat-badge.danger {
      border-color: var(--danger-border);
      background: var(--danger-bg);
      color: var(--danger);
    }

    .stat-badge.danger strong {
      color: #ff7b72;
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
      padding: 22px;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .verse-card:hover {
      border-color: #444c56;
    }

    .verse-card.has-anomaly {
      border-left: 5px solid var(--danger);
      background: linear-gradient(90deg, rgba(248, 81, 73, 0.03) 0%, var(--surface) 20%);
    }

    .verse-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid var(--surface-border);
      padding-bottom: 14px;
    }

    .verse-title-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .verse-id-badge {
      background: var(--tag-bg);
      border: 1px solid var(--surface-border);
      color: var(--accent);
      font-weight: 700;
      font-size: 0.95rem;
      padding: 5px 12px;
      border-radius: 6px;
      letter-spacing: 0.5px;
    }

    .verse-meta {
      font-size: 0.84rem;
      color: var(--text-muted);
    }

    .anomaly-badge {
      font-size: 0.8rem;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 20px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .anomaly-badge.danger {
      background: var(--danger-bg);
      color: #ff7b72;
      border: 1px solid var(--danger-border);
    }

    .anomaly-badge.success {
      background: var(--success-bg);
      color: #7ee787;
      border: 1px solid var(--success-border);
    }

    .player-container {
      display: flex;
      align-items: center;
      gap: 14px;
      background: var(--bg);
      padding: 10px 16px;
      border-radius: 8px;
      border: 1px solid var(--surface-border);
    }

    .play-btn {
      background: var(--accent);
      color: #0d1117;
      border: none;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.95rem;
      cursor: pointer;
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

    .text-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .syllabary-text {
      font-size: 1.45rem;
      font-weight: 500;
      color: #ffffff;
      line-height: 1.8;
      letter-spacing: 0.9px;
      font-family: "Plantagenet Cherokee", "Noto Sans Cherokee", "Apple Symbols", sans-serif;
    }

    .phonetic-text {
      font-size: 0.95rem;
      color: #c9d1d9;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      line-height: 1.5;
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

    .flagged-item-row .target-word {
      font-weight: 700;
      color: var(--text-primary);
    }

    .flagged-item-row .emitted-word-val {
      color: #ff7b72;
      font-weight: 700;
    }

    .details-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
      font-size: 0.84rem;
    }

    .detail-box {
      background: var(--bg);
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .detail-box .label {
      color: var(--text-muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      font-weight: 600;
    }

    .detail-box .val {
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
      color: var(--text-secondary);
      word-break: break-all;
      line-height: 1.4;
    }

    .diff-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
    }

    .diff-tag {
      background: var(--tag-bg);
      border: 1px solid var(--surface-border);
      color: #a5d6ff;
      font-size: 0.74rem;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
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
      color: var(--text-secondary);
      font-family: "SF Mono", Monaco, Menlo, Consolas, monospace;
    }

    .word-chip.flagged .emitted-word {
      color: #ff7b72;
      font-weight: 600;
    }

    .word-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.74rem;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 4px;
      margin-top: 2px;
    }

    .conf-badge {
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 0.7rem;
    }

    .conf-high {
      color: #7ee787;
      background: var(--success-bg);
    }

    .conf-mid {
      color: #d29922;
      background: var(--warning-bg);
    }

    .conf-low {
      color: #ff7b72;
      background: var(--danger-bg);
      font-weight: 700;
    }

    .time-badge {
      font-variant-numeric: tabular-nums;
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
          <span>📖</span> CTC Segmentation Benchmark & Anomaly Viewer
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
          <div class="stat-badge" id="stat-diff">
            Diff Rate: <strong id="val-diff">0%</strong>
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
          <button class="tab-btn" id="tab-clean" onclick="setFilter('clean')">✅ Clean Verses</button>
        </div>

        <select class="sort-select" id="sort-select" onchange="render()">
          <option value="anomalies-desc" selected>Sort: Flagged Words (Most First)</option>
          <option value="id-asc">Sort: Verse ID (Asc)</option>
          <option value="id-desc">Sort: Verse ID (Desc)</option>
          <option value="confidence-asc">Sort: Lowest Word Confidence</option>
          <option value="duration-desc">Sort: Longest Audio</option>
        </select>
      </div>
    </div>
  </header>

  <main id="verse-container">
    <div class="empty-state">Loading benchmark data...</div>
  </main>

  <script>
    let globalData = null;
    let currentFilter = 'anomalies'; // 'all', 'anomalies', 'clean'
    const audioPlayers = new Map(); // verse_id -> Audio instance
    let activeSnippetTimeout = null;

    async function init() {
      try {
        if (!globalData) {
          const res = await fetch('/api/data');
          globalData = await res.json();
        }
        
        document.getElementById('val-total').textContent = globalData.total_verses || (globalData.results ? globalData.results.length : 0);
        document.getElementById('val-anomalies').textContent = globalData.verses_with_anomalies || (globalData.anomalous_verses ? globalData.anomalous_verses.length : 0);
        document.getElementById('val-flagged').textContent = globalData.total_flagged_words || 0;
        document.getElementById('val-diff').textContent = (globalData.diff_percentage !== undefined ? globalData.diff_percentage.toFixed(1) : '100') + '%';

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
      document.getElementById('tab-clean').classList.toggle('active', filter === 'clean');

      document.getElementById('stat-total').classList.toggle('active', filter === 'all');
      document.getElementById('stat-anomalies').classList.toggle('active', filter === 'anomalies');
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
        if (currentFilter === 'anomalies' && !v.has_anomalies) return false;
        if (currentFilter === 'clean' && v.has_anomalies) return false;

        if (query) {
          const matchId = (v.verse_id || '').toLowerCase().includes(query);
          const matchSyllabary = (v.reference_syllabary || '').toLowerCase().includes(query);
          const matchPhonetic = (v.phonetic_citation || '').toLowerCase().includes(query);
          const matchDiff = (v.diff_details || []).some(d => d.toLowerCase().includes(query));
          const matchWords = (v.words_ctc_aligned || []).some(w => 
            (w.word || '').toLowerCase().includes(query) || ((w.emitted_word || '').toLowerCase().includes(query))
          );
          return matchId || matchSyllabary || matchPhonetic || matchDiff || matchWords;
        }
        return true;
      });

      filtered.sort((a, b) => {
        if (sortMode === 'anomalies-desc') {
          const aCount = (a.flagged_words || []).length;
          const bCount = (b.flagged_words || []).length;
          if (bCount !== aCount) return bCount - aCount;
          return (a.verse_id || '').localeCompare(b.verse_id || '');
        }
        if (sortMode === 'id-asc') return (a.verse_id || '').localeCompare(b.verse_id || '');
        if (sortMode === 'id-desc') return (b.verse_id || '').localeCompare(a.verse_id || '');
        if (sortMode === 'confidence-asc') {
          const getMinConf = (item) => {
            if (!item.words_ctc_aligned || item.words_ctc_aligned.length === 0) return 1.0;
            return Math.min(...item.words_ctc_aligned.map(w => w.confidence ?? 1.0));
          };
          return getMinConf(a) - getMinConf(b);
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
        const flaggedCount = (v.flagged_words || []).length;
        const diffCategories = v.diff_categories || [];

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
                <option value="0.75">0.75x</option>
                <option value="1" selected>1.0x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
              </select>
            </div>

            <div class="text-section">
              <div class="syllabary-text">${v.reference_syllabary || ''}</div>
              <div class="phonetic-text">${v.phonetic_citation || ''}</div>
            </div>

            ${hasAnomaly && flaggedCount > 0 ? `
              <div class="flagged-callout">
                <div class="flagged-callout-header">
                  <span>🚨</span> Flagged Low-Confidence Words / Anomalies (${flaggedCount})
                </div>
                <div class="flagged-items-grid">
                  ${(v.flagged_words || []).map(f => `
                    <div class="flagged-item-row">
                      <span>Target: <span class="target-word">${f.word}</span></span>
                      <span>→ Emitted: <span class="emitted-word-val">${f.emitted_word || f.word}</span></span>
                      <span class="conf-badge conf-low">Confidence: ${(f.confidence || 0).toFixed(6)}</span>
                      <span style="font-size: 0.76rem; color: var(--text-muted);">${(f.duration_sec || 0).toFixed(2)}s</span>
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : ''}

            <div class="details-grid">
              <div class="detail-box">
                <div class="label">New CTC Reconciled Hypothesis</div>
                <div class="val">${v.new_reconciled_phonetics || v.new_asr_hypothesis || '—'}</div>
              </div>
              <div class="detail-box">
                <div class="label">Diff Categories & Details</div>
                <div class="val">
                  <div class="diff-tags">
                    ${diffCategories.length > 0 
                      ? diffCategories.map(cat => `<span class="diff-tag">${cat}</span>`).join('')
                      : '<span style="color: var(--text-muted);">None</span>'
                    }
                  </div>
                </div>
              </div>
            </div>

            <div class="words-section">
              <div class="section-subtitle">
                <span>⚡</span> Aligned Words (${(v.words_ctc_aligned || []).length} words, click any word chip to play audio segment)
              </div>
              <div class="words-grid">
                ${(v.words_ctc_aligned || []).map((w) => {
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

    def __init__(self, *args, json_data: dict, **kwargs):
        self.json_data = json_data
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            content = HTML_TEMPLATE.encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            return

        if path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            content = json.dumps(self.json_data).encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
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

            self.serve_audio_file(audio_full_path, head_only=True)
            return

        super().do_HEAD()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

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


def main():
    parser = argparse.ArgumentParser(
        description="Launch CTC Segmentation Anomaly Webview Viewer"
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON,
        help=f"Path to comparison JSON file (default: {DEFAULT_JSON})",
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
        print("=" * 70)
        print(" CTC Segmentation Anomaly & Alignment Webview")
        print("=" * 70)
        print(
            f" Total Verses:          {json_data.get('total_verses', len(json_data.get('results', [])))}"
        )
        print(
            f" Verses with Anomalies: {json_data.get('verses_with_anomalies', len(json_data.get('anomalous_verses', [])))}"
        )
        print(f" Flagged Words:         {json_data.get('total_flagged_words', 0)}")
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
