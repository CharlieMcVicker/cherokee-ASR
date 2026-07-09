import { useState, useEffect, useRef, useMemo } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { openDB } from 'idb';

const initDB = async () => {
  return await openDB('cherokee-search-db', 2, {
    upgrade(db, oldVersion, newVersion, transaction) {
      let segStore;
      if (!db.objectStoreNames.contains('segments')) {
        segStore = db.createObjectStore('segments', { keyPath: 'id' });
      } else {
        segStore = transaction.objectStore('segments');
      }
      if (!segStore.indexNames.contains('by-csv')) {
        segStore.createIndex('by-csv', 'csv_file');
      }
      if (segStore.indexNames.contains('by-file')) {
        segStore.deleteIndex('by-file');
      }
      
      let fwStore;
      if (!db.objectStoreNames.contains('file_words')) {
        fwStore = db.createObjectStore('file_words', { keyPath: 'id' });
      } else {
        fwStore = transaction.objectStore('file_words');
      }
      if (!fwStore.indexNames.contains('by-file')) {
        fwStore.createIndex('by-file', 'file_path');
      }
    }
  });
};

const THEMES = {
  minimalLight: {
    id: "minimalLight",
    name: "Web 1.0 Light",
    description: "Classic Web 1.0 style.",
    bgApp: "bg-white text-black font-serif",
    bgMain: "bg-white",
    sidebar: "hidden", 
    topbar: "border-b-2 border-black bg-gray-100 p-2 mb-4 text-center relative",
    topbarTitle: "text-2xl font-bold text-black mb-2",
    topbarTabActive: "bg-white border-t border-l border-r border-black font-bold px-4 py-1 mx-1 inline-block",
    topbarTabInactive: "bg-gray-200 border border-gray-400 text-blue-600 underline px-4 py-1 mx-1 hover:bg-gray-300 inline-block",
    viewTitle: "text-2xl font-bold border-b border-gray-400 mb-4 pb-2",
    viewDesc: "text-sm mb-4",
    card: "border border-black p-4 mb-4 bg-white mx-auto",
    label: "font-bold text-sm block mb-1",
    input: "border border-black p-1 bg-white text-black mb-2 w-full",
    inputInfo: "border border-gray-400 bg-gray-100 p-2 text-sm mb-2",
    inputInfoSub: "text-gray-600 italic",
    inputInfoHighlight: "font-bold",
    checkbox: "mr-2",
    checkboxText: "text-sm",
    checkboxContainer: "mb-4",
    buttonPrimary: "border border-black bg-gray-200 text-black px-4 py-1 hover:bg-gray-300 cursor-pointer active:bg-gray-400 font-bold",
    buttonSecondary: "border border-black bg-gray-200 text-black px-4 py-1 hover:bg-gray-300 cursor-pointer active:bg-gray-400",
    successMsg: "border border-green-600 bg-green-100 text-green-800 p-2 font-bold mb-2",
    errorMsg: "border border-red-600 bg-red-100 text-red-800 p-2 font-bold mb-2",
    toggleIcon: "🌙"
  },
  minimalDark: {
    id: "minimalDark",
    name: "Web 1.0 Dark",
    description: "Classic Web 1.0 style Dark.",
    bgApp: "bg-black text-white font-serif",
    bgMain: "bg-black",
    sidebar: "hidden", 
    topbar: "border-b-2 border-white bg-gray-800 p-2 mb-4 text-center relative",
    topbarTitle: "text-2xl font-bold text-white mb-2",
    topbarTabActive: "bg-black border-t border-l border-r border-white font-bold px-4 py-1 mx-1 inline-block",
    topbarTabInactive: "bg-gray-700 border border-gray-500 text-blue-300 underline px-4 py-1 mx-1 hover:bg-gray-600 inline-block",
    viewTitle: "text-2xl font-bold border-b border-gray-500 mb-4 pb-2",
    viewDesc: "text-sm mb-4",
    card: "border border-white p-4 mb-4 bg-black mx-auto",
    label: "font-bold text-sm block mb-1",
    input: "border border-white p-1 bg-black text-white mb-2 w-full",
    inputInfo: "border border-gray-500 bg-gray-800 p-2 text-sm mb-2",
    inputInfoSub: "text-gray-400 italic",
    inputInfoHighlight: "font-bold",
    checkbox: "mr-2",
    checkboxText: "text-sm",
    checkboxContainer: "mb-4",
    buttonPrimary: "border border-white bg-gray-700 text-white px-4 py-1 hover:bg-gray-600 cursor-pointer active:bg-gray-500 font-bold",
    buttonSecondary: "border border-white bg-gray-700 text-white px-4 py-1 hover:bg-gray-600 cursor-pointer active:bg-gray-500",
    successMsg: "border border-green-400 bg-green-900 text-green-200 p-2 font-bold mb-2",
    errorMsg: "border border-red-400 bg-red-900 text-red-200 p-2 font-bold mb-2",
    toggleIcon: "☀️"
  }
};

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return "0s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${seconds.toFixed(1)}s`;
}

function SegmentHistogram({ preview, theme }) {
  const t = theme || {};
  const histogram = preview.histogram;
  if (!histogram || !Array.isArray(histogram) || histogram.length === 0) return null;
  const maxCount = Math.max(...histogram.map((b) => b.count), 0);

  return (
    <div className="mt-4 pt-4 border-t border-gray-300 dark:border-gray-700">
      <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2 text-left">Length Distribution (Seconds)</div>
      <div className="flex items-end h-32 gap-1 group">
        {histogram.map((bin, i) => {
          const heightPct = maxCount === 0 ? 0 : (bin.count / maxCount) * 100;
          return (
            <div key={i} className="relative flex-1 group/bar flex flex-col justify-end h-full">
              <div 
                className="w-full bg-gradient-to-t from-blue-600 to-blue-400 rounded-t-sm transition-all duration-300 group-hover:opacity-50 group-hover/bar:opacity-100 group-hover/bar:from-indigo-600 group-hover/bar:to-indigo-400 cursor-pointer"
                style={{ height: `${heightPct}%`, minHeight: bin.count > 0 ? '4px' : '0' }}
              ></div>
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover/bar:opacity-100 transition-opacity pointer-events-none z-10 w-max bg-gray-900 text-white text-xs rounded py-1 px-2 shadow-lg">
                <div className="font-bold">{bin.start.toFixed(1)}s - {bin.end.toFixed(1)}s</div>
                <div>{bin.count} segments</div>
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>{preview.min !== undefined ? preview.min.toFixed(1) : (histogram[0]?.start || 0).toFixed(1)}s</span>
        <span>{preview.max !== undefined ? preview.max.toFixed(1) : (histogram[histogram.length-1]?.end || 0).toFixed(1)}s</span>
      </div>
    </div>
  );
}

function FileSegmentCard({ preview, theme, onSettingChange, onResetSetting, onSmartSettingsChange }) {
  const t = theme;
  const [showSettings, setShowSettings] = useState(false);
  const [isSmartLoading, setIsSmartLoading] = useState(false);
  const settings = preview.settings || { silence_thresh: -40, min_silence_len: 500, keep_silence: 100 };

  const handleSmartSeg = async () => {
    setIsSmartLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/smart_segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: preview.file })
      });
      if (res.ok) {
        const data = await res.json();
        onSmartSettingsChange(preview.file, data);
        setShowSettings(true);
      }
    } catch (e) {
      console.error("Smart Segment Error:", e);
    }
    setIsSmartLoading(false);
  };

  return (
    <div className={`p-4 border ${t.card} text-left mb-4 rounded shadow-sm relative`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b border-gray-300 dark:border-gray-700">
        <div>
          <h4 className="font-bold text-base break-all">{preview.filename}</h4>
          <span className="text-xs opacity-75 font-mono">{preview.file} ({formatDuration(preview.total_duration)})</span>
        </div>
        <div className="flex gap-2">
          <button 
            type="button"
            onClick={handleSmartSeg}
            disabled={isSmartLoading}
            className={`text-xs px-3 py-1 border ${t.buttonPrimary} flex items-center gap-1 bg-indigo-100 hover:bg-indigo-200 text-indigo-900 border-indigo-300 dark:bg-indigo-900 dark:text-indigo-100 dark:border-indigo-700 disabled:opacity-50`}
          >
            {isSmartLoading ? "⏳ Computing..." : "✨ Smart Seg"}
          </button>
          <button 
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className={`text-xs px-3 py-1 border ${t.buttonSecondary} flex items-center gap-1`}
          >
            ⚙️ {showSettings ? "Hide Sliders" : "Custom Sliders"}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 mb-4 mt-2">
        <div className="flex-1 bg-gradient-to-br from-blue-50 to-blue-100 p-3 rounded-lg border border-blue-200 text-center min-w-[120px]">
          <div className="text-xs text-blue-500 font-bold uppercase tracking-wider">Segments</div>
          <div className="text-2xl font-black text-blue-900">{preview.segment_count}</div>
        </div>
        <div className="flex-1 bg-gradient-to-br from-indigo-50 to-indigo-100 p-3 rounded-lg border border-indigo-200 text-center min-w-[120px]">
          <div className="text-xs text-indigo-500 font-bold uppercase tracking-wider">Result Audio</div>
          <div className="text-2xl font-black text-indigo-900">{formatDuration(preview.result_duration)}</div>
        </div>
        <div className="flex-1 bg-gradient-to-br from-green-50 to-green-100 p-3 rounded-lg border border-green-200 text-center min-w-[120px]">
          <div className="text-xs text-green-600 font-bold uppercase tracking-wider">Coverage</div>
          <div className="text-2xl font-black text-green-900">{preview.coverage_percent}%</div>
        </div>
        <div className="flex-1 bg-gradient-to-br from-amber-50 to-amber-100 p-3 rounded-lg border border-amber-200 text-center min-w-[120px]">
          <div className="text-xs text-amber-600 font-bold uppercase tracking-wider">Overlap</div>
          <div className="text-2xl font-black text-amber-900">{preview.overlap_duration}s <span className="text-sm font-bold opacity-75">({preview.overlap_percent}%)</span></div>
        </div>
      </div>

      <SegmentHistogram preview={preview} theme={t} />

      {showSettings && (
        <div className={`mt-4 p-3 border rounded ${t.inputInfo} text-xs`}>
          <div className="flex justify-between items-center mb-2 font-bold">
            <span>File-Specific Settings:</span>
            <button 
              type="button" 
              onClick={() => onResetSetting(preview.file)}
              className="text-blue-600 dark:text-blue-400 underline hover:text-blue-800"
            >
              Reset to Batch Defaults
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="block font-medium mb-1">Silence Thresh: {settings.silence_thresh} dBFS</label>
              <input 
                type="range" min="-80" max="0" step="1" 
                value={settings.silence_thresh} 
                onChange={(e) => onSettingChange(preview.file, 'silence_thresh', parseInt(e.target.value) || 0)} 
                className="w-full"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Min Silence: {settings.min_silence_len} ms</label>
              <input 
                type="range" min="10" max="1500" step="10" 
                value={settings.min_silence_len} 
                onChange={(e) => onSettingChange(preview.file, 'min_silence_len', parseInt(e.target.value) || 0)} 
                className="w-full"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Keep Silence: {settings.keep_silence} ms</label>
              <input 
                type="range" min="0" max="500" step="10" 
                value={settings.keep_silence} 
                onChange={(e) => onSettingChange(preview.file, 'keep_silence', parseInt(e.target.value) || 0)} 
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function View0({ theme }) {
  const t = theme;
  const [files, setFiles] = useState({ wav_files: [], folders: [] });
  const [targetPath, setTargetPath] = useState("");
  const [globalSettings, setGlobalSettings] = useState({ silence_thresh: -40, min_silence_len: 500, keep_silence: 100 });
  const [fileSettings, setFileSettings] = useState({});
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewsLoading, setPreviewsLoading] = useState(false);
  const [previews, setPreviews] = useState([]);
  const [summary, setSummary] = useState(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => {
        const allWavs = data.wav_files || [];
        const folderSet = new Set();
        allWavs.forEach(f => {
          const parts = f.split('/');
          for (let i = 1; i < parts.length; i++) {
            folderSet.add(parts.slice(0, i).join('/'));
          }
        });
        setFiles({ wav_files: allWavs, folders: Array.from(folderSet).sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b)) });
      }).catch(err => console.log("Failed to fetch files", err));
  }, []);

  useEffect(() => {
    if (!targetPath) {
      setPreviews([]);
      setSummary(null);
      return;
    }
    const fetchPreviews = async () => {
      setPreviewsLoading(true);
      try {
        const res = await fetch("http://localhost:8000/api/preview_segments", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            target_path: targetPath, 
            ...globalSettings,
            file_settings: fileSettings
          })
        });
        const data = await res.json();
        if (res.ok) {
          setPreviews(data.previews || []);
          setSummary(data.summary || null);
        }
      } catch (e) {
        console.log("Failed to fetch previews", e);
      } finally {
        setPreviewsLoading(false);
      }
    };
    const timer = setTimeout(fetchPreviews, 400);
    return () => clearTimeout(timer);
  }, [targetPath, globalSettings, fileSettings]);

  const handlePerFileSettingChange = (filePath, key, value) => {
    setFileSettings(prev => ({
      ...prev,
      [filePath]: {
        ...(prev[filePath] || globalSettings),
        [key]: value
      }
    }));
  };

  const handleSmartSettingsChange = (filePath, newSettings) => {
    setFileSettings(prev => ({
      ...prev,
      [filePath]: {
        ...(prev[filePath] || globalSettings),
        ...newSettings
      }
    }));
  };

  const handleResetFileSetting = (filePath) => {
    setFileSettings(prev => {
      const next = { ...prev };
      delete next[filePath];
      return next;
    });
  };

  const handleResetAllFileSettings = () => {
    setFileSettings({});
  };

  const handleSegment = async () => {
    if (!targetPath) return;
    setLoading(true);
    setStatus("⏳ Processing batch segmentation...");
    try {
      const res = await fetch("http://localhost:8000/api/batch_segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          target_path: targetPath, 
          ...globalSettings,
          file_settings: fileSettings
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStatus("✅ " + data.message + " Saved to " + data.output_dir);
      } else {
        setStatus("❌ Error: " + data.detail);
      }
    } catch (e) {
      setStatus("❌ Error: " + e.message);
    }
    setLoading(false);
  };

  const filteredPreviews = useMemo(() => {
    if (!searchQuery.trim()) return previews;
    const q = searchQuery.toLowerCase();
    return previews.filter(p => p.file.toLowerCase().includes(q) || p.filename.toLowerCase().includes(q));
  }, [previews, searchQuery]);

  const totalPages = Math.ceil(filteredPreviews.length / itemsPerPage) || 1;
  const currentPreviews = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredPreviews.slice(start, start + itemsPerPage);
  }, [filteredPreviews, currentPage]);

  return (
    <div className="max-w-4xl mx-auto text-center">
      <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>0. Batch Segmentation</h2>
      <p className={`${t.viewDesc} mb-8`}>Select a single file or a folder to automatically cut up all audio files based on silence thresholds.</p>

      <div className={`${t.card} p-6`}>
        <div className="mb-6 text-left">
          <label className={`block text-sm font-medium ${t.label} mb-2`}>Target File or Folder</label>
          <select className={`w-full p-3 ${t.input}`} value={targetPath} onChange={e => { setTargetPath(e.target.value); setCurrentPage(1); setFileSettings({}); }}>
            <option value="">-- Select File or Folder --</option>
            {files.folders.length > 0 && <optgroup label="Folders">{files.folders.map(f => <option key={f} value={f}>{f}</option>)}</optgroup>}
            {files.wav_files.length > 0 && <optgroup label="Files">{files.wav_files.map(f => <option key={f} value={f}>{f}</option>)}</optgroup>}
          </select>
        </div>

        <div className={`p-4 border ${t.inputInfo} text-left mb-6 rounded`}>
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-bold text-sm">Global Batch Defaults</h3>
            {Object.keys(fileSettings).length > 0 && (
              <button 
                type="button" 
                onClick={handleResetAllFileSettings}
                className="text-xs text-blue-600 dark:text-blue-400 underline"
              >
                Reset All Files to Global Defaults ({Object.keys(fileSettings).length} customized)
              </button>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className={`block text-sm font-medium ${t.label} mb-2`}>Silence Threshold (dBFS)</label>
              <div className="flex items-center gap-2">
                <input type="range" min="-80" max="0" step="1" value={globalSettings.silence_thresh} onChange={(e) => setGlobalSettings({...globalSettings, silence_thresh: parseInt(e.target.value) || 0})} className="w-full" />
                <input type="number" min="-80" max="0" value={globalSettings.silence_thresh} onChange={(e) => setGlobalSettings({...globalSettings, silence_thresh: parseInt(e.target.value) || 0})} className={`w-16 p-1 text-sm ${t.input} mb-0`} />
              </div>
            </div>
            <div>
              <label className={`block text-sm font-medium ${t.label} mb-2`}>Min Silence (ms)</label>
              <div className="flex items-center gap-2">
                <input type="range" min="10" max="1500" step="10" value={globalSettings.min_silence_len} onChange={(e) => setGlobalSettings({...globalSettings, min_silence_len: parseInt(e.target.value) || 0})} className="w-full" />
                <input type="number" min="10" max="1500" value={globalSettings.min_silence_len} onChange={(e) => setGlobalSettings({...globalSettings, min_silence_len: parseInt(e.target.value) || 0})} className={`w-20 p-1 text-sm ${t.input} mb-0`} />
              </div>
            </div>
            <div>
              <label className={`block text-sm font-medium ${t.label} mb-2`}>Keep Silence (ms)</label>
              <div className="flex items-center gap-2">
                <input type="range" min="0" max="500" step="10" value={globalSettings.keep_silence} onChange={(e) => setGlobalSettings({...globalSettings, keep_silence: parseInt(e.target.value) || 0})} className="w-full" />
                <input type="number" min="0" max="500" value={globalSettings.keep_silence} onChange={(e) => setGlobalSettings({...globalSettings, keep_silence: parseInt(e.target.value) || 0})} className={`w-20 p-1 text-sm ${t.input} mb-0`} />
              </div>
            </div>
          </div>
        </div>

        {previewsLoading && (
          <div className={`flex items-center justify-center gap-3 p-4 mb-6 border ${t.inputInfo} font-bold animate-pulse text-blue-700 dark:text-blue-300`}>
            <svg className="animate-spin h-5 w-5 text-blue-600 dark:text-blue-300" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Calculating segment statistics for audio files...</span>
          </div>
        )}

        {summary && (
          <div className={`mb-6 p-4 border ${t.inputInfo} rounded text-left`}>
            <h3 className="font-bold text-base mb-3 border-b pb-1 border-gray-400">Batch Overview</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center">
              <div>
                <div className="text-xs font-semibold opacity-75">Files</div>
                <div className="text-lg font-bold">{summary.total_files}</div>
              </div>
              <div>
                <div className="text-xs font-semibold opacity-75">Total Duration</div>
                <div className="text-lg font-bold">{formatDuration(summary.total_batch_duration)}</div>
              </div>
              <div>
                <div className="text-xs font-semibold opacity-75">Total Segments</div>
                <div className="text-lg font-bold">{summary.total_segments}</div>
              </div>
              <div>
                <div className="text-xs font-semibold opacity-75">Result Audio</div>
                <div className="text-lg font-bold">{formatDuration(summary.total_result_duration)}</div>
              </div>
              <div>
                <div className="text-xs font-semibold opacity-75">Overall Coverage</div>
                <div className="text-lg font-bold text-green-700 dark:text-green-400">{summary.overall_coverage_percent}%</div>
              </div>
            </div>
          </div>
        )}

        {previews.length > 0 && (
          <div className="mb-6 text-left">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <h3 className="font-bold text-lg">All Audio Files ({filteredPreviews.length})</h3>
              {previews.length > itemsPerPage && (
                <input 
                  type="text" 
                  placeholder="Filter files by name..." 
                  value={searchQuery}
                  onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                  className={`p-1 px-3 text-sm border ${t.input} w-60 mb-0`}
                />
              )}
            </div>

            {currentPreviews.map((p) => (
              <FileSegmentCard 
                key={p.file} 
                preview={p} 
                theme={t} 
                onSettingChange={handlePerFileSettingChange}
                onResetSetting={handleResetFileSetting}
                onSmartSettingsChange={handleSmartSettingsChange}
              />
            ))}

            {totalPages > 1 && (
              <div className={`flex items-center justify-between mt-4 p-2 border ${t.inputInfo} rounded`}>
                <button 
                  type="button"
                  disabled={currentPage <= 1}
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  className={`px-3 py-1 text-sm border ${t.buttonSecondary} disabled:opacity-40`}
                >
                  ◀ Previous
                </button>
                <span className="text-xs font-semibold">Page {currentPage} of {totalPages} ({filteredPreviews.length} total)</span>
                <button 
                  type="button"
                  disabled={currentPage >= totalPages}
                  onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                  className={`px-3 py-1 text-sm border ${t.buttonSecondary} disabled:opacity-40`}
                >
                  Next ▶
                </button>
              </div>
            )}
          </div>
        )}

        <button 
          onClick={handleSegment} 
          disabled={loading || !targetPath}
          className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
        >
          {loading ? "⏳ Segmenting..." : "Start Batch Segmentation"}
        </button>

        {status && (
          <div className={`mt-6 p-4 border text-left ${status.includes('❌') ? t.errorMsg : (status.includes('✅') ? t.successMsg : t.inputInfo)}`}>
            {status}
          </div>
        )}
      </div>
    </div>
  );
}



function View1({ theme }) {
  const t = theme;
  const [files, setFiles] = useState({ wav_files: [], folders: [] });
  const [parentFolder, setParentFolder] = useState("");
  const [groupedFiles, setGroupedFiles] = useState({});
  const [selectedSubfolders, setSelectedSubfolders] = useState({});
  const [form, setForm] = useState({ 
    checkpoint: "",
    output_csv_name: "batch_inference_results.csv"
  });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchStats, setBatchStats] = useState(null);
  const [isFetchingStats, setIsFetchingStats] = useState(false);
  const [defaultCheckpoint, setDefaultCheckpoint] = useState("charliemcvicker/asr-cherokee");

  useEffect(() => {
    fetch("http://localhost:8000/api/config/best_model")
      .then(res => res.json())
      .then(data => {
        if (data.repo) setDefaultCheckpoint(data.repo);
      }).catch(err => console.log("Failed to fetch best model config", err));
  }, []);

  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => {
        const allWavs = data.wav_files || [];
        const folderSet = new Set();
        allWavs.forEach(f => {
          const parts = f.split('/');
          for (let i = 1; i < parts.length; i++) {
            folderSet.add(parts.slice(0, i).join('/'));
          }
        });
        setFiles({ wav_files: allWavs, folders: Array.from(folderSet).sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b)) });
      }).catch(err => console.log("Failed to fetch files", err));
  }, []);

  useEffect(() => {
    if (!parentFolder) {
      setGroupedFiles({});
      setSelectedSubfolders({});
      return;
    }
    
    const wavsInParent = files.wav_files.filter(f => f.startsWith(parentFolder + '/'));
    const groups = {};
    
    wavsInParent.forEach(f => {
      const relPath = f.substring(parentFolder.length + 1);
      const parts = relPath.split('/');
      
      let subfolder = "/ (root)";
      let targetPath = parentFolder;
      
      if (parts.length > 1) {
        subfolder = parts.slice(0, -1).join('/');
        targetPath = parentFolder + '/' + subfolder;
      }
      
      if (!groups[targetPath]) {
        groups[targetPath] = { name: subfolder, files: [] };
      }
      groups[targetPath].files.push(f);
    });
    
    setGroupedFiles(groups);
    
    const sel = {};
    Object.keys(groups).forEach(g => sel[g] = true);
    setSelectedSubfolders(sel);
  }, [parentFolder, files.wav_files]);

  const toggleSubfolder = (sf) => {
    setSelectedSubfolders(prev => ({...prev, [sf]: !prev[sf]}));
  };

  const toggleAll = (state) => {
    const sel = {};
    Object.keys(groupedFiles).forEach(c => sel[c] = state);
    setSelectedSubfolders(sel);
  };

  const targetFolders = Object.keys(selectedSubfolders).filter(k => selectedSubfolders[k]);

  useEffect(() => {
    if (targetFolders.length === 0) {
      setBatchStats(null);
      return;
    }
    const timeout = setTimeout(() => {
        setIsFetchingStats(true);
        fetch("http://localhost:8000/api/batch_stats", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_folders: targetFolders })
        })
        .then(res => res.json())
        .then(data => {
            setBatchStats(data);
            setIsFetchingStats(false);
        })
        .catch(err => {
            console.error("Failed to fetch stats", err);
            setIsFetchingStats(false);
        });
    }, 500);
    return () => clearTimeout(timeout);
  }, [selectedSubfolders]);

  const handleInference = async () => {
    if (targetFolders.length === 0) return;
    
    setLoading(true);
    setStatus(`⏳ Running batch inference on ${targetFolders.length} folders... (This may take several minutes)`);
    try {
      const res = await fetch("http://localhost:8000/api/batch_inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_folders: targetFolders,
          checkpoint: form.checkpoint,
          output_csv_name: form.output_csv_name
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStatus("✅ " + data.message + " Output saved to " + data.csv_path);
      } else {
        setStatus("❌ Error: " + data.detail);
      }
    } catch (e) {
      setStatus("❌ Error: " + e.message);
    }
    setLoading(false);
  };

  const groupKeys = Object.keys(groupedFiles);
  const selectedCount = Object.values(selectedSubfolders).filter(Boolean).length;

  return (
    <div className="max-w-3xl mx-auto text-center ">
      <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>1. Batch Inference</h2>
      <p className={`${t.viewDesc} mb-8`}>Run speech-to-text inference on the segmented audio folders.</p>
      
      <div className={`${t.card} p-6 `}>
        <div className="grid grid-cols-1 gap-6 mb-6 text-left">
          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Target Parent Directory</label>
            <select className={`w-full p-3 ${t.input}`} value={parentFolder} onChange={e => setParentFolder(e.target.value)}>
              <option value="">-- Select Parent Folder --</option>
              {files.folders.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          {groupKeys.length > 0 && (
            <div className={`p-4 border ${t.inputInfo}`}>
              <div className="flex justify-between items-center mb-2">
                <label className={`block text-sm font-medium ${t.label}`}>Select Folders to Process ({selectedCount} / {groupKeys.length})</label>
                <div>
                  <button onClick={() => toggleAll(true)} className={`text-xs mr-2 ${t.buttonSecondary}`}>Select All</button>
                  <button onClick={() => toggleAll(false)} className={`text-xs ${t.buttonSecondary}`}>Deselect All</button>
                </div>
              </div>
              <div className="max-h-64 overflow-y-auto border border-gray-300 p-2 bg-white text-sm">
                {groupKeys.map(k => (
                  <div key={k} className="mb-2">
                    <label className="flex items-center space-x-2 p-1 hover:bg-gray-100 cursor-pointer text-black font-bold border-b border-gray-200">
                      <input type="checkbox" checked={!!selectedSubfolders[k]} onChange={() => toggleSubfolder(k)} />
                      <span className="truncate" title={k}>{groupedFiles[k].name} ({groupedFiles[k].files.length} files)</span>
                    </label>
                    {groupedFiles[k].files.slice(0, 3).map(f => (
                      <div key={f} className="text-xs text-gray-500 pl-6 py-1 truncate">{f.split('/').pop()}</div>
                    ))}
                    {groupedFiles[k].files.length > 3 && (
                      <div className="text-xs text-gray-400 pl-6 italic">...and {groupedFiles[k].files.length - 3} more files</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stats Dashboard */}
          {targetFolders.length > 0 && (
            <div className={`p-5 rounded-xl border border-gray-200 shadow-sm overflow-hidden transition-all duration-300 ${t.card}`}>
              <h3 className={`text-lg font-bold mb-4 flex items-center ${t.viewTitle}`}>
                📊 Audio Statistics
                {isFetchingStats && <span className="ml-3 text-sm font-normal text-blue-500 animate-pulse">Calculating...</span>}
              </h3>
              
              {!isFetchingStats && batchStats && batchStats.count > 0 && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
                  {/* Summary Pills */}
                  <div className="flex flex-wrap gap-3">
                    <div className="flex-1 bg-gradient-to-br from-blue-50 to-blue-100 p-3 rounded-lg border border-blue-200">
                      <div className="text-xs text-blue-500 font-bold uppercase tracking-wider">Total Files</div>
                      <div className="text-2xl font-black text-blue-900">{batchStats.count.toLocaleString()}</div>
                    </div>
                    <div className="flex-1 bg-gradient-to-br from-indigo-50 to-indigo-100 p-3 rounded-lg border border-indigo-200">
                      <div className="text-xs text-indigo-500 font-bold uppercase tracking-wider">Total Duration</div>
                      <div className="text-2xl font-black text-indigo-900">
                        {batchStats.total_duration > 3600 
                          ? (batchStats.total_duration / 3600).toFixed(1) + " hrs" 
                          : (batchStats.total_duration / 60).toFixed(1) + " mins"}
                      </div>
                    </div>
                    <div className="flex-1 bg-gradient-to-br from-purple-50 to-purple-100 p-3 rounded-lg border border-purple-200">
                      <div className="text-xs text-purple-500 font-bold uppercase tracking-wider">Length Range</div>
                      <div className="text-xl font-black text-purple-900 mt-1">
                        {batchStats.min.toFixed(1)}s - {batchStats.max.toFixed(1)}s
                      </div>
                    </div>
                  </div>

                  {/* Histogram Chart */}
                  <div>
                    <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2">Length Distribution (Seconds)</div>
                    <div className="flex items-end h-32 gap-1 group">
                      {batchStats.histogram.map((bin, i) => {
                        const maxCount = Math.max(...batchStats.histogram.map(b => b.count));
                        const heightPct = maxCount === 0 ? 0 : (bin.count / maxCount) * 100;
                        return (
                          <div key={i} className="relative flex-1 group/bar flex flex-col justify-end h-full">
                            <div 
                              className="w-full bg-gradient-to-t from-blue-600 to-blue-400 rounded-t-sm transition-all duration-300 group-hover:opacity-50 group-hover/bar:opacity-100 group-hover/bar:from-indigo-600 group-hover/bar:to-indigo-400 cursor-pointer"
                              style={{ height: `${heightPct}%`, minHeight: bin.count > 0 ? '4px' : '0' }}
                            ></div>
                            {/* Tooltip */}
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover/bar:opacity-100 transition-opacity pointer-events-none z-10 w-max bg-gray-900 text-white text-xs rounded py-1 px-2 shadow-lg">
                              <div className="font-bold">{bin.start.toFixed(1)}s - {bin.end.toFixed(1)}s</div>
                              <div>{bin.count} files</div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>{batchStats.min.toFixed(1)}s</span>
                      <span>{batchStats.max.toFixed(1)}s</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Model Checkpoint (Leave empty to use best_model.json: {defaultCheckpoint})</label>
            <input type="text" className={`w-full p-3 ${t.input}`} value={form.checkpoint} onChange={e => setForm({...form, checkpoint: e.target.value})} placeholder={defaultCheckpoint} />
          </div>

          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Output CSV Name</label>
            <input type="text" className={`w-full p-3 ${t.input}`} value={form.output_csv_name} onChange={e => setForm({...form, output_csv_name: e.target.value})} placeholder="e.g. batch_inference_results.csv" />
          </div>
        </div>

        <button 
          onClick={handleInference} 
          disabled={loading || selectedCount === 0}
          className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
        >
          {loading ? "⏳ Running Inference..." : `Start Batch Inference on ${selectedCount} Folders`}
        </button>

        {status && (
          <div className={`mt-6 p-4 border text-left ${status.includes('❌') ? t.errorMsg : (status.includes('✅') ? t.successMsg : t.inputInfo)}`}>
            {status}
          </div>
        )}
      </div>
    </div>
  );
}


function View2({ theme }) {
 const t = theme;
 const [segments, setSegments] = useState([]);
 const [currentIndex, setCurrentIndex] = useState(-1);
 const [searchVal, setSearchVal] = useState("");
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);
 const [labelInput, setLabelInput] = useState("");
 const [csvFiles, setCsvFiles] = useState([]);
 const [selectedCsv, setSelectedCsv] = useState("");
 const [showTones, setShowTones] = useState(true);
 const [visibleCount, setVisibleCount] = useState(200);

 useEffect(() => {
   setVisibleCount(200);
 }, [searchVal, selectedCsv]);

  const formatText = (txt) => {
    if (!txt) return txt;
    return showTones ? txt : txt.replace(/[0-9]/g, '').replace(/([aeiouvAEIOUV])\1+/g, '$1');
  };

  const getProcessedWordConfidences = (segment) => {
    let wordConfs = segment.word_confidences;
    if (!wordConfs || wordConfs.length === 0) return [];

    const greedyWords = (segment.greedy_transcription || '').trim().split(/\s+/).filter(Boolean);
    
    if (wordConfs.length === 1 && greedyWords.length > 1) {
      const allChars = wordConfs[0].chars || [];
      const newWords = [];
      let charIdx = 0;

      for (const targetWord of greedyWords) {
        const wordChars = [];
        for (let i = 0; i < targetWord.length && charIdx < allChars.length; i++) {
          wordChars.push(allChars[charIdx]);
          charIdx++;
        }
        if (wordChars.length > 0) {
          const avgConf = wordChars.reduce((sum, c) => sum + (c.confidence || 0), 0) / wordChars.length;
          newWords.push({
            word: targetWord,
            confidence: avgConf,
            chars: wordChars
          });
        }
      }
      if (charIdx < allChars.length) {
        const remainingChars = allChars.slice(charIdx);
        const avgConf = remainingChars.reduce((sum, c) => sum + (c.confidence || 0), 0) / remainingChars.length;
        newWords.push({
          word: remainingChars.map(c => c.char).join(''),
          confidence: avgConf,
          chars: remainingChars
        });
      }
      return newWords;
    }

    return wordConfs;
  };

  const renderDetailedTranscription = (segment) => {
    const wordConfs = getProcessedWordConfidences(segment);
    if (wordConfs && wordConfs.length > 0) {
        return (
            <div className="flex flex-wrap gap-x-4 gap-y-4 items-end py-1">
                {wordConfs.map((wordObj, i) => {
                    const formattedWord = formatText(wordObj.word);
                    if (!formattedWord) return null;

                    const getWordConfColor = (conf) => {
                      if (conf < 0.8) return 'text-red-700 bg-red-100 border-red-300';
                      if (conf < 0.95) return 'text-orange-700 bg-orange-100 border-orange-300';
                      return 'text-green-800 bg-green-100 border-green-300';
                    };

                    return (
                        <div 
                            key={i} 
                            className="inline-flex flex-col items-center group relative cursor-pointer pt-4 pb-1 px-2 rounded-md bg-gray-50 hover:bg-gray-100 border border-gray-300/80 shadow-sm transition-all"
                            title={`Word: "${wordObj.word}"\nConfidence: ${(wordObj.confidence * 100).toFixed(1)}%`}
                        >
                            {/* Characters Row with 2nd best prediction hint above low/mid conf chars */}
                            <div className="flex items-end leading-none">
                                {wordObj.chars.map((charObj, j) => {
                                    const formattedChar = formatText(charObj.char);
                                    if (!formattedChar) return null;

                                    const isLowMidConf = charObj.confidence < 0.8;
                                    const topAlt = (charObj.alternatives && charObj.alternatives.length > 0) ? charObj.alternatives[0] : null;
                                    const formattedAltChar = topAlt ? formatText(topAlt.char) : null;

                                    // Build Top 5 alternatives list for character tooltip
                                    const allPredictions = [
                                      { char: charObj.char, confidence: charObj.confidence },
                                      ...(charObj.alternatives || [])
                                    ].slice(0, 5);

                                    const charTooltip = `Character: '${charObj.char}' (${(charObj.confidence * 100).toFixed(1)}%)\n\nTop 5 Predictions:\n` +
                                      allPredictions.map((p, idx) => `${idx + 1}. '${p.char}' — ${(p.confidence * 100).toFixed(1)}%`).join('\n');

                                    return (
                                        <div key={j} className="relative flex flex-col items-center px-[0.5px]">
                                            {/* 2nd best prediction floating in small text above character */}
                                            {isLowMidConf && formattedAltChar && (
                                                <span 
                                                    className="text-[10px] font-mono font-bold leading-none text-blue-600 absolute -top-3.5 opacity-80 hover:opacity-100 select-none"
                                                    title={`2nd Choice: '${topAlt.char}' (${(topAlt.confidence * 100).toFixed(1)}%)`}
                                                >
                                                    {formattedAltChar}
                                                </span>
                                            )}

                                            {/* Primary character */}
                                            <span 
                                                className={`
                                                    text-xl font-medium leading-none px-0.5 rounded-sm transition-colors duration-150 inline-block
                                                    ${charObj.confidence < 0.5 ? 'text-red-600 font-bold bg-red-100' : charObj.confidence < 0.8 ? 'text-orange-600 font-semibold bg-orange-100' : 'text-gray-900'}
                                                    hover:bg-blue-500/30 hover:text-blue-900
                                                `}
                                                title={charTooltip}
                                            >
                                                {formattedChar}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Word Confidence display directly below the word */}
                            <span 
                                className={`text-[11px] font-mono font-bold leading-tight mt-1.5 px-1.5 py-0.5 rounded border ${getWordConfColor(wordObj.confidence)}`}
                                title={`Word Confidence: ${(wordObj.confidence * 100).toFixed(2)}%`}
                            >
                                {(wordObj.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                    );
                })}
            </div>
        );
    }
    return formatText(segment.greedy_transcription) || <i className="text-gray-400">(Empty)</i>;
  };

 useEffect(() => {
   fetch("http://localhost:8000/api/files")
     .then(res => res.json())
     .then(data => {
       setCsvFiles(data.csv_files || []);
     })
     .catch(err => console.log("Failed to fetch files", err));
 }, []);

 const loadData = async (file) => {
   const fileToLoad = file || selectedCsv;
   if (!fileToLoad) return;
   setLoading(true);
   try {
     const res = await fetch(`http://localhost:8000/api/labeler/data?file=${encodeURIComponent(fileToLoad)}`);
     const result = await res.json();
     if (res.ok && result.status === 'success') {
       setSegments(result.data);
     } else {
       setStatus(`❌ Error: ${result.detail || result.message}`);
     }
   } catch (err) {
     setStatus(`❌ Error fetching data: ${err.message}`);
   }
   setLoading(false);
 };

 const selectSegment = (index) => {
   setCurrentIndex(index);
   const seg = segments[index];
   if (seg) {
     setLabelInput(formatText(seg.labeled_sentence || seg.greedy_transcription));
   }
 };

 const submitLabel = () => {
   if (currentIndex !== -1 && segments[currentIndex]) {
     const newSegments = [...segments];
     newSegments[currentIndex].labeled_sentence = labelInput.trim();
     setSegments(newSegments);
     setStatus(`✅ Label stored locally for segment ${currentIndex}`);
     
     // Advance
     navigate(1);
   }
 };

 const navigate = (direction) => {
   const nextIdx = currentIndex + direction;
   if (nextIdx >= 0 && nextIdx < segments.length) {
     selectSegment(nextIdx);
   } else {
     setCurrentIndex(-1);
   }
 };

 const saveAllLabels = async () => {
   const labeled = segments
     .filter(seg => !!seg.labeled_sentence)
     .map(seg => ({
       path: seg.file_path,
       sentence: seg.labeled_sentence
     }));

   setLoading(true);
   setStatus("Saving labels to CSV...");
   try {
     const res = await fetch("http://localhost:8000/api/labeler/save", {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ labels: labeled })
     });
     const result = await res.json();
     if (res.ok && result.status === 'success') {
       setStatus(`✅ Successfully saved ${labeled.length} labels to CSV.`);
     } else {
       setStatus(`❌ Error: ${result.detail || result.message}`);
     }
   } catch (err) {
     setStatus(`❌ Error saving data: ${err.message}`);
   }
   setLoading(false);
 };

 const getConfidenceClass = (conf) => {
   if (conf < 0.8) return "text-red-600 font-bold";
   if (conf < 0.95) return "text-orange-500 font-bold";
   return "text-green-600 font-bold";
 };

 const filteredSegments = segments.filter(seg => 
   !searchVal || seg.greedy_transcription.toLowerCase().includes(searchVal.toLowerCase())
 );

 return (
   <div className="max-w-6xl mx-auto text-center flex flex-col md:flex-row gap-6 text-left">
     {/* Sidebar */}
     <div className={`md:w-1/3 flex flex-col ${t.card} p-4 h-[80vh]`}>
       <h2 className={`text-xl font-bold mb-2`}>Review</h2>
       
       <div className="mb-4">
         <label className={`block text-sm font-medium ${t.label} mb-1`}>Results CSV File</label>
          <select 
            className={`w-full p-3 ${t.input}`} 
            value={selectedCsv} 
            onChange={e => {
              setSelectedCsv(e.target.value);
              loadData(e.target.value);
            }}
          >
            <option value="">-- Select CSV File --</option>
            {csvFiles.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
       </div>

       <p className={`text-sm mb-4 ${t.inputInfoSub}`}>Sorted by lowest confidence.</p>
       
       <div className={`mb-4 flex items-center ${t.checkboxContainer}`}>
         <input 
           type="checkbox" 
           checked={showTones} 
           onChange={(e) => setShowTones(e.target.checked)} 
           className={t.checkbox}
           id="showTonesLabeler"
         />
         <label htmlFor="showTonesLabeler" className={t.checkboxText}>See Tones (Numbers)</label>
       </div>
       
       <div className="flex gap-2 mb-4">
         <input 
           type="text" 
           className={`flex-1 p-2 ${t.input} mb-0`} 
           placeholder="Search transcription..." 
           value={searchVal}
           onChange={e => setSearchVal(e.target.value)}
         />
         <button onClick={saveAllLabels} disabled={loading} className={`${t.buttonPrimary}`}>Save CSV</button>
       </div>

       <div className="flex-1 overflow-y-auto border border-gray-400">
         {filteredSegments.slice(0, 200).map((seg) => {
           const idx = segments.indexOf(seg);
           const isActive = idx === currentIndex;
           const hasLabel = !!seg.labeled_sentence;
           return (
             <div 
               key={idx} 
               onClick={() => selectSegment(idx)}
               className={`p-3 border-b border-gray-500 cursor-pointer ${isActive ? t.inputInfo + ' border-l-4 border-blue-500' : 'opacity-70 hover:opacity-100'}`}
             >
               <div className="flex justify-between text-xs mb-1">
                 <span className={getConfidenceClass(seg.greedy_confidence)}>{seg.greedy_confidence.toFixed(4)}</span>
                 <span className="truncate max-w-[150px] opacity-70" title={seg.filename}>{seg.filename}</span>
               </div>
               <div className="text-sm">
                 {hasLabel ? <strong>[L] {formatText(seg.labeled_sentence)}</strong> : renderDetailedTranscription(seg)}
               </div>
             </div>
           );
         })}
         {filteredSegments.length === 0 && (
           <div className="p-4 text-center text-sm text-gray-500">No segments found.</div>
         )}
       </div>
     </div>

     {/* Workspace */}
     <div className={`md:w-2/3 flex flex-col ${t.card} p-6 h-[80vh] overflow-y-auto justify-center`}>
       {currentIndex === -1 ? (
         <div className="text-center text-gray-500">
           <h3 className="text-xl font-bold mb-2">Select a segment to start</h3>
           <p>Low confidence entries are sorted at the top of the list.</p>
         </div>
       ) : (
         <div className="text-left w-full max-w-2xl mx-auto">
           <h2 className={`text-2xl font-bold mb-6 ${t.viewTitle}`}>Label Segment</h2>
           
           <div className={`p-4 mb-6 ${t.inputInfo}`}>
             <audio src={`http://localhost:8000/api/audio/${segments[currentIndex].file_path}`} controls autoPlay className="w-full mb-2" />
             <div className="text-xs text-gray-500 break-all font-mono">{segments[currentIndex].file_path}</div>
           </div>

           <div className="mb-6">
             <label className={`block text-sm font-medium ${t.label} mb-2 uppercase tracking-wide text-gray-500`}>
               Original Transcription (Confidence: {segments[currentIndex].greedy_confidence.toFixed(4)})
             </label>
              <div className={`p-3 text-lg opacity-80 ${t.inputInfo}`}>
                {renderDetailedTranscription(segments[currentIndex])}
              </div>
           </div>

           <div className="mb-8">
             <label className={`block text-sm font-medium ${t.label} mb-2 uppercase tracking-wide text-gray-500`}>
               Correction / Labeled Sentence
             </label>
             <input 
               type="text" 
               className={`w-full p-4 text-lg ${t.input}`}
               value={labelInput}
               onChange={e => setLabelInput(e.target.value)}
               onKeyDown={e => { if (e.key === 'Enter') submitLabel(); }}
               autoFocus
             />
           </div>

           <div className="flex justify-between">
             <button onClick={() => navigate(-1)} className={`py-2 px-6 ${t.buttonSecondary}`}>Previous</button>
             <button onClick={submitLabel} className={`py-2 px-6 ${t.buttonPrimary}`}>Save & Next</button>
           </div>
         </div>
       )}
       {status && (
         <div className={`mt-6 p-4 border text-center ${status.includes('❌') ? t.errorMsg : (status.includes('✅') ? t.successMsg : t.inputInfo)}`}>
           {status}
         </div>
       )}
     </div>
   </div>
 );
}


function View3({ theme }) {
  const t = theme;
  const [csvFiles, setCsvFiles] = useState([]);
  const [selectedCsv, setSelectedCsv] = useState("");
  const [status, setStatus] = useState(null);
  const [syncing, setSyncing] = useState(false);
  const [isSynced, setIsSynced] = useState(false);

  const [searchVal, setSearchVal] = useState("");
  const [debouncedSearchVal, setDebouncedSearchVal] = useState("");
  const [results, setResults] = useState([]);
  const [manifestMap, setManifestMap] = useState({});

  useEffect(() => {
    fetch("http://localhost:8000/api/audio/audiofiles-to-transcribe/segmentation_manifest.csv")
      .then(res => {
        if (!res.ok) throw new Error("No manifest");
        return res.text();
      })
      .then(text => {
        const map = {};
        const lines = text.split('\n');
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(',');
          if (cols.length >= 4) {
            const segmentedPath = cols[1].trim();
            map[segmentedPath] = {
              original: cols[0].trim(),
              start: parseInt(cols[2]),
              end: parseInt(cols[3])
            };
          }
        }
        setManifestMap(map);
      })
      .catch(() => setManifestMap({}));
  }, []);

  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => {
        setCsvFiles(data.csv_files || []);
      })
      .catch(err => console.log("Failed to fetch files", err));
  }, []);

  const formatText = (txt) => {
    if (!txt) return txt;
    return txt.replace(/[0-9]/g, '').replace(/([aeiouvAEIOUV])\1+/g, '$1');
  };

  const getProcessedWordConfidences = (segment) => {
    let wordConfs = segment.word_confidences;
    if (!wordConfs || wordConfs.length === 0) return [];
    const greedyWords = (segment.greedy_transcription || '').trim().split(/\s+/).filter(Boolean);
    if (wordConfs.length === 1 && greedyWords.length > 1) {
      const allChars = wordConfs[0].chars || [];
      const newWords = [];
      let charIdx = 0;
      for (const targetWord of greedyWords) {
        const wordChars = [];
        for (let i = 0; i < targetWord.length && charIdx < allChars.length; i++) {
          wordChars.push(allChars[charIdx]);
          charIdx++;
        }
        if (wordChars.length > 0) {
          const avgConf = wordChars.reduce((sum, c) => sum + (c.confidence || 0), 0) / wordChars.length;
          newWords.push({ word: targetWord, confidence: avgConf, chars: wordChars });
        }
      }
      if (charIdx < allChars.length) {
        const remainingChars = allChars.slice(charIdx);
        const avgConf = remainingChars.reduce((sum, c) => sum + (c.confidence || 0), 0) / remainingChars.length;
        newWords.push({ word: remainingChars.map(c => c.char).join(''), confidence: avgConf, chars: remainingChars });
      }
      return newWords;
    }
    return wordConfs;
  };

  useEffect(() => {
    const checkIfSynced = async (file) => {
      if (!file) { setIsSynced(false); setStatus(null); return; }
      try {
        const db = await initDB();
        const count = await db.transaction('segments').store.index('by-csv').count(file);
        setIsSynced(count > 0);
        if (count > 0) setStatus(`Ready to search in ${file} (locally cached)`);
        else setStatus(`Please sync data for ${file} before searching`);
      } catch (e) {
        console.error(e);
        setIsSynced(false);
      }
    };
    checkIfSynced(selectedCsv);
  }, [selectedCsv]);

  const syncData = async () => {
    const file = selectedCsv;
    if (!file) return;
    setSyncing(true);
    setStatus("Syncing data to local database...");
    try {
      const res = await fetch(`http://localhost:8000/api/labeler/data?file=${encodeURIComponent(file)}`);
      const result = await res.json();
      if (res.ok && result.status === 'success') {
        const db = await initDB();
        const tx = db.transaction(['segments', 'file_words'], 'readwrite');
        const segStore = tx.objectStore('segments');
        const wordStore = tx.objectStore('file_words');
        
        const oldSegKeys = await segStore.index('by-csv').getAllKeys(file);
        for(const k of oldSegKeys) segStore.delete(k);
        const oldWordKeys = await wordStore.index('by-file').getAllKeys(file);
        for(const k of oldWordKeys) wordStore.delete(k);

        const segmentsData = result.data;
        const fileWordsMap = new Map();
        
        const getWordAlternatives = (wordObj) => {
          if (!wordObj.chars || wordObj.chars.length === 0) return [{ word: wordObj.word, confidence: wordObj.confidence }];
          let beams = [{ text: "", confProd: 1 }];
          for (const charObj of wordObj.chars) {
            const options = [{ char: charObj.char, conf: charObj.confidence }];
            if (charObj.alternatives) {
              charObj.alternatives.forEach(alt => options.push({ char: alt.char, conf: alt.confidence }));
            }
            options.sort((a, b) => b.conf - a.conf);
            const topOptions = options.slice(0, 5);
            
            const newBeams = [];
            for (const beam of beams) {
              for (const opt of topOptions) {
                // normalize by the max confidence character so the greedy path has confProd=1
                const normalizedConf = opt.conf / options[0].conf;
                newBeams.push({ text: beam.text + opt.char, confProd: beam.confProd * normalizedConf });
              }
            }
            newBeams.sort((a, b) => b.confProd - a.confProd);
            beams = newBeams.slice(0, 5);
          }
          return beams.map(b => ({ word: b.text, confidence: wordObj.confidence * b.confProd }));
        };

        for (let idx = 0; idx < segmentsData.length; idx++) {
           const seg = segmentsData[idx];
           seg.id = `${file}_${idx}`;
           seg.csv_file = file;
           segStore.put(seg);
           
           const wordsInfo = getProcessedWordConfidences(seg);
           wordsInfo.forEach(wObj => {
              const top5Words = getWordAlternatives(wObj);
              top5Words.forEach((altWordObj, rank) => {
                  const wordClean = formatText(altWordObj.word).toLowerCase();
                  if (!wordClean) return;
                  const wordKey = `${file}_${wordClean}_${rank}`; // ensure unique key for alternatives
                  if (!fileWordsMap.has(wordClean)) {
                      fileWordsMap.set(wordClean, {
                          id: wordKey,
                          file_path: file,
                          word: wordClean,
                          segments: []
                      });
                  }
                  fileWordsMap.get(wordClean).segments.push({
                      segmentIndex: idx,
                      wordInfo: wObj,
                      altWordInfo: altWordObj,
                      rank: rank
                  });
              });
           });
        }
        
        for (const val of fileWordsMap.values()) {
            wordStore.put(val);
        }
        
        await tx.done;
        setStatus(`✅ Sync complete! ${segmentsData.length} segments indexed.`);
        setIsSynced(true);
      } else {
        setStatus(`❌ Error: ${result.detail || result.message}`);
      }
    } catch (err) {
      setStatus(`❌ Error syncing data: ${err.message}`);
    }
    setSyncing(false);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchVal(searchVal);
    }, 500);
    return () => clearTimeout(timer);
  }, [searchVal]);

  useEffect(() => {
    if (!debouncedSearchVal.trim() || !isSynced || !selectedCsv) {
      setResults([]);
      return;
    }

    const searchAsync = async () => {
      const query = formatText(debouncedSearchVal.trim()).toLowerCase();
      const db = await initDB();
      const wordsForFile = await db.getAllFromIndex('file_words', 'by-file', selectedCsv);
      
      const bestMatchPerSegment = new Map();

      wordsForFile.forEach(fwObj => {
        const wordClean = fwObj.word;
        let baseDist = Infinity;
        
        if (wordClean === query) {
          baseDist = 0;
        } else if (wordClean.startsWith(query)) {
          baseDist = 1;
        } else if (wordClean.includes(query)) {
          baseDist = 2;
        }

        if (baseDist < Infinity) {
          fwObj.segments.forEach(segInfo => {
             const isAlt = (segInfo.rank > 0);
             const matchTypeDist = baseDist * 2 + (isAlt ? 1 : 0);
             
             const confidence = segInfo.altWordInfo ? segInfo.altWordInfo.confidence : segInfo.wordInfo.confidence;

             const existing = bestMatchPerSegment.get(segInfo.segmentIndex);
             if (!existing || matchTypeDist < existing.distance || (matchTypeDist === existing.distance && confidence > existing.confidence)) {
                 bestMatchPerSegment.set(segInfo.segmentIndex, {
                     segmentIndex: segInfo.segmentIndex,
                     matchWordInfo: segInfo.wordInfo,
                     altWordInfo: segInfo.altWordInfo,
                     rank: segInfo.rank,
                     distance: matchTypeDist,
                     confidence: confidence
                 });
             }
          });
        }
      });
      
      const matched = Array.from(bestMatchPerSegment.values());

      matched.sort((a, b) => {
        if (a.distance !== b.distance) {
          return a.distance - b.distance;
        }
        return b.confidence - a.confidence;
      });

      const tx = db.transaction('segments');
      const resultsWithSegments = [];
      for (const m of matched) {
         const seg = await tx.store.get(`${selectedCsv}_${m.segmentIndex}`);
         if (seg) {
             resultsWithSegments.push({
                 ...m,
                 segment: seg
             });
         }
      }
      
      setResults(resultsWithSegments);
    };

    searchAsync();
  }, [debouncedSearchVal, isSynced, selectedCsv]);

  const renderTranscriptionWithHighlight = (segment, matchedWordInfo) => {
    const wordConfs = getProcessedWordConfidences(segment);
    if (!wordConfs || wordConfs.length === 0) {
      return formatText(segment.greedy_transcription) || <i className="text-gray-400">(Empty)</i>;
    }

    return (
      <div className="flex flex-wrap gap-x-4 gap-y-4 items-end py-1">
        {wordConfs.map((wordObj, i) => {
          const formattedWord = formatText(wordObj.word);
          if (!formattedWord) return null;
          
          const isMatch = matchedWordInfo && matchedWordInfo.word === wordObj.word;
          
          const getWordConfColor = (conf) => {
            if (conf < 0.8) return 'text-red-700 bg-red-100 border-red-300';
            if (conf < 0.95) return 'text-orange-700 bg-orange-100 border-orange-300';
            return 'text-green-800 bg-green-100 border-green-300';
          };

          const matchClass = isMatch 
            ? "bg-yellow-100 border-2 border-yellow-400 shadow-md scale-105" 
            : "bg-gray-50 hover:bg-gray-100 border border-gray-300/80 shadow-sm";

          return (
            <div 
                key={i} 
                className={`inline-flex flex-col items-center group relative cursor-pointer pt-4 pb-1 px-2 rounded-md transition-all ${matchClass}`}
                title={`Word: "${wordObj.word}"\nConfidence: ${(wordObj.confidence * 100).toFixed(1)}%`}
            >
                {/* Characters Row with 2nd best prediction hint above low/mid conf chars */}
                <div className="flex items-end leading-none">
                    {wordObj.chars.map((charObj, j) => {
                        const formattedChar = formatText(charObj.char);
                        if (!formattedChar) return null;

                        const isLowMidConf = charObj.confidence < 0.8;
                        const topAlt = (charObj.alternatives && charObj.alternatives.length > 0) ? charObj.alternatives[0] : null;
                        const formattedAltChar = topAlt ? formatText(topAlt.char) : null;

                        // Build Top 5 alternatives list for character tooltip
                        const allPredictions = [
                          { char: charObj.char, confidence: charObj.confidence },
                          ...(charObj.alternatives || [])
                        ].slice(0, 5);

                        const charTooltip = `Character: '${charObj.char}' (${(charObj.confidence * 100).toFixed(1)}%)\n\nTop 5 Predictions:\n` +
                          allPredictions.map((p, idx) => `${idx + 1}. '${p.char}' — ${(p.confidence * 100).toFixed(1)}%`).join('\n');

                        return (
                            <div key={j} className="relative flex flex-col items-center px-[0.5px]">
                                {/* 2nd best prediction floating in small text above character */}
                                {isLowMidConf && formattedAltChar && (
                                    <span 
                                        className="text-[10px] font-mono font-bold leading-none text-blue-600 absolute -top-3.5 opacity-80 hover:opacity-100 select-none"
                                        title={`2nd Choice: '${topAlt.char}' (${(topAlt.confidence * 100).toFixed(1)}%)`}
                                    >
                                        {formattedAltChar}
                                    </span>
                                )}

                                {/* Primary character */}
                                <span 
                                    className={`
                                        text-xl font-medium leading-none px-0.5 rounded-sm transition-colors duration-150 inline-block
                                        ${charObj.confidence < 0.5 ? 'text-red-600 font-bold bg-red-100' : charObj.confidence < 0.8 ? 'text-orange-600 font-semibold bg-orange-100' : 'text-gray-900'}
                                        hover:bg-blue-500/30 hover:text-blue-900
                                    `}
                                    title={charTooltip}
                                >
                                    {formattedChar}
                                </span>
                            </div>
                        );
                    })}
                </div>

                {/* Word Confidence display directly below the word */}
                <span 
                    className={`text-[11px] font-mono font-bold leading-tight mt-1.5 px-1.5 py-0.5 rounded border ${getWordConfColor(wordObj.confidence)}`}
                    title={`Word Confidence: ${(wordObj.confidence * 100).toFixed(2)}%`}
                >
                    {(wordObj.confidence * 100).toFixed(1)}%
                </span>
            </div>
          );
        })}
      </div>
    );
  };

  const getConfidenceClass = (conf) => {
    if (conf < 0.8) return "text-red-600 font-bold";
    if (conf < 0.95) return "text-orange-500 font-bold";
    return "text-green-600 font-bold";
  };

  const formatTimestamp = (ms) => {
    const totalSec = Math.floor(ms / 1000);
    const min = Math.floor(totalSec / 60);
    const sec = totalSec % 60;
    return `${min}:${sec.toString().padStart(2, '0')}.${(ms % 1000).toString().padStart(3, '0')}`;
  };

  return (
    <div className="max-w-4xl mx-auto text-left pb-16">
      <h2 className={`text-3xl font-bold mb-6 text-center ${t.viewTitle}`}>3. Search</h2>
      <p className={`text-center ${t.viewDesc} mb-8`}>Search for a word across all transcribed segments in a CSV file.</p>
      
      <div className={`${t.card} p-6 mb-6`}>
        <div className="flex flex-col md:flex-row gap-4 mb-4">
          <div className="flex gap-2 items-end">
            <div className="flex-1">
              <label className={`block text-sm font-medium ${t.label} mb-1`}>Results CSV File</label>
              <select 
                className={`w-full p-3 ${t.input} mb-0`} 
                value={selectedCsv} 
                onChange={e => setSelectedCsv(e.target.value)}
              >
                <option value="">-- Select CSV File --</option>
                {csvFiles.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            <button
              className={`${t.buttonPrimary} py-3 mb-[8px]`}
              onClick={syncData}
              disabled={!selectedCsv || syncing}
            >
              {syncing ? "Syncing..." : "Sync Data"}
            </button>
          </div>
          <div className="flex-1">
            <label className={`block text-sm font-medium ${t.label} mb-1`}>Search Term</label>
            <input 
              type="text" 
              className={`w-full p-3 ${t.input} mb-0`} 
              placeholder="Type a word..." 
              value={searchVal}
              onChange={e => setSearchVal(e.target.value)}
              disabled={!selectedCsv || !isSynced || syncing}
            />
          </div>
        </div>
        {status && <div className={`p-2 ${status.includes('❌') ? t.errorMsg : t.inputInfo}`}>{status}</div>}
      </div>

      {(() => {
        const LOW_CONFIDENCE_THRESHOLD = 0.30;
        const highConfResults = results.filter(r => r.confidence >= LOW_CONFIDENCE_THRESHOLD);
        const lowConfResults = results.filter(r => r.confidence < LOW_CONFIDENCE_THRESHOLD);

        const renderResultItem = (res, index, isLowConf) => {
          const mainIdx = res.segmentIndex;
          const mainSeg = res.segment;
          
          let manifestInfo = null;
          for (const key in manifestMap) {
            if (mainSeg.file_path.endsWith(key)) {
              manifestInfo = manifestMap[key];
              break;
            }
          }

          return (
            <div key={`${mainIdx}-${index}`} className={`border ${t.card} p-4 rounded shadow ${isLowConf ? 'opacity-60 grayscale' : ''}`}>
              <div className="flex justify-between items-center mb-4 pb-2 border-b border-gray-300 dark:border-gray-700">
                <div>
                  <span className="font-bold mr-4 text-lg">Match {index + 1}</span>
                  {isLowConf && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 border border-gray-400 dark:border-gray-500 font-bold uppercase tracking-wider">
                      Low Confidence Result
                    </span>
                  )}
                  {res.distance === 0 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border border-blue-300 dark:border-blue-700 font-bold">
                      Exact Match
                    </span>
                  )}
                  {res.distance === 1 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-cyan-100 dark:bg-cyan-900 text-cyan-800 dark:text-cyan-200 border border-cyan-300 dark:border-cyan-700 font-bold">
                      Alternative Exact Match
                    </span>
                  )}
                  {res.distance === 2 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 border border-purple-300 dark:border-purple-700 font-bold">
                      Starts With
                    </span>
                  )}
                  {res.distance === 3 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-fuchsia-100 dark:bg-fuchsia-900 text-fuchsia-800 dark:text-fuchsia-200 border border-fuchsia-300 dark:border-fuchsia-700 font-bold">
                      Alternative Starts With
                    </span>
                  )}
                  {res.distance === 4 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-300 border border-gray-300 dark:border-gray-600 font-bold">
                      Contains
                    </span>
                  )}
                  {res.distance === 5 && (
                    <span className="text-sm mr-4 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-300 border border-slate-300 dark:border-slate-600 font-bold">
                      Alternative Contains
                    </span>
                  )}

                  <span className="text-sm opacity-80">Confidence: </span>
                  <span className={`text-sm ${getConfidenceClass(res.confidence)}`}>{(res.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {manifestInfo && (
                <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700">
                  <div className="text-xs font-bold mb-1 opacity-80 uppercase tracking-wider">Full Source Audio</div>
                  <div className="text-sm mb-2 font-mono text-gray-700 dark:text-gray-300">
                    {manifestInfo.original} <span className="ml-2 font-bold text-blue-600 dark:text-blue-400">({formatTimestamp(manifestInfo.start)} - {formatTimestamp(manifestInfo.end)})</span>
                  </div>
                  <audio 
                    src={`http://localhost:8000/api/audio/wav/${manifestInfo.original}#t=${manifestInfo.start/1000},${manifestInfo.end/1000}`} 
                    controls 
                    className="h-10 w-full opacity-90 hover:opacity-100 transition-opacity" 
                  />
                </div>
              )}

              {/* Main Segment */}
              <div className="pl-4 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-2 pr-2">
                <div className="text-xs mb-1 font-mono text-blue-800 dark:text-blue-300">
                  Segment: {mainSeg.file_path} 
                  {!manifestInfo && <span className="ml-2 italic opacity-70">(Source audio mapping not found)</span>}
                </div>
                <audio src={`http://localhost:8000/api/audio/${mainSeg.file_path}`} controls className="h-10 w-full mb-3" />
                <div>{renderTranscriptionWithHighlight(mainSeg, res.matchWordInfo)}</div>
              </div>
            </div>
          );
        };

        return (
          <>
            {debouncedSearchVal && results.length === 0 && isSynced && (
              <div className={`p-6 text-center border ${t.inputInfo}`}>
                No matches found for "{debouncedSearchVal}".
              </div>
            )}
            
            {debouncedSearchVal && results.length > 0 && (
              <div className="mb-4 font-bold opacity-80 text-center">
                Found {highConfResults.length} matches for "{debouncedSearchVal}"
                {lowConfResults.length > 0 && ` (and ${lowConfResults.length} low confidence results hidden below)`}
              </div>
            )}
            
            <div className="space-y-8">
              {highConfResults.map((res, index) => renderResultItem(res, index, false))}
              
              {lowConfResults.length > 0 && (
                <div className="mt-12 mb-6">
                  <div className="flex items-center justify-center space-x-4">
                    <div className="flex-1 h-px bg-gray-300 dark:bg-gray-700"></div>
                    <div className="text-gray-500 font-bold uppercase tracking-wider text-sm">Low Confidence Results</div>
                    <div className="flex-1 h-px bg-gray-300 dark:bg-gray-700"></div>
                  </div>
                </div>
              )}
              
              {lowConfResults.map((res, index) => renderResultItem(res, highConfResults.length + index, true))}
            </div>
          </>
        );
      })()}
    </div>
  );
}

function invertRespellConsonants(s) {
  if (!s) return s;
  // 1. ([^ht])hs -> \1s
  s = s.replace(/([^hHtT])hs/g, '$1s');
  s = s.replace(/([^hHtT])HS/g, '$1S');
  
  // 2. slh(?=[aeiouv]) -> sl
  s = s.replace(/slh(?=[aeiouv])/g, 'sl');
  s = s.replace(/SLH(?=[AEIOUV])/g, 'SL');
  
  // 3. tsh -> ch
  s = s.replace(/tsh/g, 'ch');
  s = s.replace(/TSH/g, 'CH');
  s = s.replace(/Tsh/g, 'Ch');
  
  // 4. ts -> j
  s = s.replace(/ts/g, 'j');
  s = s.replace(/TS/g, 'J');
  s = s.replace(/Ts/g, 'J');
  
  // 5. t(?!h) -> d
  s = s.replace(/t(?!h)/g, 'd');
  s = s.replace(/T(?!h|H)/g, 'D');
  
  // 6. th -> t
  s = s.replace(/th/g, 't');
  s = s.replace(/TH/g, 'T');
  s = s.replace(/Th/g, 'T');
  
  // 7. k(?!h) -> g, kh -> k
  s = s.replace(/k(?!h)/g, 'g');
  s = s.replace(/K(?!h|H)/g, 'G');
  s = s.replace(/kh/g, 'k');
  s = s.replace(/KH/g, 'K');
  s = s.replace(/Kh/g, 'K');
  
  // 8. nh -> hn, lh -> hl, yh -> hy, wh -> hw
  s = s.replace(/nh/g, 'hn');
  s = s.replace(/NH/g, 'HN');
  s = s.replace(/Nh/g, 'Hn');
  
  s = s.replace(/lh/g, 'hl');
  s = s.replace(/LH/g, 'HL');
  s = s.replace(/Lh/g, 'Hl');
  
  s = s.replace(/yh/g, 'hy');
  s = s.replace(/YH/g, 'HY');
  s = s.replace(/Yh/g, 'Hy');
  
  s = s.replace(/wh/g, 'hw');
  s = s.replace(/WH/g, 'HW');
  s = s.replace(/Wh/g, 'Hw');
  
  // 9. ' / ’ -> ?
  // s = s.replace(/'/g, '?');
  // s = s.replace(/’/g, '?');
  
  return s;
}

function View4({ theme, activeTab }) {
  const t = theme;
  const [csvFiles, setCsvFiles] = useState([]);
  const [selectedCsv, setSelectedCsv] = useState("");
  const [manifestMap, setManifestMap] = useState({});
  const [audioFiles, setAudioFiles] = useState([]);
  const [selectedAudio, setSelectedAudio] = useState("");
  const [lyricsData, setLyricsData] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const wavesurferRef = useRef(null);
  const containerRef = useRef(null);

  const { activeItem, activeSegmentIdx } = useMemo(() => {
      if (activeTab !== 4) return { activeItem: null, activeSegmentIdx: -1 };
      let actItem = null;
      let actSeg = -1;
      lyricsData.forEach((segment, segIdx) => {
          segment.forEach(item => {
              if (currentTime >= item.start_time && currentTime < item.end_time) {
                  if (!actItem || item.start_time >= actItem.start_time) {
                      actItem = item;
                      actSeg = segIdx;
                  }
              }
          });
      });
      return { activeItem: actItem, activeSegmentIdx: actSeg };
  }, [currentTime, lyricsData, activeTab]);

  useEffect(() => {
      if (activeTab === 4 && activeSegmentIdx >= 0) {
          const el = document.getElementById(`segment-${activeSegmentIdx}`);
          if (el) {
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
      }
  }, [activeSegmentIdx, activeTab]);

  useEffect(() => {
    fetch("http://localhost:8000/api/audio/audiofiles-to-transcribe/segmentation_manifest.csv")
      .then(res => {
        if (!res.ok) throw new Error("No manifest");
        return res.text();
      })
      .then(text => {
        const map = {};
        const lines = text.split('\n');
        for (const line of lines) {
          const cols = line.split(',');
          if (cols.length >= 4) {
            const segmentedPath = cols[1].trim();
            map[segmentedPath] = {
              original: cols[0].trim(),
              start: parseInt(cols[2]),
              end: parseInt(cols[3])
            };
          }
        }
        setManifestMap(map);
      })
      .catch(() => setManifestMap({}));

    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => setCsvFiles(data.csv_files || []))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedCsv) return;
    fetch(`http://localhost:8000/api/labeler/data?file=${encodeURIComponent(selectedCsv)}`)
      .then(res => res.json())
      .then(result => {
        const rows = result.data || (Array.isArray(result) ? result : []);
        const segmentsByAudio = {};
        rows.forEach(row => {
           let manifest = null;
           for (const key in manifestMap) {
              if (row.file_path && row.file_path.endsWith(key)) {
                 manifest = manifestMap[key];
                 break;
              }
           }
           
           if (manifest) {
              const orig = manifest.original;
              if (!segmentsByAudio[orig]) segmentsByAudio[orig] = { url: `audiofiles-to-transcribe/${orig}`, segments: [] };
              segmentsByAudio[orig].segments.push({ ...row, manifest });
           } else {
              const orig = row.file_path;
              if (!segmentsByAudio[orig]) segmentsByAudio[orig] = { url: orig, segments: [] };
              segmentsByAudio[orig].segments.push({ ...row, manifest: { original: orig, start: 0, end: 1000000 } });
           }
        });
        
        const files = Object.keys(segmentsByAudio).sort();
        const parsedAudioFiles = files.map(f => ({ name: f, url: segmentsByAudio[f].url, segments: segmentsByAudio[f].segments }));
        setAudioFiles(parsedAudioFiles);
        if (parsedAudioFiles.length > 0) {
          setSelectedAudio(parsedAudioFiles[0].name);
        } else {
          setSelectedAudio("");
        }
        setLyricsData([]);
      })
      .catch(console.error);
  }, [selectedCsv, manifestMap]);

  useEffect(() => {
    if (!selectedAudio || audioFiles.length === 0) return;
    const fileData = audioFiles.find(f => f.name === selectedAudio);
    if (!fileData) return;

    const sortedSegments = [...fileData.segments].sort((a, b) => a.manifest.start - b.manifest.start);
    
    let segmentsData = [];
    sortedSegments.forEach(seg => {
       const startSec = seg.manifest.start / 1000;
       const endSec = seg.manifest.end / 1000;
       const duration = endSec - startSec;
       
       if (seg.word_confidences && seg.word_confidences.length > 0) {
           const invertedWords = seg.word_confidences.map(w => ({
               ...w,
               word: invertRespellConsonants(w.word)
           }));
           const totalChars = invertedWords.reduce((sum, w) => sum + w.word.length, 0);
           let currentLen = 0;
           let segmentWords = [];
           invertedWords.forEach(w => {
               let wStart = startSec;
               let wEnd = endSec;
               
               if (w.start_time !== undefined && w.end_time !== undefined) {
                   wStart = startSec + w.start_time;
                   wEnd = startSec + w.end_time;
               } else {
                   wStart = startSec + (currentLen / Math.max(1, totalChars)) * duration;
                   wEnd = startSec + ((currentLen + w.word.length) / Math.max(1, totalChars)) * duration;
                   currentLen += w.word.length;
               }
               
               segmentWords.push({
                   word: w.word,
                   start_time: wStart,
                   end_time: wEnd,
                   confidence: w.confidence
               });
           });
           segmentsData.push(segmentWords);
       }
    });
    
    setLyricsData(segmentsData);
  }, [selectedAudio, audioFiles]);

  useEffect(() => {
    if (!selectedAudio || audioFiles.length === 0) return;
    const fileData = audioFiles.find(f => f.name === selectedAudio);
    if (!fileData) return;
    
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: 'rgba(59, 130, 246, 0.5)',
      progressColor: 'rgba(37, 99, 235, 0.8)',
      cursorColor: 'rgb(37, 99, 235)',
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 80,
      normalize: true
    });
    
    ws.load(`http://localhost:8000/api/audio/${fileData.url}`);
    
    ws.on('timeupdate', (time) => setCurrentTime(time));
    ws.on('seek', (progress) => setCurrentTime(progress * ws.getDuration()));
    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    
    wavesurferRef.current = ws;
    
    return () => {
       ws.destroy();
    };
  }, [selectedAudio]);

  const togglePlay = () => wavesurferRef.current?.playPause();
  const skip = (s) => wavesurferRef.current?.setTime(Math.max(0, Math.min(wavesurferRef.current.getCurrentTime() + s, wavesurferRef.current.getDuration())));
  const jumpToTime = (time) => {
      if (wavesurferRef.current) {
          wavesurferRef.current.setTime(time);
          wavesurferRef.current.play();
      }
  };

  return (
    <div className="max-w-4xl mx-auto text-center">
      <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>4. Listen</h2>
      <p className={`${t.viewDesc} mb-8`}>Listen to full audio files with synced lyrics highlighting.</p>

      <div className={`${t.card} p-6 mb-6`}>
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-left">
          <div className="flex-1 w-full">
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Select Results CSV</label>
            <select className={`w-full p-3 ${t.input}`} value={selectedCsv} onChange={e => setSelectedCsv(e.target.value)}>
              <option value="">-- Choose CSV File --</option>
              {csvFiles.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          {audioFiles.length > 1 && (
            <div className="flex-1 w-full">
              <label className={`block text-sm font-medium ${t.label} mb-2`}>Source Audio File</label>
              <select className={`w-full p-3 ${t.input}`} value={selectedAudio} onChange={e => setSelectedAudio(e.target.value)}>
                {audioFiles.map(f => <option key={f.name} value={f.name}>{f.name} ({f.segments.length} segments)</option>)}
              </select>
            </div>
          )}
        </div>
      </div>

      {selectedAudio && (
        <div className={`border rounded shadow-lg overflow-hidden ${t.card} p-0`}>
          <div className="p-4 flex flex-col items-center border-b border-gray-500" style={{ backgroundColor: 'rgba(128,128,128,0.1)' }}>
             <div ref={containerRef} className="w-full mb-4"></div>
             <div className="flex items-center space-x-6">
                <button onClick={() => skip(-10)} className="hover:text-blue-400 transition-colors">
                   <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 17l-5-5 5-5M18 17l-5-5 5-5"/></svg>
                </button>
                <button onClick={togglePlay} className="p-3 bg-blue-600 rounded-full hover:bg-blue-500 transition-colors">
                   {isPlaying ? (
                       <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                   ) : (
                       <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>
                   )}
                </button>
                <button onClick={() => skip(10)} className="hover:text-blue-400 transition-colors">
                   <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 17l5-5-5-5M6 17l5-5-5-5"/></svg>
                </button>
             </div>
          </div>
          <div className="p-8 max-h-[60vh] overflow-y-auto text-left scroll-smooth">
             <div className="max-w-3xl mx-auto space-y-6 leading-loose">
                 {lyricsData.length === 0 ? (
                    <div className="text-gray-500 italic text-center">No lyrics found for this audio.</div>
                 ) : (
                    lyricsData.map((segment, segIdx) => (
                        <div key={segIdx} id={`segment-${segIdx}`} className="mb-2">
                            {segment.map((item, idx) => {
                                const isActive = item === activeItem;
                                const isPast = currentTime >= item.end_time;
                                
                                let colorClass = "opacity-40 hover:opacity-70"; // future
                                if (isActive) colorClass = "bg-yellow-200 dark:bg-amber-900/50 rounded px-1 opacity-100";
                                else if (isPast) colorClass = "opacity-100 hover:opacity-80"; // past
                                
                                return (
                                   <span 
                                      key={idx} 
                                      onClick={() => jumpToTime(item.start_time)}
                                      className={`inline-block mx-1 text-2xl transition-colors duration-200 cursor-pointer ${colorClass}`}
                                      title={`Confidence: ${(item.confidence * 100).toFixed(1)}%`}
                                   >
                                      {item.word}
                                   </span>
                                );
                            })}
                        </div>
                    ))
                 )}
             </div>
          </div>
        </div>
      )}
    </div>
  );
}

const TABS = [
  { id: 0, name: "Batch Segmentation", title: "Batch Segmentation" },
  { id: 1, name: "Batch Inference", title: "Batch Inference" },
  { id: 2, name: "Review", title: "Review Tool" },
  { id: 3, name: "Search", title: "Search" },
  { id: 4, name: "Listen", title: "Listen" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [baseFolder, setBaseFolder] = useState("");
  const [selectingFolder, setSelectingFolder] = useState(false);
  const [activeThemeId, setActiveThemeId] = useState(() => {
    const saved = localStorage.getItem('cherokee-asr-theme');
    return saved === 'minimalLight' ? 'minimalLight' : 'minimalDark';
  });

  const t = THEMES[activeThemeId];

  useEffect(() => {
    localStorage.setItem('cherokee-asr-theme', activeThemeId);
  }, [activeThemeId]);

  useEffect(() => {
    fetch("http://localhost:8000/api/config/folder")
      .then(res => res.json())
      .then(data => setBaseFolder(data.folder))
      .catch(err => console.log("Failed to fetch folder", err));
  }, []);

  const handleBrowseFolder = async () => {
    setSelectingFolder(true);
    try {
      const res = await fetch("http://localhost:8000/api/config/select_folder", { method: "POST" });
      const data = await res.json();
      setBaseFolder(data.folder);
    } catch (err) {
      console.log("Failed to select folder", err);
    }
    setSelectingFolder(false);
  };

  const toggleTheme = () => {
    setActiveThemeId(prev => prev === 'minimalLight' ? 'minimalDark' : 'minimalLight');
  };

  return (
    <div className={`min-h-screen ${t.bgApp}`}>
      {/* Top Navigation */}
      <header className={t.topbar}>
        <div className={t.topbarTitle}>Transcription Tools</div>
        <button 
          onClick={toggleTheme} 
          className={`${t.buttonSecondary} absolute top-2 right-4 text-lg`} 
          style={{ padding: '2px 6px', lineHeight: '1' }}
          title="Toggle Theme"
        >
          {t.toggleIcon}
        </button>
        <div className="mb-2 text-sm">
          Base Folder: {baseFolder || "Loading..."} 
          <button onClick={handleBrowseFolder} disabled={selectingFolder} className={`ml-2 ${t.buttonSecondary}`}>
            {selectingFolder ? "Selecting..." : "Browse"}
          </button>
        </div>
        <nav>
          {TABS.map(tab => (
            <button 
              key={tab.id} 
              onClick={() => setActiveTab(tab.id)}
              className={activeTab === tab.id ? t.topbarTabActive : t.topbarTabInactive}
            >
              {tab.id}: {tab.name}
            </button>
          ))}
        </nav>
      </header>

      {/* Main Content Centered */}
      <main className={`p-4 ${t.bgMain}`}>
<div key={baseFolder} className="mx-auto max-w-4xl text-center">
          {activeTab === 0 && <View0 theme={t} />}
          {activeTab === 1 && <View1 theme={t} />}
          {activeTab === 2 && <View2 theme={t} />}
          {activeTab === 3 && <View3 theme={t} />}
          {activeTab === 4 && <View4 theme={t} activeTab={activeTab} />}
        </div>
      </main>
    </div>
  );
}
