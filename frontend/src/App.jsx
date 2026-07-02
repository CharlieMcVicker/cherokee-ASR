import { useState, useEffect, useRef, useMemo } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';

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

function PreviewSurfer({ preview, theme, zoom }) {
  const containerRef = useRef(null);
  const [ws, setWs] = useState(null);
  const [wsRegions, setWsRegions] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const regions = RegionsPlugin.create();
    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor: 'rgb(200, 0, 200)',
      progressColor: 'rgb(100, 0, 100)',
      url: `http://localhost:8000/api/audio/${preview.file}`,
      plugins: [regions],
      height: 64,
      normalize: false,
      minPxPerSec: Number(zoom)
    });
    
    wavesurfer.on('ready', () => {
      preview.segments.forEach((seg, index) => {
        regions.addRegion({
          start: seg.start,
          end: seg.end,
          color: index % 2 === 0 ? 'rgba(0, 200, 0, 0.2)' : 'rgba(0, 255, 0, 0.1)'
        });
      });
    });

    setWs(wavesurfer);
    setWsRegions(regions);

    return () => {
      wavesurfer.destroy();
    };
  }, [preview, zoom]);

  useEffect(() => {
    if (ws) {
      try {
        const wrapper = ws.getWrapper();
        const scrollLeft = wrapper.scrollLeft;
        const width = wrapper.clientWidth;
        const scrollWidth = wrapper.scrollWidth;
        const centerRatio = scrollWidth > 0 ? (scrollLeft + width / 2) / scrollWidth : 0;
        
        ws.zoom(Number(zoom));
        
        setTimeout(() => {
          const newScrollWidth = wrapper.scrollWidth;
          wrapper.scrollLeft = centerRatio * newScrollWidth - width / 2;
        }, 0);
      } catch (e) {
      }
    }
  }, [zoom, ws]);

  return (
    <div className={`mb-4 p-2 border ${theme.inputInfo}`}>
      <div className="text-xs text-left mb-2 text-gray-500 font-mono break-all">{preview.file}</div>
      <div ref={containerRef} className="w-full bg-white" />
    </div>
  );
}

function View0({ theme }) {
  const t = theme;
  const [files, setFiles] = useState({ wav_files: [], folders: [] });
  const [targetPath, setTargetPath] = useState("");
  const [settings, setSettings] = useState({ silence_thresh: -40, min_silence_len: 500, keep_silence: 100 });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previews, setPreviews] = useState([]);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => {
        const allWavs = data.wav_files || [];
        const folderSet = new Set();
        allWavs.forEach(f => {
          const parts = f.split('/');
          if (parts.length > 1) {
            folderSet.add(parts.slice(0, -1).join('/'));
          }
        });
        setFiles({ wav_files: allWavs, folders: Array.from(folderSet).sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b)) });
      }).catch(err => console.log("Failed to fetch files", err));
  }, []);

  useEffect(() => {
    if (!targetPath) {
      setPreviews([]);
      return;
    }
    const fetchPreviews = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/preview_segments", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_path: targetPath, ...settings })
        });
        const data = await res.json();
        if (res.ok) {
          setPreviews(data.previews || []);
        }
      } catch (e) {
        console.log("Failed to fetch previews", e);
      }
    };
    const timer = setTimeout(fetchPreviews, 500);
    return () => clearTimeout(timer);
  }, [targetPath, settings]);

  const handleSegment = async () => {
    if (!targetPath) return;
    setLoading(true);
    setStatus("⏳ Processing batch segmentation...");
    try {
      const res = await fetch("http://localhost:8000/api/batch_segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_path: targetPath, ...settings })
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

  return (
    <div className="max-w-3xl mx-auto text-center ">
      <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>0. Batch Segmentation</h2>
      <p className={`${t.viewDesc} mb-8`}>Select a single file or a folder to automatically cut up all audio files at once based on silence thresholds.</p>
      
      <div className={`${t.card} p-6 `}>
        <div className="mb-6 text-left">
          <label className={`block text-sm font-medium ${t.label} mb-2`}>Target File or Folder</label>
          <select className={`w-full p-3 ${t.input}`} value={targetPath} onChange={e => setTargetPath(e.target.value)}>
            <option value="">-- Select File or Folder --</option>
            {files.folders.length > 0 && <optgroup label="Folders">{files.folders.map(f => <option key={f} value={f}>{f}</option>)}</optgroup>}
            {files.wav_files.length > 0 && <optgroup label="Files">{files.wav_files.map(f => <option key={f} value={f}>{f}</option>)}</optgroup>}
          </select>
        </div>

        <div className={`grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 p-4 border ${t.inputInfo} text-left`}>
          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Silence Threshold (dBFS)</label>
            <div className="flex items-center gap-2">
              <input type="range" min="-80" max="0" step="1" value={settings.silence_thresh} onChange={(e) => setSettings({...settings, silence_thresh: parseInt(e.target.value) || 0})} className="w-full" />
              <input type="number" min="-80" max="0" value={settings.silence_thresh} onChange={(e) => setSettings({...settings, silence_thresh: parseInt(e.target.value) || 0})} className={`w-16 p-1 text-sm ${t.input} mb-0`} />
            </div>
          </div>
          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Min Silence (ms)</label>
            <div className="flex items-center gap-2">
              <input type="range" min="10" max="1500" step="10" value={settings.min_silence_len} onChange={(e) => setSettings({...settings, min_silence_len: parseInt(e.target.value) || 0})} className="w-full" />
              <input type="number" min="10" max="1500" value={settings.min_silence_len} onChange={(e) => setSettings({...settings, min_silence_len: parseInt(e.target.value) || 0})} className={`w-20 p-1 text-sm ${t.input} mb-0`} />
            </div>
          </div>
          <div>
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Keep Silence (ms)</label>
            <div className="flex items-center gap-2">
              <input type="range" min="0" max="500" step="10" value={settings.keep_silence} onChange={(e) => setSettings({...settings, keep_silence: parseInt(e.target.value) || 0})} className="w-full" />
              <input type="number" min="0" max="500" value={settings.keep_silence} onChange={(e) => setSettings({...settings, keep_silence: parseInt(e.target.value) || 0})} className={`w-20 p-1 text-sm ${t.input} mb-0`} />
            </div>
          </div>
        </div>

        {previews.length > 0 && (
          <div className="mb-6">
            <h3 className="font-bold text-left mb-2">Live Previews (up to 3 files)</h3>
            <div className="flex items-center gap-3 mb-2">
                <label className={`text-sm font-medium ${t.label} whitespace-nowrap`}>Preview Zoom:</label>
                <input 
                    type="range" 
                    min="1" 
                    max="1000" 
                    value={zoom} 
                    onChange={(e) => setZoom(e.target.value)} 
                    className="w-1/2"
                />
            </div>
            {previews.map((p, i) => <PreviewSurfer key={i} preview={p} theme={t} zoom={zoom} />)}
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
  const [form, setForm] = useState({ checkpoint: "charliemcvicker/asr-cherokee" });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchStats, setBatchStats] = useState(null);
  const [isFetchingStats, setIsFetchingStats] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then(res => res.json())
      .then(data => {
        const allWavs = data.wav_files || [];
        const folderSet = new Set();
        allWavs.forEach(f => {
          const parts = f.split('/');
          if (parts.length > 1) {
            folderSet.add(parts.slice(0, -1).join('/'));
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
          checkpoint: form.checkpoint
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
            <label className={`block text-sm font-medium ${t.label} mb-2`}>Model Checkpoint (HF Repo ID or Path)</label>
            <input type="text" className={`w-full p-3 ${t.input}`} value={form.checkpoint} onChange={e => setForm({...form, checkpoint: e.target.value})} />
          </div>
        </div>

        <button 
          onClick={handleInference} 
          disabled={loading || !form.checkpoint || selectedCount === 0}
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

  const formatText = (txt) => {
    if (!txt) return txt;
    return showTones ? txt : txt.replace(/[0-9]/g, '');
  };

  const renderDetailedTranscription = (segment) => {
    if (segment.word_confidences && segment.word_confidences.length > 0) {
        return (
            <div className="flex flex-wrap gap-2 items-center">
                {segment.word_confidences.map((wordObj, i) => {
                    const formattedWord = formatText(wordObj.word);
                    if (!formattedWord) return null;
                    return (
                        <span 
                            key={i} 
                            className="group relative cursor-pointer border-b border-transparent hover:border-gray-400 pb-[1px]"
                            title={`Word: ${wordObj.word}\nConfidence: ${wordObj.confidence.toFixed(4)}`}
                        >
                            {wordObj.chars.map((charObj, j) => {
                                const formattedChar = formatText(charObj.char);
                                if (!formattedChar) return null;
                                return (
                                    <span 
                                        key={j}
                                        className={`
                                            ${charObj.confidence < 0.5 ? 'text-red-500' : charObj.confidence < 0.8 ? 'text-orange-400' : ''}
                                            hover:bg-blue-500/30 px-[1px] rounded transition-colors duration-150
                                        `}
                                        title={`Char: '${charObj.char}'\nConf: ${charObj.confidence.toFixed(4)}${charObj.alternatives && charObj.alternatives.length > 0 ? '\nAlts: ' + charObj.alternatives.map(a => `'${a.char}': ${a.confidence.toFixed(4)}`).join(', ') : ''}`}
                                    >
                                        {formattedChar}
                                    </span>
                                );
                            })}
                        </span>
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

const TABS = [
  { id: 0, name: "Batch Segmentation", title: "Batch Segmentation" },
  { id: 1, name: "Batch Inference", title: "Batch Inference" },
  { id: 2, name: "Review", title: "Review Tool" },
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
        </div>
      </main>
    </div>
  );
}
