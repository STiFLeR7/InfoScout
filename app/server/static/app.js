// UI state
let latestResult = null;
let isAgentRunning = false;

// Utility: pretty JSON
function pretty(obj){ return JSON.stringify(obj, null, 2); }

// ---------------- Model status ----------------
async function updateModelStatus(){
  const badge = document.getElementById('modelStatus');
  try{
    const res = await fetch('/model_status');
    const data = await res.json();
    const loaded = data?.model_info?.model_loaded === true;
    const running = data?.agent_running === true;
    isAgentRunning = running;
    badge.className = 'status-badge ' + (loaded ? 'status-ok' : 'status-warn');
    badge.textContent = loaded ? '✅ Model Loaded' : (running ? '⚠ Agent Running — model not loaded' : '⚠ Model Not Loaded');
    // toggle start/stop button state
    document.getElementById('startBtn').disabled = running;
    document.getElementById('stopBtn').disabled = !running;
  }catch(e){
    badge.className = 'status-badge status-bad';
    badge.textContent = '❌ Error checking model';
    console.error('model status error', e);
  }
}

// ---------------- Start / Stop / Load ----------------
async function startAgent(){
  const headless = document.getElementById('headlessToggle').checked;
  const mp = document.getElementById('modelPath').value || undefined;
  try{
    document.getElementById('startBtn').disabled = true;
    const res = await fetch('/start_agent', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ headless, model_path: mp })
    });
    if(!res.ok){
      const errText = await res.text();
      alert('Start failed: ' + errText);
    }
  }catch(e){
    console.error(e); alert('Start request failed');
  } finally {
    setTimeout(updateModelStatus, 400); // give small time to spin up
  }
}

async function stopAgent(){
  try{
    document.getElementById('stopBtn').disabled = true;
    await fetch('/stop_agent', { method:'POST' });
  }catch(e){ console.error(e) }
  finally { setTimeout(updateModelStatus, 300) }
}

async function loadModel(){
  const path = document.getElementById('modelPath').value;
  if(!path){ alert('Enter model path'); return; }
  try{
    const res = await fetch('/load_model', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model_path: path })
    });
    if(!res.ok){
      const txt = await res.text();
      alert('Model load failed: ' + txt);
    } else {
      alert('Model load requested. Check status shortly.');
    }
  }catch(e){ console.error(e); alert('Model load error') }
  setTimeout(updateModelStatus, 500);
}

// ---------------- Execute / Render ----------------
async function executeInstruction(){
  const instruction = document.getElementById('instruction').value?.trim();
  if(!instruction){ alert('Enter instruction'); return; }

  setUIExecuting(true);
  try{
    const res = await fetch('/execute', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ instruction })
    });
    const data = await res.json();
    latestResult = data;
    renderResults(data);
    await refreshTasks();
  }catch(err){
    console.error(err);
    document.getElementById('resultsBox').textContent = 'Execution failed: ' + (err.message||err);
  } finally {
    setUIExecuting(false);
  }
}

function renderResults(data){
  const area = document.getElementById('resultsBox');
  if(!data || !data.success){
    area.innerHTML = `<pre>${pretty(data)}</pre>`;
    return;
  }
  const payload = data.data;
  if(!payload || !payload.results){
    area.innerHTML = `<pre>${pretty(data)}</pre>`;
    return;
  }
  // render clickable list + JSON block
  const list = payload.results.map(r => {
    const title = r.title || r.link || '(no title)';
    const link = r.link || '#';
    return `<li class="result-item"><a href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(title)}</a>
            <div class="result-meta">${escapeHtml(r.snippet||'')}</div></li>`;
  }).join('');
  area.innerHTML = `<ol class="results-list">${list}</ol>
                    <details style="margin-top:12px;color:#e6eef8;"><summary>Raw JSON</summary><pre>${pretty(payload)}</pre></details>`;
}

// small escape helper for safety
function escapeHtml(s){
  return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
}

// ---------------- Downloads ----------------
function downloadJSON(){
  if(!latestResult){ alert('No results to download'); return; }
  const blob = new Blob([pretty(latestResult)], {type:'application/json'});
  downloadBlob(blob, 'results.json');
}

function downloadCSV(){
  if(!latestResult || !latestResult.data || !Array.isArray(latestResult.data.results)){ alert('No tabular results'); return; }
  const rows = [['rank','title','link','snippet','source']];
  latestResult.data.results.forEach(r => {
    rows.push([r.rank ?? '', r.title ?? '', r.link ?? '', r.snippet ?? '', r.source ?? '']);
  });
  const csv = rows.map(row => row.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  downloadBlob(blob, 'results.csv');
}
function downloadBlob(blob, filename){
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(url), 2000);
}

// ---------------- Recent tasks ----------------
async function refreshTasks(){
  try{
    const res = await fetch('/recent_tasks');
    const tasks = await res.json();
    const ul = document.getElementById('recentBox');
    if(!tasks.length){ ul.innerHTML = '<li>No recent tasks.</li>'; return; }
    ul.innerHTML = tasks.slice().reverse().map(t => {
      const ts = t.timestamp ? new Date(t.timestamp*1000).toLocaleString() : '';
      const instr = escapeHtml(t.instruction || '');
      return `<li><a href="#" data-instr="${instr}" class="recent-link">${instr}</a><div class="task-meta">${ts}</div></li>`;
    }).join('');
    // attach handlers to re-run when clicked
    document.querySelectorAll('.recent-link').forEach(el=>{
      el.onclick = (ev) => { ev.preventDefault(); const instr = ev.currentTarget.getAttribute('data-instr'); document.getElementById('instruction').value = instr; executeInstruction(); };
    });
  }catch(e){
    console.error('refresh tasks error', e);
  }
}

// ---------------- STT (browser) ----------------
function setupSTT(){
  const mic = document.getElementById('micBtn');
  const input = document.getElementById('instruction');
  const Speech = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  if(!Speech){
    mic.title = 'Speech recognition not supported by your browser';
    mic.disabled = true;
    return;
  }
  const rec = new Speech();
  rec.lang = 'en-US';
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  let recording = false;
  rec.onstart = ()=> { recording = true; mic.textContent = '● recording'; mic.classList.add('recording'); };
  rec.onend = ()=> { recording = false; mic.textContent = '🎤 STT'; mic.classList.remove('recording'); };
  rec.onerror = (evt)=> { console.warn('STT error', evt); recording=false; mic.textContent='🎤 STT'; mic.classList.remove('recording'); };
  rec.onresult = (evt) => {
    if(evt.results && evt.results[0] && evt.results[0][0]) {
      input.value = evt.results[0][0].transcript;
    }
  };
  mic.onclick = () => {
    try {
      if(recording) rec.stop(); else rec.start();
    }catch(e){ console.error(e) }
  };
}

// ---------------- UI helpers ----------------
function setUIExecuting(on){
  const exec = document.getElementById('executeBtn');
  exec.disabled = on;
  exec.textContent = on ? '⏳ Running...' : '🚀 Execute';
}

// ---------------- bind UI ----------------
document.getElementById('startBtn').onclick = startAgent;
document.getElementById('stopBtn').onclick = stopAgent;
document.getElementById('loadModelBtn').onclick = loadModel;
document.getElementById('executeBtn').onclick = executeInstruction;
document.getElementById('downloadJSON').onclick = downloadJSON;
document.getElementById('downloadCSV').onclick = downloadCSV;
document.getElementById('refreshTasks').onclick = refreshTasks;

setupSTT();
updateModelStatus();
refreshTasks();
setInterval(updateModelStatus, 4000);
